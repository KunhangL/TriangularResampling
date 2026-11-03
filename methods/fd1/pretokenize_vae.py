import argparse
import os

import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from common.utils.initialize import instantiate, load_config

RAW_DATA_PATH = "raw_data"

# Per-dataset split files (missing files are skipped). Add a new entry here to
# support a new dataset. HumanML3D tokenizes the full partition union; LDF
# evaluation reads test_min, whose ids are a subset of test.txt.
DATASET_SPLITS = {
    "HumanML3D": ["train.txt", "val.txt", "test.txt"],
    "BABEL":     ["train.txt", "val.txt", "test_min.txt"],
}

# Defaults match methods/fd1/configs/263d/humanml_babel.yaml: 263D features in
# new_joint_vecs/ encoded by the official FloodDiffusion VAE
# (configs/263d/vae_official_263.yaml) into TOKENS_20251030_085836_vae_wan_z4/.
DEFAULT_DATASETS = ["HumanML3D", "BABEL"]
DEFAULT_FEATURE_PATHS = {
    "HumanML3D": "new_joint_vecs",
    "BABEL":     "new_joint_vecs",
}
DEFAULT_TOKEN_TAG = "TOKENS_20251030_085836_vae_wan_z4"


def parse_pretokenize_args():
    """CLI override for which datasets / feature dir / token tag to write.
    Also forwards --config / --override to load_config()."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--override", type=str, nargs="+", default=None,
                   help="cfg overrides as key=value (forwarded to load_config)")
    p.add_argument("--datasets", type=str, default=None,
                   help=f"comma-separated subset of {list(DATASET_SPLITS)}; "
                        f"default {DEFAULT_DATASETS}")
    p.add_argument("--raw_data_path", type=str, default=RAW_DATA_PATH,
                   help="raw-data root containing dataset directories; "
                        f"default {RAW_DATA_PATH!r}")
    p.add_argument("--feature_path", type=str, default=None,
                   help="single feature dir name applied to all datasets "
                        "(e.g. new_joint_vecs_track). If unset, per-dataset "
                        "DEFAULT_FEATURE_PATHS is used.")
    p.add_argument("--token_tag", type=str, default=None,
                   help=f"token folder name (e.g. TOKENS_<run>_<step>); "
                        f"default {DEFAULT_TOKEN_TAG}")
    p.add_argument("--no_recovered", action="store_true",
                   help="skip writing decoded recovered features")
    p.add_argument("--batch_size", type=int, default=1,
                   help="encode batch size (1 = legacy per-motion path); "
                        ">1 enables length-bucketed batching with DataLoader")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for parallel disk read "
                        "(only used when batch_size > 1)")
    p.add_argument("--sanity_check", action="store_true",
                   help="encode a few motions both per-sample and batched; "
                        "abort on mismatch. Recommended on first batched run.")
    p.add_argument("--limit", type=int, default=0,
                   help="process only first N motions per split (0 = all). "
                        "Use for benchmarking the batched throughput.")
    return p.parse_args()


def build_jobs(args):
    """Returns list of (path_files, feature_path, token_path, recovered_path)
    tuples. One tuple per dataset."""
    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets is not None else DEFAULT_DATASETS
    )
    token_tag = args.token_tag or DEFAULT_TOKEN_TAG
    jobs = []
    for ds in datasets:
        if ds not in DATASET_SPLITS:
            raise ValueError(f"Unknown dataset {ds!r}; "
                             f"add it to DATASET_SPLITS first.")
        dataset_root = os.path.join(args.raw_data_path, ds)
        path_files = [os.path.join(dataset_root, s) for s in DATASET_SPLITS[ds]]
        feature_path = args.feature_path or DEFAULT_FEATURE_PATHS[ds]
        recovered_path = (None if args.no_recovered
                          else f"{feature_path}_recovered_{token_tag}")
        jobs.append((path_files, feature_path, token_tag, recovered_path))
    return jobs


def main():
    args = parse_pretokenize_args()

    # Build cfg via load_config(), forwarding --config + --override.
    override_args = None
    if args.override:
        override_args = {}
        for ov in args.override:
            k, v = ov.split("=", 1)
            override_args[k.strip()] = v.strip()
    cfg = load_config(config_path=args.config, override_args=override_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    # vae
    vae_model = instantiate(
        cfg.model.target,
        cfg=None,
        hfstyle=False,
        **cfg.model.params,
    )
    vae_ckpt = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae_model.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae_model.parameters(), decay=cfg.model.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae_model.parameters())
        rank_zero_info(f"Loaded VAE model from {cfg.test_ckpt} with EMA")
    else:
        vae_model.load_state_dict(vae_ckpt["state_dict"], strict=True)
        rank_zero_info(f"Loaded VAE model from {cfg.test_ckpt} w/o EMA")
    vae_model = vae_model.to(device).eval()
    for p_ in vae_model.parameters():
        p_.requires_grad = False

    jobs = build_jobs(args)
    if args.batch_size > 1 and args.sanity_check:
        run_batched_sanity_check(vae_model, device, jobs[0])

    for path_files, feature_path, token_path, recovered_path in jobs:
        tokenize_motion(
            vae_model, path_files, device, cfg, feature_path, token_path,
            recovered_path, batch_size=args.batch_size,
            num_workers=args.num_workers, limit=args.limit,
        )


@torch.no_grad()
def run_batched_sanity_check(vae_model, device, job):
    """Sanity: encode same motions per-sample and as a padded batch; abort on
    significant mismatch on the real (non-padded) latent region."""
    path_files, feature_path, _, _ = job
    feat_dir = os.path.join(os.path.dirname(path_files[0]), feature_path)
    if not os.path.isdir(feat_dir):
        rank_zero_info(f"[sanity] feat_dir not found: {feat_dir}; skipping")
        return
    files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".npy")])[:5]
    if len(files) < 2:
        rank_zero_info("[sanity] not enough files for sanity check")
        return
    arrs = [np.load(os.path.join(feat_dir, f)).astype(np.float32) for f in files]
    downsample = vae_model.downsample_factor

    # Per-sample encode
    singles = []
    for a in arrs:
        x = torch.from_numpy(a).to(device)[None, :, :]
        z = vae_model.encode(x)[0].cpu().numpy()
        singles.append(z)

    # Padded-batch encode
    Tmax = max(a.shape[0] for a in arrs)
    padded = torch.zeros(len(arrs), Tmax, arrs[0].shape[1], dtype=torch.float32)
    for i, a in enumerate(arrs):
        padded[i, :a.shape[0]] = torch.from_numpy(a)
    z_b = vae_model.encode(padded.to(device)).cpu().numpy()

    max_diff = 0.0
    for i, a in enumerate(arrs):
        T_i = a.shape[0]
        t_lat = (T_i + downsample - 1) // downsample
        d = np.abs(singles[i][:t_lat] - z_b[i, :t_lat]).max()
        max_diff = max(max_diff, float(d))
        rank_zero_info(f"[sanity] {files[i]} T={T_i} t_lat={t_lat} "
                       f"max|single-batched|={d:.3e}")
    # Tolerance 5e-3: WAN VAE has T-aggregating norm layers (e.g. GroupNorm
    # over time), so padding zeros shifts norm statistics → real-region output
    # gets ~1e-3 perturbation. 5e-3 is well below typical token magnitude (~1)
    # and Mean/Std aggregated over 30k+ motions averages this out further.
    tol = 5e-3
    if max_diff > tol:
        raise RuntimeError(
            f"[sanity] batched encode diverges from per-sample by "
            f"{max_diff:.3e} > {tol:.0e}; padding pollutes real region. "
            f"Abort batched mode."
        )
    rank_zero_info(f"[sanity] OK, max diff = {max_diff:.3e} (tol {tol:.0e})")


class _MotionFileDataset(Dataset):
    """Reads (mid, feature_np) on worker. Returns None for missing/bad files —
    collator filters them out."""
    def __init__(self, names, feature_dir):
        self.names = names
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        fp = os.path.join(self.feature_dir, f"{name}.npy")
        if not os.path.exists(fp):
            return None
        try:
            arr = np.load(fp, allow_pickle=True).astype(np.float32)
            if np.isnan(arr).any() or np.isinf(arr).any():
                return None
            return name, arr
        except Exception as e:
            rank_zero_info(f"DataLoader error on {name}: {e}")
            return None


def _pad_collate(items):
    """Length-pad to max(T) on right; return (mids, padded[B,Tmax,263], lens[B])."""
    items = [x for x in items if x is not None]
    if not items:
        return None
    mids = [x[0] for x in items]
    arrs = [torch.from_numpy(x[1]) for x in items]
    lens = torch.tensor([a.shape[0] for a in arrs], dtype=torch.long)
    Tmax = int(lens.max().item())
    D = arrs[0].shape[1]
    padded = torch.zeros(len(arrs), Tmax, D, dtype=torch.float32)
    for i, a in enumerate(arrs):
        padded[i, :a.shape[0]] = a
    return mids, padded, lens


@torch.no_grad()
def _encode_batched(vae_model, names, cur_feature_path, cur_token_path,
                    cur_recovered_path, device, batch_size, num_workers,
                    desc):
    """Length-bucketed batched encode. Sorts names by file size (proxy for T)
    so each batch has near-uniform length and minimal padding waste."""
    valid_names = []
    sizes = []
    for n in names:
        fp = os.path.join(cur_feature_path, f"{n}.npy")
        if os.path.exists(fp):
            valid_names.append(n)
            sizes.append(os.path.getsize(fp))
    order = sorted(range(len(valid_names)), key=lambda i: sizes[i])
    ordered_names = [valid_names[i] for i in order]

    ds = _MotionFileDataset(ordered_names, cur_feature_path)
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=_pad_collate, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    downsample = vae_model.downsample_factor
    pbar = tqdm(loader, desc=desc, total=len(loader))
    n_done = 0
    for batch in pbar:
        if batch is None:
            continue
        mids, padded, lens = batch
        padded = padded.to(device, non_blocking=True)
        tokens = vae_model.encode(padded)  # (B, T_lat_max, z_dim)
        if cur_recovered_path is not None:
            recons = vae_model.decode(tokens)  # (B, T_rec, 263)
        tokens_np = tokens.detach().cpu().numpy()
        if cur_recovered_path is not None:
            recons_np = recons.detach().cpu().numpy()
        for i, (mid, T_i) in enumerate(zip(mids, lens.tolist())):
            t_lat = (T_i + downsample - 1) // downsample
            np.save(os.path.join(cur_token_path, f"{mid}.npy"),
                    tokens_np[i, :t_lat])
            if cur_recovered_path is not None:
                # Causal decoder: T_rec = ((t_lat - 1) * downsample + 1)-ish.
                # We cap to actual recon length to avoid out-of-bounds, and
                # cap to T_i to match the legacy per-sample behavior.
                T_rec_max = recons_np.shape[1]
                T_save = min(T_i, T_rec_max)
                np.save(os.path.join(cur_recovered_path, f"{mid}.npy"),
                        recons_np[i, :T_save])
        n_done += len(mids)
    return n_done


@torch.no_grad()
def tokenize_motion(
    vae_model, path_files, device, cfg, feature_path, token_path,
    recovered_path=None, batch_size=1, num_workers=0, limit=0,
):
    for path_file in path_files:
        print(
            f"Processing {path_file} with feature path: {feature_path}, token path: {token_path}, recovered path: {recovered_path}"
        )
        if not os.path.exists(path_file):
            continue
        data_path = os.path.dirname(path_file)
        cur_feature_path = os.path.join(data_path, f"{feature_path}")
        cur_token_path = os.path.join(data_path, f"{token_path}")
        os.makedirs(cur_token_path, exist_ok=True)
        cur_recovered_path = None
        if recovered_path is not None:
            cur_recovered_path = os.path.join(data_path, f"{recovered_path}")
            os.makedirs(cur_recovered_path, exist_ok=True)
        with open(path_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        if limit > 0:
            names = names[:limit]

        if batch_size > 1:
            _encode_batched(
                vae_model, names, cur_feature_path, cur_token_path,
                cur_recovered_path, device, batch_size, num_workers,
                desc=f"batched {os.path.basename(path_file)}",
            )
        else:
            for name in tqdm(names, desc=f"Processing {os.path.basename(path_file)}"):
                single_feature_path = f"{cur_feature_path}/{name}.npy"
                single_token_path = f"{cur_token_path}/{name}.npy"
                if cur_recovered_path is not None:
                    single_recovered_path = f"{cur_recovered_path}/{name}.npy"
                if os.path.exists(single_feature_path):
                    try:
                        feature = np.load(single_feature_path, allow_pickle=True)
                        feature = torch.from_numpy(feature).to(device)[None, :, :]
                        token = vae_model.encode(feature)
                        np.save(single_token_path, token.detach().cpu().numpy()[0])
                        if cur_recovered_path is not None:
                            recovered = vae_model.decode(token)
                            np.save(
                                single_recovered_path,
                                recovered.detach().cpu().numpy()[0],
                            )
                    except Exception as e:
                        rank_zero_info(f"Error processing {name}: {e}")

    # Compute and save mean and std across all tokens
    data_path = os.path.dirname(path_files[0])
    cur_token_path = os.path.join(data_path, token_path)
    # Parallel/online variance aggregation (Chan et al.). Keeping only the
    # 16-D accumulator avoids concatenating every latent from large datasets
    # such as SEED_138 into RAM.
    count = 0
    mean = None
    m2 = None
    skipped = 0
    for f in tqdm(os.listdir(cur_token_path), desc="Loading tokens for mean/std"):
        if f.endswith(".npy"):
            token = np.load(os.path.join(cur_token_path, f))
            if np.any(np.isnan(token)) or np.any(np.isinf(token)):
                skipped += 1
                rank_zero_info(f"Skipping {f} due to NaN/Inf")
                continue
            values = np.asarray(token, dtype=np.float64)
            if values.ndim < 2:
                values = values.reshape(-1, 1)
            else:
                values = values.reshape(-1, values.shape[-1])
            batch_count = int(values.shape[0])
            if batch_count == 0:
                continue
            batch_mean = values.mean(axis=0)
            batch_m2 = np.square(values - batch_mean).sum(axis=0)
            if mean is None:
                count = batch_count
                mean = batch_mean
                m2 = batch_m2
            else:
                if batch_mean.shape != mean.shape:
                    raise ValueError(
                        f"Token dimension changed in {f}: "
                        f"{batch_mean.shape} != {mean.shape}"
                    )
                new_count = count + batch_count
                delta = batch_mean - mean
                mean = mean + delta * (batch_count / new_count)
                m2 = (
                    m2 + batch_m2
                    + np.square(delta) * count * batch_count / new_count
                )
                count = new_count
    if skipped > 0:
        rank_zero_info(f"Skipped {skipped} files with NaN/Inf")

    if count > 0:
        std = np.sqrt(m2 / count)
        mean_path = os.path.join(data_path, f"Mean_{token_path}.npy")
        std_path = os.path.join(data_path, f"Std_{token_path}.npy")
        np.save(mean_path, mean)
        np.save(std_path, std)
        rank_zero_info(f"Saved mean to {mean_path}, shape: {mean.shape}")
        rank_zero_info(f"Saved std to {std_path}, shape: {std.shape}")


if __name__ == "__main__":
    main()
