import os
import random
from typing import Dict, List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from common.utils.initialize import instantiate


class LengthMismatchError(Exception):
    pass


class HumanML3DDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.stream_mode = cfg.data.get("stream_mode", False)
        # Per-split min_length override. Test mode hardcodes 0 (allow any
        # length, since rendering doesn't care). Train and val accept
        # split-specific overrides via {train,val}_min_length keys, falling
        # back to the unified `min_length`. This lets us train with a strict
        # min (e.g. 64 to keep windows clean) while admitting short BABEL
        # test_min motions to the val loader.
        if self.split == "train":
            self.file_list = cfg.data.train_meta_paths
            self.min_length = cfg.data.get(
                "train_min_length", cfg.data.min_length
            )
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
        elif self.split == "val":
            self.file_list = cfg.data.val_meta_paths
            self.min_length = cfg.data.get(
                "val_min_length", cfg.data.min_length
            )
            # Validation may enumerate a complete official split containing
            # motions longer than the model window.  Admit those source files
            # while retaining the configured window length for deterministic
            # token/feature cropping.
            self.max_length = cfg.data.get(
                "val_max_length", cfg.data.max_length
            )
            self.window_length = cfg.data.get(
                "val_window_length", cfg.data.max_length
            )
        elif self.split == "test":
            self.file_list = cfg.data.test_meta_paths
            self.min_length = cfg.data.get("test_min_length", 0)
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
        # Per-split feature_path override. Useful for VAE smoke verification
        # where train uses _track but val/test should use the original
        # _new_joint_vecs (i.e. the distribution the init_ckpt was trained on).
        # Falls back to the unified `feature_path` when no split-specific key
        # is set, preserving the legacy single-path behaviour.
        split_feature_key = f"{split}_feature_path"
        self.feature_path = cfg.data.get(
            split_feature_key, cfg.data.get("feature_path", None)
        )
        self.token_path = cfg.data.get("token_path", None)
        self.text_path = cfg.data.get("text_path", None)
        # Optional per-split extra feature paths. Use case: VAE test-time
        # rendering wants BOTH `new_joint_vecs` (main, GT panel) and
        # `new_joint_vecs_track` (extra, for the track panel + its VAE recon).
        # Schema: dict {name: relative_dir}. Loader saves each as
        # data[f"feature_{name}"] / data[f"feature_{name}_length"].
        # Falls back to unified `extra_feature_paths`, then empty dict.
        split_extras_key = f"{split}_extra_feature_paths"
        extras = cfg.data.get(
            split_extras_key, cfg.data.get("extra_feature_paths", None)
        )
        self.extra_feature_paths = dict(extras) if extras else {}
        self.dataset = self._load_file_list()

        rank_zero_info(f"Loaded {len(self.dataset)} samples from {split} dataset.")

    def _load_file_list(self) -> List[str]:
        """Build a metadata list (paths + lengths) from each text file.

        Lazy-load semantics: features/tokens/text are NOT read into RAM here.
        We only persist file paths plus a peek of feature/token shape[0] so
        length-based filtering still happens at init. Actual `np.load()` /
        text parsing is deferred to `_process()` per sample. This keeps the
        DataLoader-worker fork footprint tiny (no big numpy arrays in the
        dataset list to trigger CoW page accounting).
        """
        dataset = []
        ignored_cnt = 0
        for path in self.file_list:
            if os.path.exists(path):
                data_path = os.path.dirname(path)
                dataset_name = os.path.basename(data_path)
                rank_zero_info(f"Loading {path} ...")
                with open(path, "r") as f:
                    for name in f:
                        name = name.strip()
                        if name:
                            data = {}
                            try:
                                data["name"] = name
                                data["dataset"] = dataset_name
                                ##############################
                                # feature (path + peek length)
                                ##############################
                                if self.feature_path is not None:
                                    feature_path = os.path.join(
                                        data_path, self.feature_path, name + ".npy"
                                    )
                                    feature_length = self._peek_npy_length(feature_path)
                                    if (feature_length < self.min_length
                                            or feature_length > self.max_length):
                                        raise LengthMismatchError(
                                            "Feature length out of bounds, skip it."
                                        )
                                    data["feature_path"] = feature_path
                                    data["feature_length"] = feature_length
                                ##############################
                                # extra features (path only — peek not needed,
                                # crop window comes from main feature)
                                ##############################
                                for extra_name, extra_path in self.extra_feature_paths.items():
                                    extra_full = os.path.join(
                                        data_path, extra_path, name + ".npy"
                                    )
                                    if not os.path.exists(extra_full):
                                        raise FileNotFoundError(
                                            f"extra feature missing: {extra_full}"
                                        )
                                    data[f"feature_{extra_name}_path"] = extra_full
                                ##############################
                                # token (path only — load_token already verifies
                                # at __getitem__ time)
                                ##############################
                                if self.token_path is not None:
                                    token_path = os.path.join(
                                        data_path,
                                        self.token_path,
                                        name + ".npy",
                                    )
                                    if not os.path.exists(token_path):
                                        raise FileNotFoundError(
                                            f"token missing: {token_path}"
                                        )
                                    data["token_path"] = token_path
                                ##############################
                                # text (path only)
                                ##############################
                                if self.text_path is not None:
                                    text_path = os.path.join(
                                        data_path, self.text_path, name + ".txt"
                                    )
                                    if not os.path.exists(text_path):
                                        raise FileNotFoundError(
                                            f"text missing: {text_path}"
                                        )
                                    data["text_path"] = text_path
                                ##############################
                                # end
                                ##############################
                                dataset.append(data)
                            except LengthMismatchError:
                                ignored_cnt += 1
                                pass
                            except Exception as e:
                                rank_zero_info(f"Error loading data for {name}: {e}")
                                pass
                            if self.cfg.debug and len(dataset) >= 100:
                                rank_zero_info(f"debug mode, break at {len(dataset)}")
                                break
        if ignored_cnt > 0:
            rank_zero_info(f"Ignored {ignored_cnt} samples due to length mismatch.")
        if len(dataset) == 0:
            rank_zero_info(
                f"No data found in {self.file_list}. Please check the file paths and ensure they are correct."
            )
        else:
            for i in range(3):
                tmp = random.choice(dataset)
                rank_zero_info(
                    f"Random data {tmp['name']}: length={tmp.get('feature_length', 'N/A')}"
                )
        return dataset

    @staticmethod
    def _peek_npy_length(path: str) -> int:
        """Read a .npy file's shape[0] without loading data into RAM.

        `np.load(mmap_mode='r')` parses the header (~hundreds of bytes) and
        returns a memmap object; we only touch `.shape`, so no array data is
        faulted in. The memmap is dropped immediately.
        """
        arr = np.load(path, mmap_mode="r")
        return int(arr.shape[0])

    def load_feature(self, feature_path: str) -> np.ndarray:
        feature = np.load(feature_path).astype(np.float32)
        if np.isnan(feature).any():
            raise ValueError("NaN values found in feature, skip it.")
        if feature.shape[0] < self.min_length or feature.shape[0] > self.max_length:
            raise LengthMismatchError("Feature length out of bounds, skip it.")
        return feature

    def load_token(self, token_path: str) -> np.ndarray:
        token = np.load(token_path)
        if np.isnan(token).any():
            raise ValueError("NaN values found in token, skip it.")
        return token

    def load_text(self, text_path: str) -> List[Dict]:
        text_data = []
        with open(text_path, "r") as text_f:
            lines = text_f.readlines()
            for line in lines:
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                t_tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = t_tokens
                text_dict["f_tag"] = f_tag
                text_dict["to_tag"] = to_tag

                text_data.append(text_dict)
        return text_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        try:
            return self._process(data)
        except Exception as e:
            # Lazy-load may fail mid-batch (NaN, missing file, IO error). Skip
            # the sample — collate_fn filters None and shrinks the batch by 1
            # rather than crashing the whole run.
            rank_zero_info(
                f"Error processing {data.get('name', '?')}: {e}; skipping."
            )
            return None

    def _process(self, data):
        # Crop policy (aligned with BABEL, datasets/babel.py:_process):
        # When token_path is set, token-space crop is the source of truth
        # and feature is sliced by feature_start = token_start * 4. This
        # keeps token / feature / text indices time-aligned, which is
        # required for any downstream consumer that uses token_start (PHC
        # GT-RSI maps it to motion_lib time via t_start_sec).
        # When only feature_path is set (e.g. VAE-only training), fall back
        # to independent feature crop via process_feature().
        output = {}
        output["dataset"] = data["dataset"]
        output["name"] = data["name"]
        downsample_factor = 4

        ##############################
        # token (lazy-loaded) -- crop window source of truth when present
        ##############################
        token_start = 0
        if "token_path" in data:
            token = self.load_token(data["token_path"])
            token_length = token.shape[0]
            max_token_length = self.window_length // downsample_factor
            if token_length > max_token_length:
                if self.split == "train":
                    token_start = random.randint(
                        0, token_length - max_token_length
                    )
                else:
                    token_start = 0
                token = token[token_start : token_start + max_token_length]
                token_length = max_token_length
            output["token"] = token
            output["token_length"] = token_length
            output["token_start"] = token_start

        ##############################
        # feature (lazy-loaded) -- aligned to token crop window when token is present
        ##############################
        feature_start = 0
        feature_length = None
        if "feature_path" in data:
            feature_arr = self.load_feature(data["feature_path"])
            if "token_path" in data:
                feature_start = token_start * downsample_factor
                feature_end = min(
                    feature_start + self.window_length, feature_arr.shape[0]
                )
                feature = feature_arr[feature_start:feature_end]
                feature_length = feature.shape[0]
            else:
                feature, feature_length, feature_start = (
                    self.process_feature(feature_arr)
                )
            output["feature"] = feature
            output["feature_length"] = feature_length
        ##############################
        # extra features (test-time multi-render, e.g. _track for 4-panel composite)
        # Apply the SAME crop window as the main feature so panels align in
        # time. When main wasn't cropped (window_length >= feature_len, e.g.
        # test split), feature_start=0 and we just pass through truncated to
        # the main length.
        ##############################
        for extra_name in list(self.extra_feature_paths.keys()):
            path_key = f"feature_{extra_name}_path"
            if path_key not in data:
                continue
            src = self.load_feature(data[path_key])
            out_key = f"feature_{extra_name}"
            if feature_length is None:
                # No main feature — take full extra
                output[out_key] = src
                output[f"{out_key}_length"] = src.shape[0]
                continue
            end = feature_start + feature_length
            cropped = src[feature_start:end]
            # If extra is shorter than the requested window (rare — extras
            # generally have the same T as the main), truncate to its actual
            # length and adjust the recorded length so collate doesn't pad
            # with zeros that get fed into the VAE.
            output[out_key] = cropped
            output[f"{out_key}_length"] = cropped.shape[0]
        ##############################
        # text (lazy-loaded from path)
        ##############################
        if "text_path" in data:
            text_data = self.load_text(data["text_path"])
            text_dict = random.choice(text_data)
            text, text_tokens, f_tag, to_tag = self.process_text_dict(text_dict)
            if self.stream_mode:
                output["text"] = [text]
                output["text_tokens"] = text_tokens
                output["feature_text_end"] = [output["feature_length"]]
                output["token_text_end"] = [output["token_length"]]
            else:
                output["text"] = text
                output["text_tokens"] = text_tokens
                # Always emit feature_text_end / token_text_end (single-element
                # lists, same value as in stream_mode). Required when paired
                # with BABEL in MultiDataset: multi.collate_fn iterates
                # batch[0].keys() and accesses each key on every sample. BABEL
                # always sets these keys; if batch[0] is BABEL and a HumanML3D
                # sample comes next, missing key → KeyError (job 8858436
                # crashed at training step 2).
                output["feature_text_end"] = [output["feature_length"]]
                output["token_text_end"] = [output["token_length"]]
        return output

    def process_feature(self, feature):
        """Independent feature-space crop. Only used when token_path is absent
        (e.g. VAE-only training). When token_path is set, _process slices
        feature by token_start * 4 instead so token / feature stay aligned."""
        feature_len = feature.shape[0]
        start = 0
        # if the motion is longer than window_length, randomly crop a window_length clip
        if feature_len > self.window_length:
            start = random.randint(0, feature_len - self.window_length)
            feature = feature[start : start + self.window_length]
            feature_len = self.window_length
        return feature, feature_len, start

    def process_text_dict(self, text_dict: Dict):
        text = text_dict["caption"]
        text_tokens = text_dict["tokens"]
        f_tag = text_dict["f_tag"]
        to_tag = text_dict["to_tag"]
        return text, text_tokens, f_tag, to_tag


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    # Mirror datasets.multi.collate_fn — prefix-based dispatch so any
    # `feature_<extra>` / `feature_<extra>_length` pair (e.g. feature_track) is
    # padded / stacked correctly. `feature_text_end` / `token_text_end` are
    # per-sample int lists (variable length, consumed as list-of-lists by
    # diffusion_forcing_wan); not features — must passthrough as list[list].
    _passthrough_feature_meta = {"feature_text_end", "token_text_end"}

    def _is_seq_key(k):
        if k in _passthrough_feature_meta:
            return False
        return k == "feature" or k == "token" or (
            k.startswith("feature_") and not k.endswith("_length")
        )

    def _is_len_key(k):
        return k in ("feature_length", "token_length") or (
            k.startswith("feature_") and k.endswith("_length")
        )

    for key in keys:
        if _is_seq_key(key):
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif _is_len_key(key):
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            output[key] = [b[key] for b in batch]
    return output


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    dataset = HumanML3DDataset(cfg, split="val")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.train_bs, shuffle=True, collate_fn=collate_fn
    )
    for idx, data in tqdm(enumerate(dataloader)):
        pass
