import os

import numpy as np
import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info


def convert_motion_138_to_263(*args, **kwargs):
    """138D (MEI) -> 263D transcoding is not part of this release."""
    raise NotImplementedError(
        "138D->263D conversion needs the SMPL-H FK / physics tree, which is not included; "
        "train and evaluate with 263D features.")


from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from common.metrics.t2m import T2MMetrics
from common.utils.initialize import (
    compare_statedict_and_parameters,
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from methods.fd1.utils.lightning_module import BasicLightningModule
from common.utils.motion_process import recover_joint_positions_263
from methods.fd1.utils.visualize import (  # evaluate_video
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def initialize_metrics(self):
        # vae
        self.vae = instantiate(
            target=self.cfg.test_vae.target,
            cfg=None,
            hfstyle=False,
            **self.cfg.test_vae.params,
        )
        test_vae_ckpt = self.cfg.get("test_vae_ckpt", None)
        if test_vae_ckpt:
            vae_ckpt = torch.load(
                test_vae_ckpt, map_location="cpu", weights_only=False
            )
            if "ema_state" in vae_ckpt:
                self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
                self.vae_ema = ExponentialMovingAverage(
                    self.vae.parameters(), decay=self.cfg.test_vae.ema_decay
                )
                self.vae_ema.load_state_dict(vae_ckpt["ema_state"])
                self.vae_ema.copy_to(self.vae.parameters())
                rank_zero_info(f"Loaded VAE model from {test_vae_ckpt} with EMA")
            else:
                self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
                rank_zero_info(f"Loaded VAE model from {test_vae_ckpt} w/o EMA")
        else:
            rank_zero_info(
                f"Using {self.cfg.test_vae.target} without a VAE checkpoint"
            )

        compare_statedict_and_parameters(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )

        # VAE is used only for inference (validate/test decode + resampler
        # encode/decode in `_phc_*_path`). It must NOT be in the DDP grad
        # graph: with requires_grad=True its params get registered by DDP but
        # never receive grad in training_step → "parameters that were not
        # used in producing the loss" RuntimeError when find_unused=False.
        # Freezing also lets us drop find_unused_parameters=True for ~5-10%
        # backward speedup. T2MMetrics already freezes its sub-encoders.
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # metric models
        self.recover_dim = self.cfg.metrics.dim
        self.t2m_metrics = T2MMetrics(self.cfg.metrics.t2m)

        # Optional physics/smoothness metrics (PJ/AUJ + CLoSD joint-space
        # skating/foot_sliding/penetration/floating). Bucketed per dataset so
        # the BABEL + HumanML3D split in ldf_stream.yaml is reported separately.
        if self.cfg.metrics.get("physics", None) is not None:
            from methods.fd1.metrics.physics import PhysicsMetrics
            self.physics_metrics = PhysicsMetrics(self.cfg.metrics.physics)
        else:
            self.physics_metrics = None

    def setup(self, stage):
        # Lightning fires setup once per stage (validate/fit/test/predict) and
        # only after the trainer is attached and the DDP process group is up —
        # so this is where rank/trainer-aware init that can't run in __init__
        # belongs. init_ckpt is NOT here: it's a plain weights load that main()
        # calls explicitly before trainer.validate() / trainer.fit(), so it's
        # in scope for the pre-fit sanity metric too.
        # Resampler (PHC) is training-forward-only (fit-only); the naive
        # ["model"] path is lazy-built inside DiffForcingWanModel.forward, so
        # only the PHC-backed case needs construction here.
        if stage == "fit" and not getattr(self, "_fit_setup_done", False):
            self._fit_setup_done = True
            self._maybe_build_phc_resampler()
            self._wire_debug_viz()

    def load_init_ckpt(self):
        """Weights-only LDF init from a pre-trained ckpt (NOT a full Lightning
        resume). Call once before trainer.validate()/trainer.fit() — this way
        the pre-fit sanity metric is measured on the init weights, not random
        ones. Used e.g. to start resampling-forcing fine-tune from a converged
        baseline without dragging in that ckpt's PL-version-specific loops /
        optimizer / lr_scheduler state (see notes.md Phase E — ckpt trained
        under PL 2.6, we run PL 2.3/2.4; also lr schedule resets)."""
        init_ckpt = self.cfg.get("init_ckpt", None)
        if not init_ckpt:
            return
        rank_zero_info(f"[init_ckpt] loading LDF weights from {init_ckpt}")
        ldf_ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        init_ckpt_format = self.cfg.get("init_ckpt_format", None)
        state_dict = ldf_ckpt["state_dict"]
        if init_ckpt_format == "flooddiffusion_official":
            # The released checkpoint predates the descriptive norm-layer
            # rename in this repository and does not contain the later
            # latent mean/std identity buffers. The backbone parameter order
            # and tensor shapes are otherwise identical.
            state_dict = {
                key.replace(".norm3.", ".cross_attn_norm_layer."): value
                for key, value in state_dict.items()
            }
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if missing != {"mean", "std"} or unexpected:
                raise RuntimeError(
                    "Official FloodDiffusion checkpoint conversion produced "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            rank_zero_info(
                "[init_ckpt] converted official FloodDiffusion keys "
                "(.norm3 -> .cross_attn_norm_layer); mean/std remain identity"
            )
        else:
            self.model.load_state_dict(state_dict, strict=True)
        if "ema_state" in ldf_ckpt:
            # Load into self.ema (the module's persistent EMA) — NOT a tmp
            # object. validation_step wraps forward in
            # `with self.ema.average_parameters(self.model.parameters()):`,
            # which swaps in self.ema.shadow_params; writing to a tmp_ema
            # leaves self.ema at random-init and validation runs on garbage.
            self.ema.load_state_dict(ldf_ckpt["ema_state"])
            self.ema.copy_to(self.model.parameters())
            rank_zero_info(
                "[init_ckpt] EMA state loaded into self.ema and copied to model "
                "(model + ema shadow both carry the EMA-averaged weights)"
            )
        else:
            rank_zero_info("[init_ckpt] no ema_state in ckpt; using raw state_dict")
        # Restore mean/std buffers from config paths. EMA.copy_to() only touches
        # parameters(), not buffers, so this is safe to do after EMA load.
        # Needed when init_ckpt was trained on a different token space (e.g.
        # old new_joint_vecs vs current _track tokens): strict=True loads ckpt's
        # stale buffer values, which we must override with the current npy files.
        if getattr(self.model, "mean_path", None):
            self.model.mean.copy_(
                torch.from_numpy(np.load(self.model.mean_path)).float()
            )
            rank_zero_info(f"[init_ckpt] restored mean buffer from {self.model.mean_path}")
        if getattr(self.model, "std_path", None):
            self.model.std.copy_(
                torch.from_numpy(np.load(self.model.std_path)).float()
            )
            rank_zero_info(f"[init_ckpt] restored std buffer from {self.model.std_path}")

    def _maybe_build_phc_resampler(self):
        mp = self.cfg.model.params
        if mp.get("resampling_ratio", 0.0) <= 0:
            return
        resample_source = mp.get("resample_source", ["model"])
        if isinstance(resample_source, str):
            resample_source = [resample_source]
        else:
            resample_source = list(resample_source)
        if "phc" not in resample_source:
            return  # pure ["model"] is lazy-built in DiffForcingWanModel.forward

        from methods.fd1.models.resampler import Resampler
        phc_cfg = mp.get("phc", None)
        assert phc_cfg is not None, (
            'resample_source containing "phc" requires '
            "model.params.phc config block"
        )
        # Per-rank device pinning for the physics backend (not included in
        # this release): force the backend onto cuda:0 of the visible GPU.
        phc_cfg_dict = dict(phc_cfg)
        _dev = phc_cfg_dict.get("device", "cpu")
        if isinstance(_dev, str) and _dev.startswith("cuda"):
            phc_cfg_dict["device"] = "cuda:0"
            phc_cfg_dict["device_id"] = 0

        # First-time motion_lib_cache build is ~20 minutes and writes a single
        # ~13.2 GB file. To keep multi-rank from racing on the same path, rank-0
        # builds (or restores) first, then barrier; other ranks see the warm
        # cache and fast-restore (~10s). When cache already exists, rank-0
        # also fast-restores, so steady state is just N parallel reads.
        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        world_size = self.trainer.world_size

        def _build():
            rank_zero_info(
                f"[resampler] rank={self.trainer.global_rank} "
                f"constructing PHC-backed Resampler "
                f"(device={phc_cfg_dict.get('device','cpu')}, "
                f"device_id={phc_cfg_dict.get('device_id',0)}, "
                f"world_size={world_size})"
            )
            self.model.resampler = Resampler(
                sources=resample_source,
                shift=mp.get("resample_shift", 0.6),
                resample_len=mp.get("resample_len", 5),
                prediction_type=mp.prediction_type,
                time_scheduler=self.model.time_scheduler,
                time_embedding_scale=self.model.time_embedding_scale,
                clamp_mode=mp.get("resample_clamp_mode", "frontier"),
                vae=self.vae,
                phc_config=phc_cfg_dict,
            )

        if is_distributed and world_size > 1:
            if self.trainer.is_global_zero:
                _build()
            torch.distributed.barrier()
            if not self.trainer.is_global_zero:
                _build()
            torch.distributed.barrier()
        else:
            _build()
        rank_zero_info(
            f"[resampler] PHC Resampler attached to model "
            f"(sources={resample_source}, "
            f"anchor_source={getattr(self.model.resampler, 'phc_anchor_source', 'decoded_138')})"
        )

    def _wire_debug_viz(self):
        """Wire training-time debug viz dump (rank0, periodic). Stays inert when
        cfg.debug_viz is absent or dump_every <= 0. Runs for BOTH no-PHC and
        PHC paths — extracted out of _maybe_build_phc_resampler so no-PHC
        training (which early-returns from that method) still gets viz."""
        dbg_cfg = self.cfg.get("debug_viz", None)
        if dbg_cfg is not None:
            dump_every = int(dbg_cfg.get("dump_every", 0) or 0)
        else:
            dump_every = 0
        self.model._dbg_train_dump_every = dump_every
        self.model._dbg_train_n_sample_per_dataset = (
            int(dbg_cfg.get("n_sample_per_dataset", 1)) if dbg_cfg is not None else 1
        )
        self.model._dbg_train_dataset_filter = (
            dbg_cfg.get("dataset_filter", None) if dbg_cfg is not None else None
        )
        self.model._dbg_train_require_full_resample = (
            bool(dbg_cfg.get("require_full_resample", False))
            if dbg_cfg is not None else False
        )
        self.model._dbg_train_force_context_end = (
            int(dbg_cfg.get("force_context_end", 0) or 0)
            if dbg_cfg is not None else 0
        )
        self.model._dbg_train_force_context_end_by_name = (
            dict(dbg_cfg.get("force_context_end_by_name", {}) or {})
            if dbg_cfg is not None else {}
        )
        # Text directories per dataset for caption banner under the dbgviz mp4.
        # Format: {"HumanML3D": "/path/to/texts", "BABEL": "/path/to/texts"}.
        self.model._dbg_train_text_dirs = (
            dict(dbg_cfg.get("text_dirs", {}) or {})
            if dbg_cfg is not None else {}
        )
        self.model._dbg_train_save_dir = self.cfg.save_dir
        # Only global-rank-0 dumps debug_viz. Resolved here (setup runs after
        # trainer attach + DDP group init, so is_global_zero is authoritative)
        # and stored as a plain bool: forward() is an nn.Module and can't read
        # self.trainer, while os.environ["LOCAL_RANK"] is 0 in EVERY process
        # under the PHC multi-node-single-GPU launch — reading it made all N
        # ranks dump (N× duplicate files). is_global_zero is correct for both
        # the no-PHC native DDP and the PHC multi-node paths.
        self.model._dbg_train_is_rank0 = bool(self.trainer.is_global_zero)
        # Per-chain wall-clock render budget (see render_phc_chain): caps a
        # single chain's render time so a near-full sequence can't run for ~1h
        # and trip the NCCL collective timeout / abort DDP.
        self.model._dbg_train_max_render_seconds = (
            int(dbg_cfg.get("max_render_seconds", 180)) if dbg_cfg is not None else 180
        )
        # Bypass nn.Module.__setattr__ submodule auto-registration: assigning a
        # nn.Module to an attribute of self.model would otherwise add the VAE's
        # parameters to self.model.parameters(), breaking the EMA's strict
        # length check at the first mid-train validation (shadow_params was
        # sized in __init__ before this assignment).
        object.__setattr__(self.model, "_dbg_train_vae", self.vae)
        # Shared SMPLMeshRenderer for dbgviz mesh panels (cold init ~2-5s,
        # so build once when dbgviz is enabled and reuse across all dumps).
        if dump_every > 0:
            from methods.fd1.utils.render_mesh import SMPLMeshRenderer
            object.__setattr__(
                self.model, "_dbg_train_mesh_renderer",
                SMPLMeshRenderer(gender="neutral"),
            )
        else:
            self.model._dbg_train_mesh_renderer = None
        if dump_every > 0 and self.trainer.is_global_zero:
            rank_zero_info(
                f"[debug_viz] enabled: dump_every={dump_every} "
                f"n_sample_per_dataset={self.model._dbg_train_n_sample_per_dataset} "
                f"force_context_end={self.model._dbg_train_force_context_end or None} "
                f"save_dir={self.model._dbg_train_save_dir}/debug_viz"
            )

    def _step(self, batch, is_training=True):
        # Resampling forcing guard in DiffForcingWanModel.forward reads this to
        # decide whether warmup_steps has passed.
        if is_training:
            batch["global_step"] = self.global_step
        out = self.model(batch)
        return out

    def update_metrics(self, batch, generated=None):
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        generated = output["generated"]
        ground_truth_token = batch["token"]
        gt_token_length = batch["token_length"]
        ground_truth_feature = batch["feature"]
        gt_feature_length = batch["feature_length"]
        text_tokens = batch["text_tokens"]
        dataset_ids = batch.get("dataset", None)  # per-sample dataset name
        # Per-sample list of cumulative segment-end frame indices. BABEL sets
        # multiple entries per sample (text segments); HumanML3D sets a single
        # [feature_length] entry. Used by PhysicsMetrics to slice transition
        # windows for PJ/AUJ on multi-segment samples.
        text_ends_list = batch.get("feature_text_end", None)

        for i in range(len(generated)):
            # Decode motion (force fp32 — Conv3d bf16 is catastrophically slow for small spatial dims)
            with torch.autocast("cuda", enabled=False):
                single_generated = generated[i]
                decoded_single_generated = self.vae.decode(
                    single_generated[None, :].float().to(self.device)
                )[0]
                decoded_single_generated = decoded_single_generated.float().to(
                    self.device
                )
            text_tokens_single = text_tokens[i]
            # The evaluator is HumanML3D-trained, so HumanML3D remains the
            # default allow-list. Closely aligned datasets (for example
            # SEED_138) can opt in explicitly without pooling unrelated BABEL
            # samples into the same FID statistics.
            t2m_dataset_ids = set(
                self.cfg.metrics.get("t2m_dataset_ids", ["HumanML3D"])
            )
            run_t2m = (
                dataset_ids is None or str(dataset_ids[i]) in t2m_dataset_ids
            )
            # T2M evaluator is HumanML3D-263d-trained.  When the model operates
            # in 138d feature space (cfg.metrics.dim==138), we transcode the
            # generated + reference motions through SMPL-H FK back into 263d
            # before feeding T2MMetrics.  See utils/math/mei_to_263.py.
            needs_138_to_263 = int(self.cfg.metrics.dim) == 138
            def _to_t2m_feat(feat_t):
                if not needs_138_to_263:
                    return feat_t, int(feat_t.shape[0])
                metric_fps = int(self.cfg.metrics.get("fps", 20))
                arr_263 = convert_motion_138_to_263(
                    feat_t.detach().cpu().numpy(),
                    fps=metric_fps,
                    device=str(self.device),
                )
                t = torch.from_numpy(arr_263).float().to(self.device)
                return t, int(t.shape[0])

            if run_t2m and self.cfg.metrics.t2m.fid_target == "vae":
                # Decode ground truth
                with torch.autocast("cuda", enabled=False):
                    single_gt_r = ground_truth_token[i][: gt_token_length[i]]
                    decoded_single_gt_r = self.vae.decode(
                        single_gt_r[None, :].float().to(self.device)
                    )[0]
                    decoded_single_gt_r = decoded_single_gt_r.float().to(self.device)
                rst_t2m, len_rst = _to_t2m_feat(decoded_single_generated)
                ref_t2m, len_ref = _to_t2m_feat(decoded_single_gt_r)
                self.t2m_metrics.update(
                    feats_rst=rst_t2m[None, ...],
                    feats_ref=ref_t2m[None, ...],
                    lengths_rst=[len_rst],
                    lengths_ref=[len_ref],
                    text_tokens=[text_tokens_single],
                )
            elif run_t2m:
                # Original ground truth
                single_gt_o = ground_truth_feature[i]
                decoded_single_gt_o = single_gt_o[: gt_feature_length[i], :].to(
                    self.device
                )
                decoded_single_gt_o = decoded_single_gt_o.float().to(self.device)
                rst_t2m, len_rst = _to_t2m_feat(decoded_single_generated)
                ref_t2m, len_ref = _to_t2m_feat(decoded_single_gt_o)
                self.t2m_metrics.update(
                    feats_rst=rst_t2m[None, ...],
                    feats_ref=ref_t2m[None, ...],
                    lengths_rst=[len_rst],
                    lengths_ref=[len_ref],
                    text_tokens=[text_tokens_single],
                )

            # Physics / smoothness metrics on the generated motion only.
            # recover_joint_positions_263 returns (T, 22, 3) numpy (y-up).
            # PhysicsMetrics silently skips samples for datasets not in
            # cfg.metrics.physics.datasets; configs that omit a dataset (e.g.
            # ldf_stream skipping HumanML3D PJ/AUJ) won't crash here.
            if (self.physics_metrics is not None and dataset_ids is not None
                    and dataset_ids[i] in self.physics_metrics.dataset_ids):
                gen_np = decoded_single_generated.detach().cpu().numpy()
                joints_t223 = recover_joint_positions_263(gen_np, 22)  # (T, 22, 3) np
                joints_1t = torch.from_numpy(joints_t223).unsqueeze(0)  # (1, T, 22, 3)
                te_i = None
                if text_ends_list is not None and i < len(text_ends_list):
                    raw = text_ends_list[i]
                    if raw is not None:
                        te_i = [int(t) for t in raw]
                self.physics_metrics.update(
                    joints_1t,
                    lengths=[int(joints_t223.shape[0])],
                    dataset_ids=[dataset_ids[i]],
                    text_ends=[te_i],
                )
        return

    def compute_metrics(self):
        # Skip T2M compute when no HumanML3D samples were seen (e.g. eval
        # batches happened to all come from BABEL, which is skipped by the
        # run_t2m guard in update_metrics). Without this guard, TorchMetrics
        # emits "compute called before update" warning and logs meaningless
        # zeros. T2MMetrics.compute() has its own empty-state fallback but it
        # doesn't prevent the framework-level warning.
        if self.t2m_metrics.recmotion_embeddings:
            t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in t2m_output.items():
                self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=False)

        if self.physics_metrics is not None:
            phys_output = self.physics_metrics.compute(
                sanity_flag=self.trainer.sanity_checking
            )
            for key, value in phys_output.items():
                self.log(f"metrics/physics/{key}", value, sync_dist=False)
            # Also emit a compact rank-0 summary to stdout: when wandb is off
            # (debug mode) self.log() has no visible side effect, so this is
            # the only way the values surface for human review.
            if phys_output:
                summary = " ".join(
                    f"{k}={v:.4f}" for k, v in sorted(phys_output.items())
                )
                rank_zero_info(f"[physics_metrics step={self.global_step}] {summary}")

    def _phc_viz_panels(self, single_generated):
        """PHC-trained models only: correct one generated latent with the live
        PHC backend, return (phcraw_138, phc_decoded_138) numpy arrays.

        phcraw       = controller's physical truth (tracked mei138, no VAE).
        phc_decoded  = whole-sequence decode(encode(phcraw)) — the corrected
                       latent decoded in one shot (one VAE round-trip).
        Returns (None, None) when PHC is unavailable (e.g. resampler not yet
        lazily built at pre-fit validate, or no-PHC training) or on failure, so
        callers fall back to the plain 2-column composite.
        """
        rs = getattr(self.model, "resampler", None)
        if (rs is None or getattr(rs, "_phc_backend", None) is None
                or "phc" not in (getattr(self.model, "resample_source", []) or [])):
            return None, None
        try:
            # (T,C) denormalized -> normalized model latent buffer (C,T,1,1).
            g = single_generated.float().to(self.device)
            buf = ((g - self.model.mean) / self.model.std)
            buf = buf.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()
            T = int(buf.shape[1])
            rs.reset_inference_state()
            with torch.autocast("cuda", enabled=False):
                for ce in range(1, T + 1):  # correct one token per step
                    rs.apply_inference_correction(buf, ce, min_rs_start=0)
                phcraw = rs.get_corrected_trajectory(0)
                if phcraw is None:
                    return None, None
                corr = (buf.squeeze(-1).squeeze(-1).transpose(0, 1)
                        * self.model.std + self.model.mean)
                phc_decoded = self.vae.decode(corr[None].float())[0]
            return (phcraw.float().cpu().numpy(),
                    phc_decoded.float().cpu().numpy())
        except Exception as e:
            rank_zero_info(f"[phc-viz] correction failed: {e}")
            return None, None

    def update_test(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        generated = output["generated"]
        text = output["text"]
        # Save motion
        generated_id = batch["name"]  # [batch_size]
        dataset_id = batch["dataset"]  # [batch_size]
        feature_text_end = batch.get(
            "feature_text_end", None
        )  # [batch_size, feature_length] or None

        # Decode motion to latent space
        # NOTE: inside the to() function, if will check current_tensor.device == target_device, then run this in each loop is fine.
        for i in range(len(generated)):
            single_generated = generated[i]
            single_generated_id = generated_id[i]
            single_dataset_id = dataset_id[i]
            single_text = text[i]
            if feature_text_end is not None:
                single_feature_text_end = feature_text_end[i]
                frames = np.array(single_feature_text_end)
            else:
                frames = None
            try:
                with torch.autocast("cuda", enabled=False):
                    decoded_single_generated = self.vae.decode(
                        single_generated[None, :].float().to(self.device)
                    )[0]
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text", exist_ok=True
                )
                with open(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text/{single_generated_id}.txt",
                    "w",
                ) as f:
                    f.write(single_text)
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token/{single_generated_id}.npy",
                    single_generated.float().cpu().numpy(),
                )
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature/{single_generated_id}.npy",
                    decoded_single_generated.float().cpu().numpy(),
                )
                # PHC test-viz panels (model+phc only; no-op otherwise).
                phcraw, phc_decoded = self._phc_viz_panels(single_generated)
                if phcraw is not None:
                    for sub, arr in (("phcraw", phcraw), ("phc_decoded", phc_decoded)):
                        os.makedirs(
                            f"{self.cfg.save_dir}/{single_dataset_id}/{sub}",
                            exist_ok=True,
                        )
                        np.save(
                            f"{self.cfg.save_dir}/{single_dataset_id}/{sub}/{single_generated_id}.npy",
                            arr,
                        )
                # Save text_end if available
                if frames is not None:
                    os.makedirs(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames", exist_ok=True
                    )
                    np.save(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames/{single_generated_id}.npy",
                        frames,
                    )
            except Exception as e:
                rank_zero_info(
                    f"Error in saving motion {single_generated_id} of dataset {single_dataset_id}: {e}"
                )

        return {"output": output}

    def process_test_results(self):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        for dataset_id in os.listdir(self.cfg.save_dir):
            feature_dir = f"{self.cfg.save_dir}/{dataset_id}/feature"
            if not os.path.exists(feature_dir):
                continue
            # render video and save
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=feature_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video",
                    render_setting=self.cfg.test_setting,
                    frames_dir=f"{self.cfg.save_dir}/{dataset_id}/frames",
                )

                # On-the-fly GT panels: render any 263d npy folders listed in
                # `compare_motion_folders` (raw_data subdirs) into per-cycle
                # mp4 dirs and feed them as compare panels alongside any
                # pre-rendered `compare_folders`. Staging via symlink keeps
                # the per-cycle render to just the test_min subset.
                ds_cfg = self.cfg.test_setting.get(dataset_id, {}) or {}
                cmp_motion_dirs = ds_cfg.get("compare_motion_folders", None) or []
                rendered_cmp_dirs = []
                if cmp_motion_dirs:
                    ids_this_cycle = [
                        os.path.splitext(f)[0]
                        for f in os.listdir(feature_dir) if f.endswith(".npy")
                    ]
                    for i, mdir in enumerate(cmp_motion_dirs):
                        if not os.path.isdir(mdir):
                            rank_zero_info(
                                f"[render] compare_motion_folders[{i}] missing: {mdir}"
                            )
                            continue
                        staging = f"{self.cfg.save_dir}/{dataset_id}/gt_npy_{i}"
                        out_vid = f"{self.cfg.save_dir}/{dataset_id}/gt_video_{i}"
                        os.makedirs(staging, exist_ok=True)
                        for mid in ids_this_cycle:
                            src = os.path.abspath(os.path.join(mdir, f"{mid}.npy"))
                            dst = os.path.join(staging, f"{mid}.npy")
                            if os.path.exists(src) and not os.path.exists(dst):
                                try:
                                    os.symlink(src, dst)
                                except OSError:
                                    import shutil
                                    shutil.copy2(src, dst)
                        render_video(
                            motion_dir=staging,
                            save_dir=out_vid,
                            render_setting=self.cfg.test_setting,
                            frames_dir=None,
                        )
                        rendered_cmp_dirs.append(out_vid)

                pre_rendered_dirs = list(
                    ds_cfg.get("compare_folders", None) or []
                )
                cmp_names = list(ds_cfg.get("compare_names", None) or [])
                # Order: on-the-fly GT first, then pre-rendered.
                all_compare_folders = rendered_cmp_dirs + pre_rendered_dirs
                # Pad/truncate names to match folder count to avoid IndexError.
                if len(cmp_names) < len(all_compare_folders):
                    cmp_names = cmp_names + [
                        f"Compare {k+1}"
                        for k in range(len(all_compare_folders) - len(cmp_names))
                    ]
                cmp_names = cmp_names[: len(all_compare_folders)]

                # Composite. PHC-trained models emit 3-col
                # [PHC_raw | PHC_decoded | GT]; otherwise the plain 2-col.
                phcraw_dir = f"{self.cfg.save_dir}/{dataset_id}/phcraw"
                if os.path.isdir(phcraw_dir):
                    phcraw_vid = f"{self.cfg.save_dir}/{dataset_id}/phcraw_video"
                    phcdec_vid = f"{self.cfg.save_dir}/{dataset_id}/phc_decoded_video"
                    render_video(motion_dir=phcraw_dir, save_dir=phcraw_vid,
                                 render_setting=self.cfg.test_setting, frames_dir=None)
                    render_video(
                        motion_dir=f"{self.cfg.save_dir}/{dataset_id}/phc_decoded",
                        save_dir=phcdec_vid,
                        render_setting=self.cfg.test_setting, frames_dir=None)
                    make_composite_compare_videos(
                        result_folder=phcraw_vid,
                        result_name="PHC_raw",
                        compare_folders=[phcdec_vid] + all_compare_folders,
                        compare_names=["PHC_decoded"] + cmp_names,
                        text_folder=f"{self.cfg.save_dir}/{dataset_id}/text",
                        save_dir=f"{self.cfg.save_dir}/{dataset_id}/composite",
                    )
                else:
                    make_composite_compare_videos(
                        result_folder=f"{self.cfg.save_dir}/{dataset_id}/video",
                        compare_folders=all_compare_folders if all_compare_folders else None,
                        compare_names=cmp_names if cmp_names else None,
                        text_folder=f"{self.cfg.save_dir}/{dataset_id}/text",
                        save_dir=f"{self.cfg.save_dir}/{dataset_id}/composite",
                    )

                # wandb log video
                if (
                    not self.cfg.debug
                    and self.logger is not None
                    and isinstance(self.logger, WandbLogger)
                ):
                    video_to_log = []
                    for video_path in sorted(
                        os.listdir(f"{self.cfg.save_dir}/{dataset_id}/composite")
                    ):
                        video_to_log.append(
                            wandb.Video(
                                f"{self.cfg.save_dir}/{dataset_id}/composite/{video_path}",
                                format="gif",
                            )
                        )
                    wandb.log(
                        {f"{dataset_id}_video": video_to_log},
                        step=self.global_step,
                    )


def _build_loaders(ds, data_cfg, collate_fn, prefix):
    """Build one DataLoader per sub-dataset, or one total for a single dataset.

    Used for both val and test; pass prefix="val" or "test" to drive per-dataset
    config lookup. Splitting per sub-dataset matters because each batch must be
    homogeneous: BabelDataset and HumanML3DDataset return per-sample dicts with
    different key sets (e.g. only HumanML3D includes "token_start"), and
    multi.collate_fn iterates batch[0].keys(), which crashes on KeyError when a
    later sample is missing one. Iterates ds.datasets when present (MultiDataset)
    or wraps a single dataset as a 1-element list. Empty sub-datasets are
    skipped (covers ldf_babel.yaml BABEL.val_meta_paths=[]). Per-dataset
    overrides via data_cfg.<prefix>_<name>_bs / <prefix>_<name>_samples, where
    <name> is the dataset class name lowercased without the "Dataset" suffix.
    """
    if hasattr(ds, "datasets") and hasattr(data_cfg, "datasets"):
        children = list(zip(ds.datasets, data_cfg.datasets))
    else:
        children = [(ds, data_cfg)]
    bs_default = data_cfg.val_bs if prefix == "val" else data_cfg.test_bs

    loaders = []
    for child, ds_conf in children:
        if len(child) == 0:
            continue
        target = ds_conf.get("target", "") if hasattr(ds_conf, "get") else getattr(ds_conf, "target", "")
        ds_key = target.split(".")[-1].replace("Dataset", "").lower()
        cap = data_cfg.get(f"{prefix}_{ds_key}_samples", None)
        if cap is not None and len(child) > cap:
            child = torch.utils.data.Subset(child, range(cap))
        bs = data_cfg.get(f"{prefix}_{ds_key}_bs", bs_default)
        loaders.append(DataLoader(
            child, batch_size=bs, shuffle=False, drop_last=False,
            num_workers=data_cfg.num_workers, persistent_workers=False,
            prefetch_factor=8, collate_fn=collate_fn,
        ))
    return loaders


def main():
    # init
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    # Segment-based sweeps can resume the same experiment several times.  An
    # explicit run_dir keeps every 5k checkpoint, log, and render in one
    # stable directory instead of creating a new timestamped directory for
    # every segment.  Legacy configs keep the timestamped behavior.
    configured_run_dir = cfg.get("run_dir", None)
    if configured_run_dir:
        save_dir = os.path.abspath(str(configured_run_dir))
    else:
        save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    if cfg.get("save_code_snapshot", True):
        save_config_and_codes(cfg, cfg.save_dir)
    else:
        # Evaluation-only milestone runs are numerous.  They still record the
        # fully resolved config, but avoid copying the repository 96 times.
        OmegaConf.save(
            config=cfg.config,
            f=os.path.join(cfg.save_dir, "resolved_config.yaml"),
        )

    logger = None
    if not cfg.debug:
        wandb_key = cfg.logger.wandb.wandb_key
        if wandb_key and wandb_key.strip():
            os.environ["WANDB_API_KEY"] = wandb_key
            logger = WandbLogger(
                project=cfg.logger.wandb.project,
                name=f"{cfg.exp_name}_{run_time}",
                entity=cfg.logger.wandb.entity,
                config=OmegaConf.to_container(cfg.config, resolve=True),
                save_dir=cfg.save_dir,
            )
            rank_zero_info("WandB logging enabled")
        else:
            rank_zero_info("WandB API key not provided, skipping WandB logging")

    # dataloader
    collate_fn = (
        get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    )

    train_dataset = (
        instantiate(cfg.data.target, cfg=cfg.config, split="train")
        if cfg.train
        else None
    )
    val_dataset = instantiate(
        cfg.data.get("val_target", cfg.data.target), cfg=cfg.config, split="val"
    )
    test_dataset = instantiate(
        cfg.data.get("test_target", cfg.data.target), cfg=cfg.config, split="test"
    )
    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, Val dataset: {len(val_dataset) if val_dataset is not None else 0}, Test dataset: {len(test_dataset)}"
    )

    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=True,
            prefetch_factor=8,
            collate_fn=collate_fn,
        )
        if cfg.train
        else None
    )
    val_loaders = _build_loaders(val_dataset, cfg.data, collate_fn, prefix="val")
    test_loaders = _build_loaders(test_dataset, cfg.data, collate_fn, prefix="test")

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)
    # validation_step dispatches by dataloader_idx using this count: the first
    # _n_val_loaders dataloaders run val_loss/metrics, the rest is test_loader.
    model._n_val_loaders = len(val_loaders)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="step_{step}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor="step",
        mode="max",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    if cfg.train:
        callbacks.append(checkpoint_callback)

    # Handle devices as either int or list
    num_devices = (
        cfg.trainer.devices
        if isinstance(cfg.trainer.devices, int)
        else len(cfg.trainer.devices)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        # find_unused_parameters=False: VAE/T2M/Physics metric submodules
        # are explicitly frozen (`.eval()` + `requires_grad=False`); WanModel
        # is the sole grad bearer and every training_step's forward touches
        # all its params (validated in P4.2 + P4.3 smokes — PyTorch itself
        # logged `find_unused_parameters=True ... did not find any unused
        # parameters in the forward pass. ... consider turning this flag off`).
        # Saves the per-iteration extra autograd graph traversal.
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if (num_devices > 1 or world_size > 1)
            else "auto"
        ),
        callbacks=callbacks,
        default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    if cfg.train:
        # Load init weights before the pre-fit sanity validate so that first
        # metric measures the init ckpt, not random LDF weights. When
        # resume_ckpt is set, Lightning overrides this load inside fit().
        model.load_init_ckpt()
        # pre_fit_validate: explicit flag, default follows old behavior
        # (enabled iff debug=false). Set pre_fit_validate=false via --override
        # to skip the ~45-min pre-fit validate without flipping debug mode
        # (which would also shrink the dataset to 100 samples).
        if cfg.get("pre_fit_validate", not cfg.debug):
            trainer.validate(model, dataloaders=[*val_loaders, *test_loaders])
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[*val_loaders, *test_loaders],
            ckpt_path=cfg.resume_ckpt,
        )
    else:
        # Official FloodDiffusion releases use the pre-rename state-dict
        # format handled by load_init_ckpt().  Allow evaluation with
        # test_ckpt=null so the common initializer can enter the same
        # validation protocol without first materializing a converted copy.
        if cfg.test_ckpt is None:
            model.load_init_ckpt()
        for i in range(cfg.config.val_repeat):
            # Set different seed for each validation run to get diverse results
            # But keep it deterministic: same i -> same seed -> same result
            seed_everything(cfg.seed + i)
            # Lightning 2.3/2.4 Trainer.validate() does not accept weights_only
            # (that was a torch.load arg mis-placed here); removed to avoid
            # TypeError. ckpt_path alone loads the checkpoint correctly.
            trainer.validate(
                model,
                dataloaders=[*val_loaders, *test_loaders],
                ckpt_path=cfg.test_ckpt,
            )
            model.cfg.test_setting.render = False  # only render once

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train
    # train.py --config configs/138d/ldf_138d_streaming_model.yaml
    main()
