import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from methods.fd1.metrics.mr import MRMetrics
from common.metrics.t2m import T2MMetrics
from common.utils.initialize import (
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from methods.fd1.utils.lightning_module import BasicLightningModule
from common.utils.motion_process import convert_motion_to_joints
from methods.fd1.utils.visualize import (  # evaluate_video
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def _step(self, batch, is_training=True):
        batch = self._pad_batch(batch)
        return self.model(batch)

    @staticmethod
    def _pad_batch(batch):
        """Pad list-of-tensors motion/motion_length to batched tensors.

        No-op when the dataset's collate_fn already returns tensors
        (e.g. HumanML3DDataset uses 'feature'/'feature_length' and pads there).
        """
        if "motion" in batch and isinstance(batch["motion"], list):
            batch["motion"] = torch.nn.utils.rnn.pad_sequence(
                batch["motion"], batch_first=True, padding_value=0
            )
        if "motion_length" in batch and isinstance(batch["motion_length"], list):
            batch["motion_length"] = torch.tensor(batch["motion_length"])
        return batch

    def initialize_metrics(self):
        # metric models
        self.recover_dim = self.cfg.metrics.dim
        self.features2joints = convert_motion_to_joints
        # Per-dataset MR metric buckets so HumanML3D and BABEL MPJPE/PAMPJPE/
        # ACCEL show up as separate keys on wandb. Datasets list is read from
        # cfg.metrics.datasets; falls back to ["HumanML3D"] for legacy single-
        # dataset configs (vae_wan.yaml etc.) where no datasets key is set.
        metric_datasets = self.cfg.metrics.get("datasets", ["HumanML3D"])
        self.mr_metrics = nn.ModuleDict(
            {ds: MRMetrics() for ds in metric_datasets}
        )
        t2m_cfg = self.cfg.metrics.get("t2m", None)
        # T2M evaluator is HumanML3D-trained; pooling BABEL into FID corrupts
        # stats. update_metrics gates t2m.update() on dataset_id == HumanML3D.
        self.t2m_metrics = T2MMetrics(t2m_cfg) if t2m_cfg is not None else None

    def update_metrics(self, batch, generated=None):
        batch = self._pad_batch(batch)
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        motion = output["generated"]
        feature_key = self.model.input_keys.get("feature", "feature")
        ground_truth = batch[feature_key]
        dataset_ids = batch.get("dataset", None)  # per-sample dataset name
        metric_fps = self.cfg.get("test_setting", {}).get("fps", 20 if self.recover_dim in (138, "138") else 30)

        # recover to joint positions
        for i in range(len(motion)):
            single_motion = motion[i]
            single_gt = ground_truth[i]
            length = min(single_motion.shape[0], single_gt.shape[0])
            single_motion = single_motion[:length]
            single_gt = single_gt[:length]
            joints = self.features2joints(
                single_motion.float().cpu().numpy(),
                self.recover_dim,
                fps=metric_fps,
            )
            gt_joints = self.features2joints(
                single_gt.float().cpu().numpy(),
                self.recover_dim,
                fps=metric_fps,
            )
            # float32
            single_motion = single_motion.float().to(self.device)
            single_gt = single_gt.float().to(self.device)
            # Route MR per dataset; if dataset id missing or absent from buckets
            # (legacy single-dataset config), fall back to "HumanML3D".
            # nn.ModuleDict has no .get(), so use `in` + `[]` indexing.
            ds_id = dataset_ids[i] if dataset_ids is not None else "HumanML3D"
            if ds_id in self.mr_metrics:
                mr_bucket = self.mr_metrics[ds_id]
            elif "HumanML3D" in self.mr_metrics:
                mr_bucket = self.mr_metrics["HumanML3D"]
            else:
                mr_bucket = None
            if mr_bucket is not None:
                mr_bucket.update(
                    joints_rst=torch.tensor(joints)[None, ...],
                    joints_ref=torch.tensor(gt_joints)[None, ...],
                    lengths=[length],
                )
            # T2M evaluator only trained on HumanML3D — pooling BABEL would
            # corrupt FID stats. Mirror train_ldf.py:249's run_t2m guard.
            run_t2m = (dataset_ids is None) or (ds_id == "HumanML3D")
            if self.t2m_metrics is not None and run_t2m:
                self.t2m_metrics.update(
                    feats_rst=single_motion[None, ...],
                    feats_ref=single_gt[None, ...],
                    lengths_rst=[length],
                    lengths_ref=[length],
                )
        return

    def compute_metrics(self):
        # Per-dataset MR. compute() calls reset() internally (metrics/mr.py:37),
        # so each bucket self-cleans for next epoch.
        for ds_name, mr in self.mr_metrics.items():
            mr_output = mr.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in mr_output.items():
                self.log(
                    f"metrics/mr_metrics/{ds_name}/{key}", value, sync_dist=False
                )
        if self.t2m_metrics is not None:
            t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in t2m_output.items():
                self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=False)

    def load_init_ckpt(self):
        """Weights-only VAE init (NOT a full Lightning resume). Loads the model
        state_dict + EMA from an existing VAE checkpoint before trainer.fit() —
        so optimizer/scheduler are fresh and PL version mismatches don't bite.
        Mirrors train_ldf.py:112-139."""
        init_ckpt = self.cfg.get("init_ckpt", None)
        if not init_ckpt:
            return
        rank_zero_info(f"[init_ckpt] loading VAE weights from {init_ckpt}")
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        if "ema_state" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state"])
            self.ema.copy_to(self.model.parameters())
            rank_zero_info(
                "[init_ckpt] EMA state loaded into self.ema and copied to model "
                "(model + ema shadow both carry the EMA-averaged weights)"
            )
        else:
            rank_zero_info("[init_ckpt] no ema_state in ckpt; using raw state_dict")

    def update_test(self, batch):
        """Save raw + VAE-recon for the main feature, plus optional `track`
        extra (when the dataset emits `feature_track`/`feature_track_length`).
        Output layout per dataset:
            <save>/<dataset>/gt/<id>.npy         raw main feature
            <save>/<dataset>/vae_gt/<id>.npy     VAE encode→decode of main
            <save>/<dataset>/track/<id>.npy      raw track feature (if present)
            <save>/<dataset>/vae_track/<id>.npy  VAE encode→decode of track
            <save>/<dataset>/text/<id>.txt       caption (or "Motion: <id>" fallback)
        process_test_results renders these into a 4-panel composite."""
        batch = self._pad_batch(batch)
        # Main pass — VAE recon of `feature`.
        with self.ema.average_parameters(self.model.parameters()):
            output_main = self.model.generate(batch)
        motion_main = output_main["generated"]

        # Optional second pass — VAE recon of `feature_track`. The VAE.generate
        # reads `feature` / `feature_length` from the dict, so build a shallow
        # alias dict that points at the track tensors.
        has_track = (
            "feature_track" in batch and "feature_track_length" in batch
        )
        motion_track = None
        if has_track:
            batch_track = {**batch,
                           "feature": batch["feature_track"],
                           "feature_length": batch["feature_track_length"]}
            with self.ema.average_parameters(self.model.parameters()):
                output_track = self.model.generate(batch_track)
            motion_track = output_track["generated"]

        motion_id = batch["name"]
        dataset_id = batch["dataset"]
        feature = batch["feature"]
        feature_length = batch["feature_length"]
        feature_track = batch.get("feature_track")
        feature_track_length = batch.get("feature_track_length")
        text = batch.get("text", None)
        if text is None:
            text = [f"Motion: {mid}" for mid in motion_id]

        for i, (single_motion_id, single_dataset_id, single_text) in enumerate(
            zip(motion_id, dataset_id, text)
        ):
            base = os.path.join(self.cfg.save_dir, single_dataset_id)
            # gt: raw main feature, trimmed to its valid length
            os.makedirs(os.path.join(base, "gt"), exist_ok=True)
            np.save(
                os.path.join(base, "gt", f"{single_motion_id}.npy"),
                feature[i, : int(feature_length[i])].float().cpu().numpy(),
            )
            # vae_gt: VAE recon of main
            os.makedirs(os.path.join(base, "vae_gt"), exist_ok=True)
            np.save(
                os.path.join(base, "vae_gt", f"{single_motion_id}.npy"),
                motion_main[i].float().cpu().numpy(),
            )
            if has_track:
                # track: raw track feature
                os.makedirs(os.path.join(base, "track"), exist_ok=True)
                np.save(
                    os.path.join(base, "track", f"{single_motion_id}.npy"),
                    feature_track[i, : int(feature_track_length[i])].float().cpu().numpy(),
                )
                # vae_track: VAE recon of track
                os.makedirs(os.path.join(base, "vae_track"), exist_ok=True)
                np.save(
                    os.path.join(base, "vae_track", f"{single_motion_id}.npy"),
                    motion_track[i].float().cpu().numpy(),
                )
            # caption file
            os.makedirs(os.path.join(base, "text"), exist_ok=True)
            with open(
                os.path.join(base, "text", f"{single_motion_id}.txt"), "w"
            ) as f:
                f.write(single_text)
        return

    def process_test_results(self):
        """Render the per-id npy outputs from update_test into a 4-panel
        composite per dataset:
            panel 1: GT (raw new_joint_vecs)         ← gt/
            panel 2: VAE(GT)                          ← vae_gt/
            panel 3: track (raw new_joint_vecs_track) ← track/
            panel 4: VAE(track)                       ← vae_track/
        Falls back gracefully to a 2-panel (GT | VAE(GT)) composite when track
        wasn't loaded for this dataset (e.g. legacy single-feature configs)."""
        if not self.cfg.test_setting.render:
            return

        # Hard-coded panel order. result_folder drives id enumeration; result
        # ends up on the LEFT (default), compares fill the rest in this order.
        # NOTE: each id must exist in `gt/` for the composite to enumerate it;
        # update_test always writes gt/, so this is safe.
        panel_subdirs = ["gt", "vae_gt", "track", "vae_track"]
        panel_labels = {
            "gt": "GT (AMASS)",
            "vae_gt": "VAE(GT)",
            "track": "track (PHC)",
            "vae_track": "VAE(track)",
        }

        for dataset_id in os.listdir(self.cfg.save_dir):
            base = f"{self.cfg.save_dir}/{dataset_id}"
            gt_npy_dir = f"{base}/gt"
            if not os.path.isdir(gt_npy_dir):
                continue

            # Render each panel that exists. (track / vae_track may be absent
            # when the dataset wasn't configured with extra_feature_paths.)
            present = []
            for sub in panel_subdirs:
                npy_dir = f"{base}/{sub}"
                if not os.path.isdir(npy_dir):
                    continue
                vid_dir = f"{base}/{sub}_video"
                render_video(
                    motion_dir=npy_dir,
                    save_dir=vid_dir,
                    render_setting=self.cfg.test_setting,
                    frames_dir=None,
                )
                present.append(sub)

            if "gt" not in present:
                continue

            # Compose: result = gt (left), compares = the others in declared order
            compare_subs = [s for s in panel_subdirs if s in present and s != "gt"]
            compare_folders = [f"{base}/{s}_video" for s in compare_subs]
            compare_names = [panel_labels[s] for s in compare_subs]
            make_composite_compare_videos(
                result_folder=f"{base}/gt_video",
                compare_folders=compare_folders,
                compare_names=compare_names,
                text_folder=f"{base}/text",
                save_dir=f"{base}/composite",
                result_name=panel_labels["gt"],
                result_position="left",
            )

            # wandb log composite videos for this dataset
            if (
                not self.cfg.debug
                and self.logger is not None
                and isinstance(self.logger, WandbLogger)
            ):
                composite_dir = f"{base}/composite"
                video_to_log = [
                    wandb.Video(
                        os.path.join(composite_dir, vp), format="gif"
                    )
                    for vp in sorted(os.listdir(composite_dir))
                ]
                wandb.log(
                    {f"{dataset_id}_video": video_to_log},
                    step=self.global_step,
                )


def initialize_config():
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    OmegaConf.update(cfg.config, "run_time", run_time)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)
    return cfg


def main():
    # init
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)

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
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)

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

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True)
        if num_devices > 1
        else "auto",
        callbacks=callbacks,
        default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    if cfg.train:
        # Weights-only init from an existing VAE ckpt (e.g. fine-tune setup).
        # Must run BEFORE pre-fit validate so the sanity metric reflects the
        # init weights, not random ones. resume_ckpt overrides this inside
        # trainer.fit() if both are set.
        model.load_init_ckpt()
        if not cfg.debug:
            trainer.validate(model, dataloaders=[val_dataloader, test_dataloader])
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[val_dataloader, test_dataloader],
            ckpt_path=cfg.resume_ckpt,
        )
    else:
        for i in range(cfg.config.val_repeat):
            # Set different seed for each validation run to get diverse results
            # But keep it deterministic: same i -> same seed -> same result
            seed_everything(cfg.seed + i)
            trainer.validate(
                model,
                dataloaders=[val_dataloader, test_dataloader],
                ckpt_path=cfg.test_ckpt,
            )
            model.cfg.test_setting.render = False  # only render once

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train
    # train.py --config configs/default_vae.yaml
    main()
