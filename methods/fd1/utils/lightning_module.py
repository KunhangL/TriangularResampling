import os
import time

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch_ema import ExponentialMovingAverage

from common.utils.initialize import (
    compare_statedict_and_parameters,
    instantiate,
    print_model_size,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BasicLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_params = dict(cfg.model.params)
        # PHC is resampler/runtime configuration, not a model constructor arg.
        # Keep cfg.model.params unchanged so train_ldf.py can still build the
        # PHC resampler from the original config during setup("fit").
        if cfg.model.target == "methods.fd1.models.diffusion_forcing_wan.DiffForcingWanModel":
            model_params.pop("phc", None)
        self.model = instantiate(
            target=cfg.model.target, cfg=None, hfstyle=False, **model_params
        )

        # NOTE: ligntning init stage the device is cpu, so no need to move to device
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=cfg.model.ema_decay
        )
        print_model_size(self.model)

        # logging
        self.last_batch_end_time, self.batch_ready_time = None, None
        self._train_batch_wall_start = None
        self._ema_device = None
        self.validation_step_outputs = []

        # metric
        self.initialize_metrics()

    def configure_optimizers(self):
        optim_target = self.cfg.optimizer.target
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target
        optimizer = instantiate(
            target=optim_target,
            cfg=None,
            hfstyle=False,
            params=self.model.parameters(),
            **self.cfg.optimizer.params,
        )

        scheduler_target = self.cfg.lr_scheduler.target
        if len(scheduler_target.split(".")) == 1:
            scheduler_target = "torch.optim.lr_scheduler." + scheduler_target
        scheduler_params = dict(self.cfg.lr_scheduler.params)
        warmup_steps = scheduler_params.pop("warmup_steps", 0)

        lr_scheduler = instantiate(
            target=scheduler_target,
            cfg=None,
            hfstyle=False,
            optimizer=optimizer,
            **scheduler_params,
        )

        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0,
                total_iters=warmup_steps,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, lr_scheduler],
                milestones=[warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        pass

    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
            self._ema_device = None
            rank_zero_info("init ema from ckpt")
        else:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.model.ema_decay
            )
            self._ema_device = None
            rank_zero_info("init ema from current model weights")

        # Compare state_dict and parameters
        compare_statedict_and_parameters(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()
        checkpoint["state_dict"] = self.model.state_dict()

    def _step(self, batch, is_training=True):
        out = self.model(batch)
        return out

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_ready_time = time.time()
        self._train_batch_wall_start = self.batch_ready_time

    def _ensure_ema_device(self):
        device = torch.device(self.device)
        if self._ema_device != device:
            self.ema.to(device)
            self._ema_device = device

    def training_step(self, batch, batch_idx):
        net_start_time = time.time()
        # forward
        loss_dict = self._step(batch, is_training=True)
        # logging
        net_end_time = time.time()
        data_time = (
            self.batch_ready_time - self.last_batch_end_time
            if self.last_batch_end_time is not None
            else 0.0
        )
        net_time = net_end_time - net_start_time
        batch_size = self.cfg.data.train_bs
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "data_time", data_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "net_time", net_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        for key, value in loss_dict.items():
            self.log(
                f"train_loss/{key}",
                value.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss_dict["total"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = self.cfg.data.train_bs
        pre_ema_time = time.time()
        if self._train_batch_wall_start is not None:
            self.log(
                "train_step_time",
                pre_ema_time - self._train_batch_wall_start,
                on_step=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=False,
            )

        ema_start_time = time.time()
        self._ensure_ema_device()
        self.ema.update()
        # Calculate average difference using vectorized operations
        if self.global_step % 100 == 0:
            self.log("ema_decay", self.ema.decay, sync_dist=False)
            with torch.no_grad():
                diff_sum = torch.zeros((), device=self.device)
                elem_count = 0
                for i, p in enumerate(self.model.parameters()):
                    if not p.requires_grad:
                        continue
                    shadow = self.ema.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    diff_sum = diff_sum + (p.detach() - shadow).abs().sum()
                    elem_count += p.numel()
                avg_diff = diff_sum / max(elem_count, 1)
                self.log("ema_diff/avg", avg_diff, sync_dist=True)
        ema_time = time.time() - ema_start_time
        self.log(
            "ema_time",
            ema_time,
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=False,
        )
        if self._train_batch_wall_start is not None:
            self.log(
                "train_wall_time",
                time.time() - self._train_batch_wall_start,
                on_step=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=False,
            )
        self.last_batch_end_time = time.time()

    # NOTE: lightning handles with torch.no_grad() and model.eval() automatically
    # First _n_val_loaders dataloaders run val_loss/metrics; the rest (a single
    # test loader by convention) runs render. val_loss is computed only on
    # dataloader_idx == 0 (HumanML3D loader by config convention, see ldf_stream
    # datasets ordering). Metrics dispatch is per-sample inside update_metrics
    # via run_t2m / physics gt_jerk guards, so all val loaders can share it.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        n_val = getattr(self, "_n_val_loaders", 1)
        is_test = dataloader_idx >= n_val

        if is_test:
            if self.global_step % self.cfg.validation.test_steps == 0:
                self.update_test(batch)
            return

        if dataloader_idx == 0:
            with self.ema.average_parameters(self.model.parameters()):
                loss_dict = self._step(batch, is_training=False)
            batch_size = self.cfg.data.val_bs
            for key, value in loss_dict.items():
                self.log(
                    f"val_loss/{key}",
                    value.detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        if self.global_step % self.cfg.validation.test_steps == 0:
            self.update_metrics(batch)
        return

    def on_validation_epoch_end(self):
        if self.global_step % self.cfg.validation.test_steps == 0:
            self.compute_metrics()
            if self.trainer.global_rank == 0:
                self.process_test_results()

    # NOTE: lightning handles with torch.no_grad() and model.eval() automatically
    # Kept for standalone trainer.test() — bundles save + metric on the same loader.
    def test_step(self, batch, batch_idx):
        self.update_test(batch)
        self.update_metrics(batch)
        return

    def on_test_epoch_end(self):
        self.compute_metrics()
        if self.trainer.global_rank == 0:
            self.process_test_results()

    def initialize_metrics(self):
        pass

    def update_metrics(self, batch, generated=None):
        pass

    def compute_metrics(self):
        pass

    def update_test(self, batch):
        pass

    def process_test_results(self):
        pass
