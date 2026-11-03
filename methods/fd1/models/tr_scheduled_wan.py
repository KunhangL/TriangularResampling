"""Optional curriculum wrapper; model state and inference are unchanged."""
from .diffusion_forcing_wan import DiffForcingWanModel
from .tr_schedule import resampling_shift


class ScheduledTRWan(DiffForcingWanModel):
    def __init__(self, resampling_schedule=None, **kwargs):
        super().__init__(**kwargs)
        self.resampling_schedule = dict(resampling_schedule or {})
        self.fixed_resampling_shift = self.resample_shift
        resampling_shift(0, self.resampling_schedule, self.fixed_resampling_shift)

    def forward(self, batch):
        if self.training and self.resampling_ratio > 0:
            self.resample_shift = resampling_shift(
                batch.get("global_step", float("inf")), self.resampling_schedule,
                self.fixed_resampling_shift)
            if self.resampler is not None:
                self.resampler.shift = self.resample_shift
        return super().forward(batch)
