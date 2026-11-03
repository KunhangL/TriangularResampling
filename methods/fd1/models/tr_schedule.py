"""Training-only TR curriculum, independent of LR and inference schedules."""
import math


def resampling_shift(step, config=None, default=0.6):
    cfg = dict(config or {})
    mode = cfg.get("mode", "constant")
    if mode == "constant":
        value = float(cfg.get("value", default))
    elif mode == "cosine_log":
        start, end = float(cfg.get("start", 3)), float(cfg.get("end", .6))
        duration = int(cfg.get("duration", 15000))
        if duration <= 0 or start <= 0 or end <= 0 or not math.isfinite(float(step)):
            raise ValueError("Invalid TR schedule or missing optimizer step")
        u = min(max(float(step), 0) / duration, 1)
        h = (1 - math.cos(math.pi * u)) / 2
        value = start if u == 0 else end if u == 1 else math.exp((1-h)*math.log(start)+h*math.log(end))
    else:
        raise ValueError(f"Unknown TR schedule: {mode}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("TR shift must be finite and positive")
    return value
