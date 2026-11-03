import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.t5 import T5EncoderModel
from .tools.wan_model import WanModel

EPSILON = 0.05


class TriangularTimeScheduler:
    def __init__(self, config):
        self.steps = config["steps"]
        self.chunk_size = config["chunk_size"]
        self.random_epsilon = config.get("random_epsilon", 0.00)  # schedule jittering
        self.noise_type = config.get("noise_type", "linear")
        self.sigma_type = config.get("sigma_type", "zero")  # "zero", "memoryless"

        if self.noise_type == "exponential" or self.noise_type == "exponential_rev":
            self.exp_max = config.get("exp_max", 20.0)
        elif self.noise_type == "diffusion":
            self.T = config.get("T", 1000)
            self.beta_start = config.get("beta_start", 0.0001)
            self.beta_end = config.get("beta_end", 0.02)

        if self.sigma_type == "memoryless":
            self.sigma_scale = config.get("sigma_scale", 1.0)
        self.content_len = config.get("content_len", None)
        # For simplicity we require steps to be divisible by chunk_size, so that time windows align well.

    def get_total_steps(self, seq_len):
        return int(self.steps * seq_len / self.chunk_size)

    def get_time_steps(self, device, valid_len, current_step=None):
        time_steps = []
        if current_step is None:
            for i in range(len(valid_len)):
                max_time = valid_len[i] / self.chunk_size
                time_steps.append(
                    torch.tensor(np.random.uniform(0, max_time), device=device)
                )
        elif isinstance(current_step, int):
            for i in range(len(valid_len)):
                t = current_step * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device))
        elif isinstance(current_step, list):
            for i in range(len(valid_len)):
                t = current_step[i] * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device))
        return time_steps

    def get_time_schedules(self, device, valid_len, time_steps, training=False):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            current_time_schedules = torch.clamp(
                -torch.arange(valid_len[i], device=device) / self.chunk_size + t,
                min=0.0,
                max=1.0,
            )
            current_time_schedules_derivative = torch.ones_like(
                current_time_schedules
            ) * (1 / self.steps)
            if training:
                current_time_schedules = torch.clamp(
                    current_time_schedules
                    + torch.randn_like(current_time_schedules) * self.random_epsilon,
                    min=0.0,
                    max=1.0,
                )
            time_schedules.append(current_time_schedules)
            time_schedules_derivative.append(current_time_schedules_derivative)
        return time_schedules, time_schedules_derivative

    def get_windows(self, valid_len, time_steps):
        # for the floating point issue, we can add the start_index by 0.5 / [steps * chunk_size]
        # for convenience, we just choose 0.5 * (1 / (self.steps * self.chunk_size)) here
        input_start, input_end, output_start, output_end = [], [], [], []
        for i in range(len(time_steps)):
            t = time_steps[i].item()
            start_index = max(
                0,
                math.floor(
                    (t - 1) * self.chunk_size
                    + 0.5 * (1 / (self.steps * self.chunk_size))
                )
                + 1,
            )
            end_index = min(
                valid_len[i],
                math.floor(
                    t * self.chunk_size + 0.5 * (1 / (self.steps * self.chunk_size))
                )
                + 1,
            )

            if self.content_len is not None:
                input_start.append(max(0, end_index - self.content_len))
            else:
                input_start.append(0)
            input_end.append(end_index)
            output_start.append(start_index)
            output_end.append(end_index)
        return input_start, input_end, output_start, output_end

    def get_noise_levels(self, device, valid_len, time_schedules):
        alpha = []
        dalpha = []
        dlog_alpha = []
        beta = []
        dbeta = []
        dlog_beta = []
        sigma = []
        for i in range(len(valid_len)):
            t = time_schedules[i]
            if self.noise_type == "linear":
                alpha_i = t
                dalpha_i = torch.ones_like(alpha_i)
                dlog_alpha_i = dalpha_i / torch.clamp(alpha_i, min=EPSILON)
                beta_i = 1 - t
                dbeta_i = -torch.ones_like(beta_i)
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            elif self.noise_type == "exponential":
                # "eps" prediction
                k = self.exp_max
                alpha_i = torch.exp(-k * (1 - t))
                dalpha_i = k * alpha_i
                dlog_alpha_i = k * torch.ones_like(alpha_i)
                beta_i = 1 - alpha_i
                dbeta_i = -dalpha_i
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            elif self.noise_type == "exponential_rev":
                # "x0" prediction
                k = self.exp_max
                beta_i = torch.exp(-k * t)
                dbeta_i = -k * beta_i
                dlog_beta_i = -k * torch.ones_like(beta_i)
                alpha_i = 1 - beta_i
                dalpha_i = -dbeta_i
                dlog_alpha_i = dalpha_i / torch.clamp(alpha_i, min=EPSILON)
            elif self.noise_type == "diffusion":
                t_rev = 1.0 - t
                beta_rate = (
                    self.beta_start + t_rev * (self.beta_end - self.beta_start)
                ) * self.T
                Gamma = (
                    self.beta_start * t_rev
                    + 0.5 * (self.beta_end - self.beta_start) * t_rev * t_rev
                ) * self.T
                alpha_i = torch.exp(-0.5 * Gamma)
                dalpha_i = 0.5 * beta_rate * alpha_i
                dlog_alpha_i = 0.5 * beta_rate
                beta_i = torch.sqrt(torch.clamp(1 - torch.exp(-Gamma), min=0.0))
                dbeta_i = (
                    -0.5
                    * torch.exp(-Gamma)
                    * beta_rate
                    / torch.clamp(beta_i, min=EPSILON)
                )
                dlog_beta_i = dbeta_i / torch.clamp(beta_i, min=EPSILON)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            alpha.append(torch.clamp(alpha_i, min=0.0, max=1.0))
            dalpha.append(dalpha_i)
            dlog_alpha.append(dlog_alpha_i)
            beta.append(torch.clamp(beta_i, min=0.0, max=1.0))
            dbeta.append(dbeta_i)
            dlog_beta.append(dlog_beta_i)
            if self.sigma_type == "zero":
                sigma_i = torch.zeros_like(t)
            elif self.sigma_type == "memoryless":
                if (
                    self.noise_type == "linear"
                    or self.noise_type == "exponential"
                    or self.noise_type == "exponential_rev"
                ):
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * dlog_alpha_i * beta_i, min=0.0)
                    )
                elif self.noise_type == "diffusion":
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(2 * dlog_alpha_i, min=0.0)
                    )
                else:
                    sigma_i = self.sigma_scale * torch.sqrt(
                        torch.clamp(
                            2 * beta_i * (dlog_alpha_i * beta_i - dbeta_i), min=0.0
                        )
                    )
            sigma.append(sigma_i)
        return alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta

    def add_noise(
        self,
        x,
        alpha,
        beta,
        input_start,
        input_end,
        output_start,
        output_end,
        training=False,
        noise=None,
        x0_source=None,
        self_forcing_mask=None,
    ):
        """Add noise and slice into input/reference regions.
        Args:
            x: list of (C, T, H, W). xt is built by noising this tensor (i.e.
               the model-input source). In training the resampling-forcing
               path passes post-resample latents here so the model input
               reflects the self-rollout state.
            alpha: list of (T,)
            beta: list of (T,)
            input_start/input_end: per-sample input window indices
            output_start/output_end: per-sample output window indices
            x0_source: optional list of (C, T, H, W). When given, x0 is sliced
               from this tensor instead of `x`. Used by training to point the
               loss target at the pre-resample original GT while xt still
               reflects post-resample model input. Default None → x0_src = x
               (legacy behavior, used by inference paths).
            self_forcing_mask: optional per-sample bool list. For masked samples
               `x[i]` is the post-resample latent that ALREADY sits on the
               triangular schedule, so it is used directly as xt (no fresh
               noise) and eps is RECOVERED from (xt, x0=GT): the velocity target
               dα·x0 + dβ·eps then flows the self-rollout state toward GT
               without double-noising. Unmasked samples use the standard
               noised-GT path. None → all samples standard.
        Returns:
            x0: list of (C, output_len, H, W)
            eps: list of (C, output_len, H, W)
            xt: list of (C, input_len, H, W)
        """
        x0_src = x if x0_source is None else x0_source
        x0 = []
        eps = []
        xt = []
        if training:
            for i in range(len(x)):
                x0_i = x0_src[i][:, output_start[i] : output_end[i], ...]
                x0.append(x0_i)
                if self_forcing_mask is not None and self_forcing_mask[i]:
                    # xt = post-resample feature directly (already on-schedule).
                    # Recover the noise implied by (xt, GT) over the output
                    # window so the velocity target points back to GT.
                    xt.append(x[i][:, input_start[i] : input_end[i], ...])
                    a_o = alpha[i][output_start[i] : output_end[i]][None, :, None, None]
                    b_o = beta[i][output_start[i] : output_end[i]].clamp(
                        min=EPSILON)[None, :, None, None]
                    x_o = x[i][:, output_start[i] : output_end[i], ...]
                    eps.append((x_o - x0_i * a_o) / b_o)
                    continue
                if noise is not None:
                    noise_i = noise[i]
                else:
                    noise_i = torch.randn_like(x[i])
                alpha_i = alpha[i][None, :, None, None]  # (1, T, 1, 1)
                beta_i = beta[i][None, :, None, None]  # (1, T, 1, 1)
                noisy_x_i = x[i] * alpha_i + noise_i * beta_i  # (C, T, H, W)
                eps.append(noise_i[:, output_start[i] : output_end[i], ...])
                xt.append(noisy_x_i[:, input_start[i] : input_end[i], ...])
        else:
            for i in range(len(x)):
                xt.append(x[i][:, input_start[i] : input_end[i], ...])
        return x0, eps, xt

    # --- Streaming support ---

    def get_committable(self, total_frames):
        """Given total accumulated conditions, return how many frames can be committed.
        Currently, we suppose steps % chunk_size == 0 for simplicity."""
        committable_length = max(0, total_frames - self.chunk_size + 1)
        committable_steps = total_frames * (self.steps // self.chunk_size)
        return committable_length, committable_steps

    def get_step_rollback(self, seq_len):
        """Get the step count to subtract when wrapping the buffer by seq_len.
        Corresponds to how many steps were consumed by seq_len frames."""
        steps = seq_len * (self.steps // self.chunk_size)
        return steps


class T5TextCrossModule(nn.Module):
    """Cross-attention module for T5 text conditioning."""

    def __init__(
        self,
        len=512,
        dim=4096,
        t5_size="xxl",
        checkpoint_path=None,
        tokenizer_path=None,
        drop_out=0.1,
        extend_last_context=False,
        input_keys={
            "text": "text",
            "text_end": "text_end",
        },
    ):
        assert checkpoint_path is not None and tokenizer_path is not None, (
            "T5 checkpoint and tokenizer paths must be provided."
        )
        super().__init__()
        self.len = len
        self.dim = dim
        self.cross_attn_norm = True
        self.cross_rope = False
        self.drop_out = drop_out
        self.extend_last_context = bool(extend_last_context)
        self.input_keys = input_keys

        self.text_encoder = T5EncoderModel(
            text_len=len,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            t5_size=t5_size,
        )
        self.text_cache = {}

    def encode(self, text_list, device):
        """Encode text list with cache. Returns List[Tensor]."""
        # Deduplicate uncached texts
        texts_to_encode = []
        for text in text_list:
            if text not in self.text_cache and text not in texts_to_encode:
                texts_to_encode.append(text)

        # Batch encode deduplicated texts
        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)
            for text, feature in zip(texts_to_encode, encoded):
                self.text_cache[text] = feature.cpu()

        # Collect from cache
        return [self.text_cache[text].to(device) for text in text_list]

    def get_context(self, x, valid_len, device, param_dtype, training=False):
        """
        Get cross-attention context from input dict.

        Returns:
            context: List[Tensor]
            metadata: dict, may contain 'full_text'
        """
        text_key = self.input_keys.get("text", "text")
        text_end_key = self.input_keys.get("text_end", "text_end")
        metadata = {}

        if text_key not in x:
            text_list = ["" for _ in range(len(valid_len))]
        else:
            text_list = x[text_key]

        if isinstance(text_list[0], list):
            # Multi-segment text (stream mode)
            full_text = []
            all_context = []
            text_end_list = x[text_end_key]

            for i in range(len(valid_len)):
                if training and np.random.rand() <= self.drop_out:
                    single_text_list = [""]
                    single_text_end_list = [0, valid_len[i]]
                else:
                    single_text_list = text_list[i]
                    single_text_end_list = [0] + [
                        min(t, valid_len[i]) for t in text_end_list[i]
                    ]
                single_text_length_list = [
                    t - b
                    for t, b in zip(single_text_end_list[1:], single_text_end_list[:-1])
                ]

                full_text.append(
                    " ////////// ".join(
                        [
                            f"{u} //dur:{t}"
                            for u, t in zip(single_text_list, single_text_length_list)
                        ]
                    )
                )

                single_text_context = self.encode(single_text_list, device)
                single_text_context = [u.to(param_dtype) for u in single_text_context]
                sample_context = []
                for u, duration in zip(single_text_context, single_text_length_list):
                    sample_context.extend([u for _ in range(duration)])
                if self.extend_last_context and len(sample_context) < valid_len[i]:
                    sample_context.extend(
                        [single_text_context[-1]] * (valid_len[i] - len(sample_context))
                    )
                all_context.append(sample_context)
            metadata["full_text"] = full_text
            return all_context, metadata
        else:
            # Single text per sample
            full_text = [u for u in text_list]
            metadata["full_text"] = full_text
            if training:
                text_list = [
                    ("" if np.random.rand() <= self.drop_out else u) for u in text_list
                ]
            else:
                text_list = [u for u in text_list]
            context = self.encode(text_list, device)
            context = [u.to(param_dtype) for u in context]

            return context, metadata

    def get_null_context(self, batch_size, device, param_dtype):
        """Get null/empty context for classifier-free guidance."""
        null_ctx = self.encode([""] * batch_size, device)
        return [u.to(param_dtype) for u in null_ctx]

    # --- Streaming state management ---

    def init_stream(self, batch_size):
        self.stream_condition_list = [[] for _ in range(batch_size)]

    def update_stream(self, x, device, param_dtype):
        """Add one frame of context for a streaming step."""
        text_key = self.input_keys.get("text", "text")
        text_input = x[text_key]
        new_ctx = self.encode(text_input, device)
        new_ctx = [u.to(param_dtype) for u in new_ctx]
        for i in range(len(self.stream_condition_list)):
            self.stream_condition_list[i].append(new_ctx[i])

    def get_stream_context(self, start_index, end_index):
        context = []
        for i in range(len(self.stream_condition_list)):
            context.append(self.stream_condition_list[i][start_index:end_index])
        return context

    def trim_stream(self, trim_len):
        """Trim stream state when wrapping around."""
        for i in range(len(self.stream_condition_list)):
            self.stream_condition_list[i] = self.stream_condition_list[i][trim_len:]


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        input_dim=256,
        mean_path=None,
        std_path=None,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        time_conditioning="alpha",  # "alpha" (current TR) or "noise" (official FloodDiffusion)
        context_pad_to_text_len=False,
        causal=False,
        rope_channel_split=[1, 0, 0],
        spatial_shape=(1, 1),
        prediction_type="vel",  # "vel", "x0", "eps"
        loss_reduction="mean",  # "mean" (current TR) or "sum" (official FloodDiffusion)
        text_config={
            "len": 512,
            "dim": 4096,
        },
        schedule_config={
            "noise_type": "linear",
            "chunk_size": 5,
            "steps": 10,
            "extra_len": 4,
            "random_epsilon": 0.00,
        },
        cfg_config={
            "text_scale": 5.0,
            "null_scale": -4.0,
        },
        input_keys={
            "feature": "feature",
            "feature_length": "feature_length",
            "text": "text",
            "text_end": "text_end",
        },
        # Resampling forcing. See notes.md Phase D and models/resampler.py.
        resampling_ratio: float = 0.0,      # 0 = off, 1 = every sample resampled
        warmup_steps: int = 0,              # only resample once global_step >= warmup_steps
        resample_len: int = 5,              # replace last N latents of the context (= chunk_size by default)
        resample_shift: float = 0.6,        # logit-normal shift for resample_t sampling
        resample_source=None,               # list: ["model"] | ["phc"] | ["model","phc"]; defaults to ["model"]
        resample_clamp_mode: str = "frontier",  # frontier | random | none
        history_augmentation: str = "none",     # none | gaussian | two_forward
        history_noise_std: float = 0.05,
        two_forward_max_steps: int = 30000,
        two_forward_noise_alpha: float = 0.5,
        resample_time_alignment: str = "legacy_ceil",
    ):
        super().__init__()
        self.input_keys = input_keys

        self.mean_path = mean_path
        self.std_path = std_path
        self.input_dim = input_dim
        self.spatial_shape = tuple(spatial_shape)
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        if time_conditioning not in ("alpha", "noise"):
            raise ValueError(
                "time_conditioning must be either 'alpha' or 'noise' "
                f"(got {time_conditioning!r})"
            )
        self.time_conditioning = time_conditioning
        self.context_pad_to_text_len = bool(context_pad_to_text_len)
        self.causal = causal
        self.rope_channel_split = rope_channel_split
        self.prediction_type = prediction_type
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                "loss_reduction must be either 'mean' or 'sum' "
                f"(got {loss_reduction!r})"
            )
        self.loss_reduction = loss_reduction
        self.cfg_config = cfg_config
        self.schedule_config = schedule_config
        self.time_scheduler = TriangularTimeScheduler(schedule_config)

        # Resampling forcing knobs, consumed in forward().
        self.resampling_ratio = resampling_ratio
        self.warmup_steps = warmup_steps
        self.resample_len = resample_len
        self.resample_shift = resample_shift
        if resample_clamp_mode not in ("frontier", "random", "none"):
            raise ValueError(
                "resample_clamp_mode must be one of {'frontier', 'random', 'none'} "
                f"(got {resample_clamp_mode!r})"
            )
        if history_augmentation not in ("none", "gaussian", "two_forward"):
            raise ValueError(
                "history_augmentation must be one of "
                "{'none', 'gaussian', 'two_forward'} "
                f"(got {history_augmentation!r})"
            )
        if history_noise_std < 0:
            raise ValueError(
                f"history_noise_std must be non-negative (got {history_noise_std})"
            )
        if two_forward_max_steps <= 0:
            raise ValueError(
                f"two_forward_max_steps must be positive (got {two_forward_max_steps})"
            )
        if not 0 < two_forward_noise_alpha < 1:
            raise ValueError(
                "two_forward_noise_alpha must be in (0, 1) "
                f"(got {two_forward_noise_alpha})"
            )
        if resampling_ratio > 0 and history_augmentation != "none":
            raise ValueError(
                "Resampling forcing and history_augmentation are separate "
                "ablations and cannot be enabled in the same run."
            )
        self.resample_clamp_mode = resample_clamp_mode
        if resample_time_alignment not in ("legacy_ceil", "exact"):
            raise ValueError(f"Unknown replay time alignment: {resample_time_alignment!r}")
        self.resample_time_alignment = resample_time_alignment
        self.history_augmentation = history_augmentation
        self.history_noise_std = float(history_noise_std)
        self.two_forward_max_steps = int(two_forward_max_steps)
        self.two_forward_noise_alpha = float(two_forward_noise_alpha)
        # Legacy string form ("model" / "phc") is normalized to a list for
        # backward compatibility with old configs/checkpoints.
        if resample_source is None:
            resample_source = ["model"]
        elif isinstance(resample_source, str):
            resample_source = [resample_source]
        else:
            resample_source = list(resample_source)
        self.resample_source = resample_source
        if self.resample_time_alignment == "exact" and resample_source != ["model"]:
            raise NotImplementedError("Exact replay time alignment is validated for model-only TR")
        self.resampler = None  # lazily built in forward() when first needed
        # Cross-attention module (text)
        self.text_module = T5TextCrossModule(**text_config)

        if self.mean_path is not None:
            self.register_buffer(
                "mean", torch.from_numpy(np.load(self.mean_path)).float()
            )
        else:
            self.register_buffer("mean", torch.zeros(input_dim))

        if self.std_path is not None:
            self.register_buffer(
                "std", torch.from_numpy(np.load(self.std_path)).float()
            )
        else:
            self.register_buffer("std", torch.ones(input_dim))

        self.model = WanModel(
            patch_size=(1, 1, 1),
            text_len=self.text_module.len,
            text_dim=self.text_module.dim,
            cross_attn_norm=self.text_module.cross_attn_norm,
            cross_rope=self.text_module.cross_rope,
            in_dim=self.input_dim,
            dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            out_dim=self.input_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            eps=1e-6,
            causal=self.causal,
            rope_channel_split=self.rope_channel_split,
            time_conditioning=self.time_conditioning,
            time_conditioning_scale=self.time_embedding_scale,
            context_pad_to_text_len=self.context_pad_to_text_len,
        )
        self.param_dtype = torch.float32

    def _extract_inputs(self, x):
        """Extract inputs from x using input_keys mapping."""
        inputs = {}
        for internal_key, external_key in self.input_keys.items():
            if external_key in x:
                inputs[internal_key] = x[external_key]
        return inputs

    def preprocess(self, x):
        """Convert last-channel format to channel-first, padding to 4D (C, T, H, W).
        (T, C) -> (C, T, 1, 1)
        (T, H, C) -> (C, T, H, 1)
        (T, H, W, C) -> (C, T, H, W)
        """
        for i in range(len(x)):
            ndim = x[i].ndim
            if ndim == 2:  # (T, C)
                x[i] = x[i].permute(1, 0)[:, :, None, None]
            elif ndim == 3:  # (T, H, C)
                x[i] = x[i].permute(2, 0, 1)[:, :, :, None]
            elif ndim == 4:  # (T, H, W, C)
                x[i] = x[i].permute(3, 0, 1, 2)
        return x

    def postprocess(self, x):
        """Reverse of preprocess: channel-first 4D back to last-channel, stripping padding dims.
        (C, T, 1, 1) -> (T, C)
        (C, T, H, 1) -> (T, H, C)
        (C, T, H, W) -> (T, H, W, C)
        """
        for i in range(len(x)):
            shape = x[i].shape  # (C, T, H, W)
            if shape[2] == 1 and shape[3] == 1:  # (C, T, 1, 1) -> (T, C)
                x[i] = x[i][:, :, 0, 0].permute(1, 0)
            elif shape[3] == 1:  # (C, T, H, 1) -> (T, H, C)
                x[i] = x[i][:, :, :, 0].permute(1, 2, 0)
            else:  # (C, T, H, W) -> (T, H, W, C)
                x[i] = x[i].permute(1, 2, 3, 0)
        return x

    def forward(self, x):
        # Preserve per-sample metadata for Resampler (PHC mid-motion RSI) and
        # for the debug viz dump (dataset label, global_step) before
        # _extract_inputs strips non-mapped keys. Each is None-or-list.
        meta_motion_ids = x.get("name")        # List[str], e.g. "000000"/"M000000"
        meta_datasets = x.get("dataset")       # List[str], e.g. "HumanML3D"/"BABEL"
        meta_global_step = x.get("global_step", None)  # int when set by train_ldf._step
        meta_raw_feature_lengths = x.get("feature_length", None)
        x = self._extract_inputs(x)
        feature_original = x["feature"]  # (B, T, C)
        feature_length = x["feature_length"]  # (B,)
        feature_original = (feature_original - self.mean) / self.std
        batch_size = feature_original.shape[0]
        seq_len = feature_original.shape[1]
        device = feature_original.device
        feature = []
        valid_len = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), seq_len)
            valid_len.append(length)
            feature.append(feature_original[i, :length, ...])

        # Preprocess to (C, T, 1, 1) per sample
        feature = self.preprocess(feature)

        # get time steps (per-sample t) and windows first — resampling needs
        # context_end (= output_start_index) before we can mutate feature.
        time_steps = self.time_scheduler.get_time_steps(device, valid_len)  # (B,)
        input_start_index, input_end_index, output_start_index, output_end_index = (
            self.time_scheduler.get_windows(valid_len, time_steps)
        )

        # Get context from text cross module once, up front — resampling and
        # the main forward share the same encoded context (get_context has
        # random text drop_out inside, so we must not call it twice).
        context, _ = self.text_module.get_context(
            x,
            valid_len,
            device,
            self.param_dtype,
            training=True,
        )

        # -------- Resampling forcing (Phase D) --------
        # Lazily construct self.resampler the first time we'd actually touch it.
        # Safe when resampling_ratio=0: the guard below stays False forever.
        # global_step survives _extract_inputs only via meta_global_step;
        # x.get fallback is for legacy callers that mapped it through input_keys.
        global_step = meta_global_step if meta_global_step is not None else x.get(
            "global_step", float("inf")
        )
        # Debug viz dump trigger: global-rank-0 only, periodic, and only when
        # training-time resample actually fires (so the dump captures a real
        # replacement). Stays False when global_step is non-numeric. The rank0
        # gate reads _dbg_train_is_rank0 (set by train_ldf._wire_debug_viz from
        # Lightning's is_global_zero), NOT os.environ — under the PHC multi-node
        # launch every process has LOCAL_RANK=0, so the env var made all N ranks
        # dump (N× duplicate files). Defaults True so single-process runs that
        # never set the attr still dump.
        dump_every = getattr(self, "_dbg_train_dump_every", 0) or 0
        enable_dbg_train = (
            self.training
            and self.resampling_ratio > 0
            and isinstance(global_step, int)
            and global_step >= self.warmup_steps
            and dump_every > 0
            and global_step % dump_every == 0
            and getattr(self, "_dbg_train_is_rank0", True)
        )
        if enable_dbg_train:
            force_ctx = int(getattr(self, "_dbg_train_force_context_end", 0) or 0)
            force_ctx_by_name = getattr(
                self, "_dbg_train_force_context_end_by_name", {}) or {}
            if force_ctx > 0 or force_ctx_by_name:
                target_width = max(1, int(getattr(self.time_scheduler, "chunk_size", 1)))
                for b, length in enumerate(valid_len):
                    force_ctx_b = force_ctx
                    if meta_motion_ids is not None and force_ctx_by_name:
                        sid = str(meta_motion_ids[b])
                        if sid in force_ctx_by_name:
                            force_ctx_b = int(force_ctx_by_name[sid])
                    if length <= 1:
                        continue
                    if force_ctx_b <= 0:
                        continue
                    ctx = min(force_ctx_b, length - 1)
                    end = min(length, ctx + target_width)
                    if end <= ctx:
                        ctx = max(0, length - 1)
                        end = length
                    input_start_index[b] = 0
                    input_end_index[b] = end
                    output_start_index[b] = ctx
                    output_end_index[b] = end
        feature_pre_resample = None  # set inside the resample branch below
        if (
            self.training
            and self.resampling_ratio > 0
            and global_step >= self.warmup_steps
        ):
            if self.resampler is None:
                # Lazy construction only supports pure ["model"] (no external
                # VAE / PHC backend needed). Any sources containing "phc"
                # must be attached by train_ldf.py:initialize_metrics with a
                # VAE reference and phc_config; if resampler is still None
                # here, the config is inconsistent.
                if self.resample_source != ["model"]:
                    raise RuntimeError(
                        f"resample_source={self.resample_source} requires an "
                        f"externally-constructed Resampler attached as "
                        f"model.resampler; lazy construction only supports ['model']."
                    )
                from methods.fd1.models.resampler import Resampler
                self.resampler = Resampler(
                    sources=["model"],
                    shift=self.resample_shift,
                    resample_len=self.resample_len,
                    prediction_type=self.prediction_type,
                    time_scheduler=self.time_scheduler,
                    time_embedding_scale=self.time_embedding_scale,
                    clamp_mode=self.resample_clamp_mode,
                    time_alignment=self.resample_time_alignment,
                )
            if enable_dbg_train and self.resampler is not None:
                self.resampler._dbg_train = {}
            if "phc" in self.resample_source and self.resampler is not None:
                self.resampler._set_latent_norm_from_model(self)
            # Resampling-forcing: snapshot the pre-resample latents so the
            # loss target (x0) below uses the original GT, while xt is built
            # from the post-resample feature (= model self-rollout state).
            feature_pre_resample = [f.clone() for f in feature]
            feature = self.resampler.resample_batch(
                feature,
                context_end=output_start_index,
                output_end=output_end_index,
                time_steps=time_steps,
                text_context=context,
                model=self.model,
                resampling_ratio=self.resampling_ratio,
            )
            if enable_dbg_train and self.resampler is not None:
                train_capture = self.resampler._dbg_train or {}
                self.resampler._dbg_train = None
                try:
                    from methods.fd1.utils.debug_viz import (
                        decode_latent_to_138,
                        render_phc_chain,
                        render_resample_diff,
                    )
                    vae = getattr(self, "_dbg_train_vae", None)
                    mesh_renderer = getattr(self, "_dbg_train_mesh_renderer", None)
                    save_dir = getattr(self, "_dbg_train_save_dir", "./outputs/_dbg")
                    dataset_filter = getattr(self, "_dbg_train_dataset_filter", None)
                    require_full_resample = bool(
                        getattr(self, "_dbg_train_require_full_resample", False))
                    n_sample_per_dataset = int(
                        getattr(self, "_dbg_train_n_sample_per_dataset", 1))
                    text_dirs = getattr(self, "_dbg_train_text_dirs", None) or {}

                    windows_all = train_capture.get("windows", [])
                    ranges_all = train_capture.get("resample_ranges", [])
                    latent_pre = train_capture.get("latent_pre_resample") or []
                    latent_post = train_capture.get("latent_post_resample") or []

                    # Render up to n_sample_per_dataset videos PER DATASET (keyed
                    # by dataset name) so HumanML3D and BABEL each get
                    # representation instead of the first-N-in-batch, which could
                    # be all one dataset. =1 → one video per dataset.
                    dumped_per_ds = {}
                    for i in range(len(latent_pre)):
                        ds = (str(meta_datasets[i])
                              if meta_datasets is not None else "unknown")
                        if dumped_per_ds.get(ds, 0) >= n_sample_per_dataset:
                            continue
                        ctx_end = int(output_start_index[i])
                        if ctx_end <= 0:
                            continue
                        if (dataset_filter is not None and meta_datasets is not None
                                and str(meta_datasets[i]) != str(dataset_filter)):
                            continue
                        if (require_full_resample
                                and ctx_end < self.resampler.resample_len):
                            continue
                        sample_windows = [w for w in windows_all
                                          if w.get("sample_i") == i]
                        sample_range = next(
                            (r for r in ranges_all if r.get("sample_i") == i),
                            None,
                        )
                        # Skip if neither PHC windows (chain viz) nor resample
                        # range (no-PHC 2-col diff) is available for this sample.
                        if not sample_windows and sample_range is None:
                            continue
                        try:
                            col_left = decode_latent_to_138(
                                latent_pre[i], self, vae)
                            col_right = decode_latent_to_138(
                                latent_post[i], self, vae)
                        except Exception as e:
                            print(f"[debug_viz] step={int(global_step)} sample "
                                  f"{i} decode failed: {e}")
                            continue
                        sid = (str(meta_motion_ids[i])
                               if meta_motion_ids is not None else f"idx{i}")
                        sid_safe = sid.replace("/", "_")
                        out_mp4 = os.path.join(
                            save_dir, "debug_viz",
                            f"step_{int(global_step):07d}",
                            f"{ds}_{sid_safe}.mp4",
                        )
                        try:
                            if sample_windows:
                                render_phc_chain(
                                    out_mp4=out_mp4,
                                    windows=sample_windows,
                                    col_left_motion_138=col_left,
                                    col_left_label="clean vae",
                                    col_right_motion_138=col_right,
                                    col_right_label="after resample",
                                    mesh_renderer=mesh_renderer,
                                    text_dir=text_dirs.get(ds),
                                    motion_id=sid,
                                    max_render_seconds=getattr(
                                        self, "_dbg_train_max_render_seconds", 180),
                                )
                                viz_kind = f"chain rows={len(sample_windows)}"
                            else:
                                render_resample_diff(
                                    out_mp4=out_mp4,
                                    sample_range=sample_range,
                                    col_left_motion_138=col_left,
                                    col_left_label="clean vae",
                                    col_right_motion_138=col_right,
                                    col_right_label="after resample",
                                    mesh_renderer=mesh_renderer,
                                    text_dir=text_dirs.get(ds),
                                    motion_id=sid,
                                )
                                viz_kind = "diff 1x2"
                        except Exception as e:
                            print(f"[debug_viz] step={int(global_step)} sample "
                                  f"{i} render failed: {e}")
                            continue
                        dumped_per_ds[ds] = dumped_per_ds.get(ds, 0) + 1
                        print(f"[debug_viz] step={int(global_step)} dumped "
                              f"{ds}_{sid_safe} {viz_kind} "
                              f"→ {out_mp4}", flush=True)
                except Exception as e:
                    print(f"[debug_viz] render_phc_chain orchestration failed at "
                          f"step={int(global_step)}: {e}")
        # -------- End resampling forcing --------

        # Build the noise schedule. On a resampling-forcing step the
        # post-resample feature already sits on the triangular schedule (clean
        # prefix at α=1, transition zone at α<1 from the no-grad rollout), so the
        # self-forcing loss uses each rolled-out sample's feature DIRECTLY as xt
        # and regresses toward the velocity that flows it back to GT — no second
        # round of noising. Each actually rolled-out sample therefore keeps an
        # UNjittered schedule so β matches the α left by replay. Samples not
        # selected by the ratio gate retain the ordinary per-sample schedule
        # jitter and noised-GT training path.
        is_resample_step = feature_pre_resample is not None
        # Per-sample self-forcing mask: True only for samples the resampler
        # actually rolled out. This is deliberately computed before schedule
        # jitter so ratio<1 runs do not silently remove jitter from the
        # untriggered part of the batch.
        self_forcing_mask = None
        if is_resample_step:
            resampled = getattr(self.resampler, "_last_resampled_idx", None) or set()
            self_forcing_mask = [b in resampled for b in range(batch_size)]
        time_schedules, _ = self.time_scheduler.get_time_schedules(
            device, valid_len, time_steps, training=False
        )  # (B, T)
        schedule_jitter = float(self.time_scheduler.random_epsilon)
        if schedule_jitter > 0:
            for b in range(batch_size):
                is_self_forced = (
                    self_forcing_mask is not None and self_forcing_mask[b]
                )
                if not is_self_forced:
                    time_schedules[b] = torch.clamp(
                        time_schedules[b]
                        + torch.randn_like(time_schedules[b]) * schedule_jitter,
                        min=0.0,
                        max=1.0,
                    )
        alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta = (
            self.time_scheduler.get_noise_levels(device, valid_len, time_schedules)
        )  # (B, T)
        # x0_source points the loss target at the pre-resample original GT
        # (None when no resample happened → x0 = GT).
        x0, eps, xt = self.time_scheduler.add_noise(
            feature,
            alpha,
            beta,
            input_start_index,
            input_end_index,
            output_start_index,
            output_end_index,
            training=True,
            x0_source=feature_pre_resample,
            self_forcing_mask=self_forcing_mask,
        )

        # Slice per-frame context to match input window
        if isinstance(context[0], (list, tuple)):
            context = [
                context[i][input_start_index[i] : input_end_index[i]]
                for i in range(batch_size)
            ]

        # Slice time_schedules per sample
        time_schedules_input = [
            time_schedules[i][input_start_index[i] : input_end_index[i]]
            * self.time_embedding_scale
            for i in range(batch_size)
        ]
        model_seq_len = max(
            input_end_index[i] - input_start_index[i]
            for i in range(batch_size)
        )

        # History-only baselines. They leave the supervised triangular output
        # band and its clean-GT velocity target unchanged.
        augmentation_rate = 0.0
        if self.training and self.history_augmentation == "gaussian":
            for b in range(batch_size):
                history_len = output_start_index[b] - input_start_index[b]
                if history_len <= 0:
                    continue
                xt[b][:, :history_len, ...] = (
                    xt[b][:, :history_len, ...]
                    + self.history_noise_std
                    * torch.randn_like(xt[b][:, :history_len, ...])
                )
            augmentation_rate = 1.0
        elif self.training and self.history_augmentation == "two_forward":
            # A triangular-model adaptation of MotionStreamer's Two-Forward:
            # first predict clean history latents from a fixed noisy probe,
            # then detach and randomly replace clean history tokens before the
            # second, gradient-carrying training forward. The replacement
            # probability follows the original cosine curriculum (0 -> 1).
            progress = min(
                max(float(global_step), 0.0) / float(self.two_forward_max_steps),
                1.0,
            )
            replace_prob = 0.5 * (1.0 - math.cos(math.pi * progress))
            probe_xt = [x_i.clone() for x_i in xt]
            probe_schedule = [s_i.clone() for s_i in time_schedules_input]
            probe_alpha = self.two_forward_noise_alpha
            for b in range(batch_size):
                history_len = output_start_index[b] - input_start_index[b]
                if history_len <= 0:
                    continue
                clean_history = xt[b][:, :history_len, ...]
                probe_xt[b][:, :history_len, ...] = (
                    probe_alpha * clean_history
                    + (1.0 - probe_alpha) * torch.randn_like(clean_history)
                )
                probe_schedule[b][:history_len] = (
                    probe_alpha * self.time_embedding_scale
                )
            with torch.no_grad():
                probe_prediction = self.model(
                    probe_xt,
                    probe_schedule,
                    context,
                    model_seq_len,
                    y=None,
                )
            replaced = 0
            eligible = 0
            for b in range(batch_size):
                history_len = output_start_index[b] - input_start_index[b]
                if history_len <= 0:
                    continue
                predicted_clean = (
                    probe_xt[b][:, :history_len, ...]
                    + (1.0 - probe_alpha)
                    * probe_prediction[b][:, :history_len, ...]
                ).detach()
                replace_mask = (
                    torch.rand(history_len, device=device) < replace_prob
                )[None, :, None, None]
                xt[b][:, :history_len, ...] = torch.where(
                    replace_mask,
                    predicted_clean,
                    xt[b][:, :history_len, ...],
                )
                replaced += int(replace_mask.sum().item())
                eligible += history_len
            augmentation_rate = replaced / max(eligible, 1)

        # Through WanModel
        predicted_result = self.model(
            xt,
            time_schedules_input,
            context,
            model_seq_len,
            y=None,
        )  # (B, C, T, 1, 1)

        loss = 0.0
        for b in range(batch_size):
            pred_os = output_start_index[b] - input_start_index[b]
            pred_oe = output_end_index[b] - input_start_index[b]
            dalpha_i = dalpha[b][output_start_index[b] : output_end_index[b]]
            dbeta_i = dbeta[b][output_start_index[b] : output_end_index[b]]
            if self.prediction_type == "vel":
                vel = (
                    x0[b] * dalpha_i[None, :, None, None]
                    + eps[b] * dbeta_i[None, :, None, None]
                )  # (C, output_length, 1, 1)
                squared_error = (
                    predicted_result[b][:, pred_os:pred_oe, ...] - vel
                ) ** 2
            elif self.prediction_type == "x0":
                squared_error = (
                    predicted_result[b][:, pred_os:pred_oe, ...] - x0[b]
                ) ** 2
            elif self.prediction_type == "eps":
                squared_error = (
                    predicted_result[b][:, pred_os:pred_oe, ...] - eps[b]
                ) ** 2
            sample_loss = (
                squared_error.sum()
                if self.loss_reduction == "sum"
                else squared_error.mean()
            )
            loss += sample_loss
        loss = loss / batch_size
        loss_dict = {"total": loss, "mse": loss}
        if self.history_augmentation != "none":
            loss_dict["history_augmentation/triggered_rate"] = torch.tensor(
                augmentation_rate, device=device
            )

        # Expose per-step resampling stats for wandb logging.
        if self.resampler is not None:
            stats = self.resampler.get_stats()
            loss_dict["resampling/triggered_rate"] = torch.tensor(
                stats["triggered_rate"], device=device
            )
            loss_dict["resampling/avg_resample_t"] = torch.tensor(
                stats["avg_resample_t"], device=device
            )
            if "phc_fail_rate" in stats:
                loss_dict["resampling/phc_fail_rate"] = torch.tensor(
                    stats["phc_fail_rate"], device=device
                )
            if "phc_avg_latency_ms" in stats:
                loss_dict["resampling/phc_avg_latency_ms"] = torch.tensor(
                    stats["phc_avg_latency_ms"], device=device
                )
            for key, value in stats.items():
                if key in (
                    "triggered_rate",
                    "avg_resample_t",
                    "phc_fail_rate",
                    "phc_avg_latency_ms",
                ):
                    continue
                if key.startswith("phc_"):
                    loss_dict[f"resampling/{key}"] = torch.tensor(
                        float(value), device=device
                    )
            self.resampler.reset_stats()

        return loss_dict

    def generate(self, x):
        """
        Generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        x = self._extract_inputs(x)
        extra_len = self.schedule_config.get("extra_len", 0)
        feature_length = x["feature_length"]  # (B,)
        batch_size = len(feature_length)
        seq_len = max(feature_length).item() + extra_len
        device = next(self.parameters()).device
        valid_len = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), seq_len)
            valid_len.append(length)
        generated_len = [seq_len for _ in range(batch_size)]

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len, *self.spatial_shape, self.input_dim, device=device
        )
        generated = [generated[i] for i in range(batch_size)]
        generated = self.preprocess(generated)

        # Precompute text and null contexts for CFG
        text_context, metadata = self.text_module.get_context(
            x,
            generated_len,
            device,
            self.param_dtype,
            training=False,
        )
        null_context = self.text_module.get_null_context(
            batch_size, device, self.param_dtype
        )
        full_text = metadata["full_text"]

        total_steps = self.time_scheduler.get_total_steps(seq_len)
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # get time steps and noise levels
            time_steps = self.time_scheduler.get_time_steps(
                device, generated_len, step
            )  # (B,)
            time_schedules, time_schedules_derivative = (
                self.time_scheduler.get_time_schedules(
                    device, generated_len, time_steps
                )
            )  # (B, T)
            alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta = (
                self.time_scheduler.get_noise_levels(
                    device, generated_len, time_schedules
                )
            )  # (B, T)
            input_start_index, input_end_index, output_start_index, output_end_index = (
                self.time_scheduler.get_windows(generated_len, time_steps)
            )
            _, _, xt = self.time_scheduler.add_noise(
                generated,
                alpha,
                beta,
                input_start_index,
                input_end_index,
                output_start_index,
                output_end_index,
                training=False,
            )

            # Slice time_schedules per sample
            time_schedules_input = [
                time_schedules[i][input_start_index[i] : input_end_index[i]]
                * self.time_embedding_scale
                for i in range(batch_size)
            ]
            model_seq_len = max(
                input_end_index[i] - input_start_index[i]
                for i in range(batch_size)
            )

            # Slice per-frame context to match input window
            if isinstance(text_context[0], (list, tuple)):
                window_text_context = [
                    text_context[i][input_start_index[i] : input_end_index[i]]
                    for i in range(batch_size)
                ]
            else:
                window_text_context = text_context

            # CFG: text_scale * pred_text + null_scale * pred_null
            pred_text = self.model(
                xt,
                time_schedules_input,
                window_text_context,
                model_seq_len,
                y=None,
            )
            pred_null = self.model(
                xt,
                time_schedules_input,
                null_context,
                model_seq_len,
                y=None,
            )
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            for i in range(batch_size):
                os, oe = output_start_index[i], output_end_index[i]
                pred_os = os - input_start_index[i]
                pred_oe = oe - input_start_index[i]
                predicted_result_i = predicted_result[i][:, pred_os:pred_oe, ...]
                generated_i = generated[i][:, os:oe, ...]
                dt = time_schedules_derivative[i][os:oe][None, :, None, None]
                alpha_i = alpha[i][os:oe][None, :, None, None]
                dalpha_i = dalpha[i][os:oe][None, :, None, None]
                beta_i = beta[i][os:oe][None, :, None, None]
                dbeta_i = dbeta[i][os:oe][None, :, None, None]
                sigma_i = sigma[i][os:oe][None, :, None, None]
                dlog_alpha_i = dlog_alpha[i][os:oe][None, :, None, None]
                dlog_beta_i = dlog_beta[i][os:oe][None, :, None, None]
                if self.prediction_type == "vel":
                    vel = predicted_result_i
                elif self.prediction_type == "x0":
                    vel = (
                        predicted_result_i * (-dlog_beta_i * alpha_i + dalpha_i)
                        + generated_i * dlog_beta_i
                    )
                elif self.prediction_type == "eps":
                    vel = (
                        predicted_result_i * (-dlog_alpha_i * beta_i + dbeta_i)
                        + generated_i * dlog_alpha_i
                    )
                st = (vel - generated_i * dlog_alpha_i) / (
                    (beta_i * dlog_alpha_i - dbeta_i) * beta_i
                )
                generated[i][:, os:oe, ...] += (
                    vel * dt
                    + st * 0.5 * sigma_i**2 * dt
                    + sigma_i * torch.sqrt(dt) * torch.randn_like(generated_i)
                )

        generated = self.postprocess(generated)  # list of (T, C)
        y_hat_out = []
        for i in range(batch_size):
            single_generated = generated[i][: valid_len[i], :] * self.std + self.mean
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text

        return out

    def init_generated(self, seq_len, batch_size=1, schedule_config={}):
        """Initialize streaming generation state.

        Args:
            seq_len: Model window size (how many frames WanModel processes per step).
            schedule_config: Optional schedule config overrides.

        Buffer is 2*seq_len. Model window is always buffer[0:seq_len].
        When conditions overflow seq_len, shift buffer by seq_len and restart.
        """
        self.schedule_config.update(schedule_config)
        content_len = self.schedule_config.get("content_len", None)
        if content_len is None:
            self.schedule_config["content_len"] = seq_len
        else:
            self.schedule_config["content_len"] = min(seq_len, content_len)
        self.time_scheduler = TriangularTimeScheduler(self.schedule_config)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buf_len = seq_len * 2
        self.current_step = 0
        self.current_commit = 0
        self.condition_frames = 0

        device = next(self.parameters()).device
        # Initialize entire buffer as pure noise
        generated = torch.randn(
            batch_size, self.buf_len, *self.spatial_shape, self.input_dim, device=device
        )
        generated = [generated[i] for i in range(batch_size)]
        self.generated = self.preprocess(generated)

        # Initialize streaming state for cross module
        self.text_module.init_stream(self.batch_size)

    def _rollback(self):
        """Shift buffer by seq_len when conditions overflow the window."""
        for i in range(self.batch_size):
            self.generated[i][:, : self.seq_len, ...] = self.generated[i][
                :, self.seq_len :, ...
            ].clone()
            self.generated[i][:, self.seq_len :, ...] = torch.randn_like(
                self.generated[i][:, self.seq_len :, ...]
            )
        self.current_step -= self.time_scheduler.get_step_rollback(self.seq_len)
        self.condition_frames -= self.seq_len
        self.current_commit -= self.seq_len
        self.text_module.trim_stream(self.seq_len)

    @torch.no_grad()
    def stream_generate_step(self, x):
        """
        Streaming generation step. Each call provides 1 frame of conditions.
        The scheduler determines committable frames from accumulated conditions.

        Returns:
            dict with "generated": list of one (N, C) tensor, or [] if nothing to commit.
        """
        x = self._extract_inputs(x)
        device = next(self.parameters()).device
        self.generated = [g.to(device) for g in self.generated]

        # 1. Update conditions (1 frame per call)
        self.text_module.update_stream(x, device, self.param_dtype)
        self.condition_frames += 1

        # 2. Rollback if conditions overflow the window
        if self.condition_frames > self.buf_len:
            self._rollback()

        # 3. Determine how many frames can be committed
        committable_length, committable_steps = self.time_scheduler.get_committable(
            self.condition_frames
        )
        while self.current_step < committable_steps:
            time_steps = self.time_scheduler.get_time_steps(
                device, [self.buf_len] * self.batch_size, self.current_step
            )
            time_schedules, time_schedules_derivative = (
                self.time_scheduler.get_time_schedules(
                    device, [self.buf_len] * self.batch_size, time_steps
                )
            )
            alpha, dalpha, beta, dbeta, sigma, dlog_alpha, dlog_beta = (
                self.time_scheduler.get_noise_levels(
                    device, [self.buf_len] * self.batch_size, time_schedules
                )
            )
            is_, ie_, os_, oe_ = self.time_scheduler.get_windows(
                [self.buf_len] * self.batch_size, time_steps
            )
            _, _, xt = self.time_scheduler.add_noise(
                self.generated, alpha, beta, is_, ie_, os_, oe_, training=False
            )

            time_schedules_input = [
                time_schedules[0][is_[0] : ie_[0]] * self.time_embedding_scale
            ] * self.batch_size
            model_seq_len = max(ie_[i] - is_[i] for i in range(self.batch_size))

            # CFG: text_scale * pred_text + null_scale * pred_null
            text_context = self.text_module.get_stream_context(is_[0], ie_[0])
            null_context = self.text_module.get_null_context(
                self.batch_size, device, self.param_dtype
            )
            pred_text = self.model(
                xt,
                time_schedules_input,
                text_context,
                model_seq_len,
                y=None,
            )
            pred_null = self.model(
                xt,
                time_schedules_input,
                null_context,
                model_seq_len,
                y=None,
            )
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            os_idx, oe_idx = os_[0], oe_[0]
            dt = time_schedules_derivative[0][os_idx:oe_idx][None, :, None, None]
            pred_os_idx = os_idx - is_[0]
            pred_oe_idx = oe_idx - is_[0]
            alpha_i = alpha[0][os_idx:oe_idx][None, :, None, None]
            dalpha_i = dalpha[0][os_idx:oe_idx][None, :, None, None]
            beta_i = beta[0][os_idx:oe_idx][None, :, None, None]
            dbeta_i = dbeta[0][os_idx:oe_idx][None, :, None, None]
            sigma_i = sigma[0][os_idx:oe_idx][None, :, None, None]
            dlog_alpha_i = dlog_alpha[0][os_idx:oe_idx][None, :, None, None]
            dlog_beta_i = dlog_beta[0][os_idx:oe_idx][None, :, None, None]
            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i][
                    :, pred_os_idx:pred_oe_idx, ...
                ]
                generated_i = self.generated[i][:, os_idx:oe_idx, ...]
                if self.prediction_type == "vel":
                    vel = predicted_result_i
                elif self.prediction_type == "x0":
                    vel = (
                        predicted_result_i * (-dlog_beta_i * alpha_i + dalpha_i)
                        + generated_i * dlog_beta_i
                    )
                elif self.prediction_type == "eps":
                    vel = (
                        predicted_result_i * (-dlog_alpha_i * beta_i + dbeta_i)
                        + generated_i * dlog_alpha_i
                    )
                st = (vel - generated_i * dlog_alpha_i) / (
                    (beta_i * dlog_alpha_i - dbeta_i) * beta_i
                )
                self.generated[i][:, os_idx:oe_idx, ...] += (
                    vel * dt
                    + st * 0.5 * sigma_i**2 * dt
                    + sigma_i * torch.sqrt(dt) * torch.randn_like(generated_i)
                )
            self.current_step += 1

        # 5. Extract newly committed frames
        if self.current_commit < committable_length:
            output = [
                self.generated[i][:, self.current_commit : committable_length, ...]
                for i in range(self.batch_size)
            ]
            output = self.postprocess(output)
            output = [o * self.std + self.mean for o in output]
            self.current_commit = committable_length
            return {"generated": output}
        else:
            empty = [
                torch.zeros(self.input_dim, 0, *self.spatial_shape, device=device)
                for _ in range(self.batch_size)
            ]
            empty = self.postprocess(empty)
            return {"generated": empty}
