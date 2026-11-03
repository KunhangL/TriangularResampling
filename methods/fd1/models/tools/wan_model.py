# This module uses modified code from Alibaba Wan Team
# Original source: https://github.com/Wan-Video/Wan2.2
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified to support stream mode for cross-attention.
# Added causal attention for self-attention (1d case)
# Added context length corrrection.

import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, channel_split=None, token_lens=None):
    n, c = x.size(2), x.size(3) // 2
    # split freqs
    if channel_split is not None:
        part_length = c // (channel_split[0] + channel_split[1] + channel_split[2])
        freqs = freqs.split(
            [
                part_length * channel_split[0],
                part_length * channel_split[1],
                part_length * channel_split[2],
            ],
            dim=1,
        )
    else:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        grid_len = f * h * w
        total_len = token_lens[i].item() if token_lens is not None else grid_len

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :total_len].to(torch.float64).reshape(total_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(grid_len, 1, -1)

        # repeat for local tokens (e.g., L tokens per grid position share same RoPE)
        if total_len > grid_len:
            freqs_i = freqs_i.repeat_interleave(total_len // grid_len, dim=0)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, total_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, causal=False
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.causal = causal
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, rope_channel_split=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs, rope_channel_split),
            k=rope_apply(k, grid_sizes, freqs, rope_channel_split),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            causal=self.causal,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_rope=False,
        eps=1e-6,
        causal=False,
    ):
        assert dim % num_heads == 0
        super().__init__(
            dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, causal=False
        )
        self.cross_rope = cross_rope

    def forward(
        self,
        x,
        context,
        context_lens,
        grid_sizes=None,
        content_grid_sizes=None,
        freqs=None,
        rope_channel_split=None,
    ):
        r"""
        Args non-stream mode:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        Args stream mode:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [BxL1, L2, C]
            context_lens(Tensor): Shape [BxL1]
        """
        out_sizes = x.size()
        b, n, d = context.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if self.cross_rope:
            assert q.size(0) == grid_sizes.size(0), (
                "cross_rope does not support per-frame context"
            )
            q = rope_apply(q, grid_sizes, freqs, rope_channel_split)
            k = rope_apply(
                k,
                content_grid_sizes,
                freqs,
                rope_channel_split,
                token_lens=context_lens,
            )

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2).view(*out_sizes)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        cross_rope=False,
        eps=1e-6,
        causal=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.causal = causal
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps, causal
        )
        self.cross_attn_norm_layer = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, cross_rope, eps, causal
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        rope_channel_split,
        context,
        context_lens,
        content_grid_sizes=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            rope_channel_split,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(
                self.cross_attn_norm_layer(x),
                context,
                context_lens,
                grid_sizes=grid_sizes,
                content_grid_sizes=content_grid_sizes,
                freqs=freqs,
                rope_channel_split=rope_channel_split,
            )
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        text_dim=4096,
        cross_attn_norm=True,
        cross_rope=False,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        causal=False,
        rope_channel_split=None,
        rope_max_seq_len=1024,
        time_conditioning="alpha",
        time_conditioning_scale=1.0,
        context_pad_to_text_len=False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            causal (`bool`, *optional*, defaults to False):
                Enable causal attention for self-attention
            rope_channel_split (`int`, *optional*, defaults to None):
                Channel split for rotary positional embeddings
        """

        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.text_dim = text_dim
        self.cross_attn_norm = cross_attn_norm
        self.cross_rope = cross_rope
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.causal = causal
        self.rope_channel_split = rope_channel_split
        self.rope_max_seq_len = rope_max_seq_len
        if time_conditioning not in ("alpha", "noise"):
            raise ValueError(
                "time_conditioning must be either 'alpha' or 'noise' "
                f"(got {time_conditioning!r})"
            )
        self.time_conditioning = time_conditioning
        self.time_conditioning_scale = float(time_conditioning_scale)
        self.context_pad_to_text_len = bool(context_pad_to_text_len)

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    cross_rope,
                    eps,
                    causal,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads // 2
        self.freqs = self._build_rope_freqs(self.rope_max_seq_len)

        # initialize weights
        self.init_weights()

    def _build_rope_freqs(self, max_seq_len):
        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads // 2
        if self.rope_channel_split is not None:
            assert d % sum(self.rope_channel_split) == 0
            part_length = d // (
                self.rope_channel_split[0]
                + self.rope_channel_split[1]
                + self.rope_channel_split[2]
            )
            return torch.cat(
                [
                    rope_params(max_seq_len, 2 * (part_length * self.rope_channel_split[0])),
                    rope_params(max_seq_len, 2 * (part_length * self.rope_channel_split[1])),
                    rope_params(max_seq_len, 2 * (part_length * self.rope_channel_split[2])),
                ],
                dim=1,
            )
        else:
            return torch.cat(
                [
                    rope_params(max_seq_len, 2 * (d - 2 * (d // 3))),
                    rope_params(max_seq_len, 2 * (d // 3)),
                    rope_params(max_seq_len, 2 * (d // 3)),
                ],
                dim=1,
            )

    def _ensure_rope_freqs(self, required_len, device):
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if required_len <= self.freqs.shape[0]:
            return
        new_len = max(required_len, self.freqs.shape[0] * 2)
        self.freqs = self._build_rope_freqs(new_len).to(device)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor or List[Tensor]):
                Diffusion timesteps. Supports:
                - [B] tensor: uniform time for all frames, auto-expanded to [B, seq_len]
                - [B, T] tensor: per-frame time, auto-padded to [B, seq_len] if T < seq_len
                - List of 1D tensors: variable-length per-frame time, auto-padded to seq_len
            context (List[Tensor] or List[List[Tensor]]):
                Text embeddings. Supports:
                - List[Tensor] (B items): uniform, one context [L, C] per sample → [B, max_L, C]
                - List[List[Tensor]] (B items of variable-length lists): per-frame,
                  auto-padded to seq_len frames per sample then flattened → [B*seq_len, max_L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        self._ensure_rope_freqs(int(grid_sizes.max().item()), device)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings (auto-pad to seq_len, similar to feature padding above)
        if isinstance(t, (list, tuple)):
            # Per-frame time as list of 1D tensors (variable length), pad each to seq_len
            t = torch.stack(
                [
                    torch.cat([ti, ti.new_zeros(seq_len - ti.size(0))])
                    if ti.size(0) < seq_len
                    else ti
                    for ti in t
                ]
            )
        elif t.dim() == 1:  # [B] uniform time for all frames, expand to [B, seq_len]
            t = t.unsqueeze(1).expand(-1, seq_len)
        elif (
            t.dim() == 2 and t.size(1) < seq_len
        ):  # [B, T] where T < seq_len, pad to [B, seq_len]
            t = torch.cat([t, t.new_zeros(t.size(0), seq_len - t.size(1))], dim=1)
        # The enhanced TR code uses denoising progress alpha (0=noise, 1=clean),
        # while the released FloodDiffusion checkpoint was trained with the
        # complementary noise level (1=noise, 0=clean). Keep alpha as the
        # default so existing experiments are unchanged, and convert only for
        # explicitly marked official-checkpoint runs.
        if self.time_conditioning == "noise":
            t = self.time_conditioning_scale - t
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context - auto-detect and auto-pad (similar to feature padding above)
        # List[Tensor]       (B items)  → uniform: one context per sample → [B, max_L, C]
        # List[List[Tensor]] (B items)  → per-frame: variable-length inner lists, pad to seq_len then flatten → [B*seq_len, max_L, C]
        if len(context) > 0 and isinstance(context[0], (list, tuple)):
            # Per-frame: nested list, pad each sample's frame list to seq_len then flatten
            flat_context = []
            for sample_ctx in context:
                assert len(sample_ctx) <= seq_len
                flat_context.extend(sample_ctx)
                if len(sample_ctx) < seq_len:
                    zero_ctx = sample_ctx[0].new_zeros(1, sample_ctx[0].size(-1))
                    flat_context.extend([zero_ctx] * (seq_len - len(sample_ctx)))
            context = flat_context
        # content grid sizes for cross-attention RoPE (before padding, from raw tensor shapes)
        # ndim==2: [L, C] → grid dims = shape[:-1] = (L,), L is positional
        # ndim>=3: [F,(H),(W),L,C] → grid dims = shape[:-2], L is local (not in RoPE)
        if self.cross_rope:
            content_grid_sizes = []
            for u in context:
                grid_dims = u.shape[:-1] if u.ndim == 2 else u.shape[:-2]
                grid = list(grid_dims) + [1] * (3 - len(grid_dims))
                content_grid_sizes.append(torch.tensor(grid, dtype=torch.long))
            content_grid_sizes = torch.stack(content_grid_sizes)
        else:
            content_grid_sizes = None

        context_lens = torch.tensor(
            [math.prod(u.shape[:-1]) for u in context], dtype=torch.long
        )
        assert context_lens.max() <= self.text_len
        context_pad_len = (
            self.text_len
            if self.context_pad_to_text_len
            else max(1, int(context_lens.max().item()))
        )
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [
                            u.flatten(0, -2),
                            u.new_zeros(
                                context_pad_len - math.prod(u.shape[:-1]), u.size(-1)
                            ),
                        ]
                    )
                    for u in context
                ]
            )
        )

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            rope_channel_split=self.rope_channel_split,
            context=context,
            context_lens=context_lens,
            content_grid_sizes=content_grid_sizes,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
