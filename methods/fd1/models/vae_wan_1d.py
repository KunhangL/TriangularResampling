import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.wan_vae_1d import WanVAE_


class VAEWanModel(nn.Module):
    def __init__(
        self,
        input_dim,
        mean_path=None,
        std_path=None,
        z_dim=256,
        dim=160,
        dec_dim=512,
        num_res_blocks=1,
        dropout=0.0,
        dim_mult=[1, 1, 1],
        temperal_downsample=[True, True],
        vel_window=[0, 0],
        **kwargs,
    ):
        super().__init__()

        self.mean_path = mean_path
        self.std_path = std_path
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.dim = dim
        self.dec_dim = dec_dim
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.dim_mult = dim_mult
        self.temperal_downsample = temperal_downsample
        self.vel_window = vel_window
        self.RECONS_LOSS = nn.SmoothL1Loss()
        self.LAMBDA_FEATURE = kwargs.get("LAMBDA_FEATURE", 1.0)
        self.LAMBDA_VELOCITY = kwargs.get("LAMBDA_VELOCITY", 0.5)
        self.LAMBDA_KL = kwargs.get("LAMBDA_KL", 10e-6)

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

        self.model = WanVAE_(
            input_dim=self.input_dim,
            dim=self.dim,
            dec_dim=self.dec_dim,
            z_dim=self.z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            temperal_downsample=self.temperal_downsample,
            dropout=self.dropout,
        )

        downsample_factor = 1
        for flag in self.temperal_downsample:
            if flag:
                downsample_factor *= 2
        self.downsample_factor = downsample_factor

    def preprocess(self, x):
        # (bs, T, C) -> (bs, C, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, C, T) ->  (bs, T, C)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        features = x["feature"]
        feature_length = x["feature_length"]
        features = (features - self.mean) / self.std
        # create mask based on feature_length
        batch_size, seq_len = features.shape[:2]
        mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=features.device
        )
        for i in range(batch_size):
            mask[i, : feature_length[i]] = True

        x_in = self.preprocess(features)  # (bs, input_dim, T)
        mu, log_var = self.model.encode(
            x_in, scale=[0, 1], return_dist=True
        )  # (bs, z_dim, T)
        z = self.model.reparameterize(mu, log_var)
        x_decoder = self.model.decode(z, scale=[0, 1])  # (bs, input_dim, T)
        x_out = self.postprocess(x_decoder)  # (bs, T, input_dim)

        if x_out.size(1) != features.size(1):
            min_len = min(x_out.size(1), features.size(1))
            x_out = x_out[:, :min_len, :]
            features = features[:, :min_len, :]
            mask = mask[:, :min_len]

        mask_expanded = mask.unsqueeze(-1)
        x_out_masked = x_out * mask_expanded
        features_masked = features * mask_expanded
        loss_recons = self.RECONS_LOSS(x_out_masked, features_masked)
        vel_start = self.vel_window[0]
        vel_end = self.vel_window[1]
        loss_vel = self.RECONS_LOSS(
            x_out_masked[..., vel_start:vel_end],
            features_masked[..., vel_start:vel_end],
        )

        # Compute KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # log_var = log(sigma^2), so we can use it directly

        # Build mask for latent space
        T_latent = mu.size(2)
        mask_downsampled = torch.zeros(
            batch_size, T_latent, dtype=torch.bool, device=features.device
        )
        for i in range(batch_size):
            latent_length = (
                feature_length[i] + self.downsample_factor - 1
            ) // self.downsample_factor
            mask_downsampled[i, :latent_length] = True
        mask_latent = mask_downsampled.unsqueeze(1)  # (B, 1, T_latent)

        # Compute KL loss per element
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        # Apply mask: only compute KL loss for valid timesteps
        kl_masked = kl_per_element * mask_latent
        # Sum over all dimensions and normalize by the number of valid elements
        kl_loss = torch.sum(kl_masked) / (
            torch.sum(mask_downsampled) * mu.size(1)
        )  # normalize by valid timesteps * latent_dim

        # Total loss
        total_loss = (
            self.LAMBDA_FEATURE * loss_recons
            + self.LAMBDA_VELOCITY * loss_vel
            + self.LAMBDA_KL * kl_loss
        )

        loss_dict = {}
        loss_dict["total"] = total_loss
        loss_dict["recons"] = loss_recons
        loss_dict["velocity"] = loss_vel
        loss_dict["kl"] = kl_loss

        return loss_dict

    def encode(self, x):
        x = (x - self.mean) / self.std
        x_in = self.preprocess(x)  # (bs, T, input_dim) -> (bs, input_dim, T)
        mu = self.model.encode(x_in, scale=[0, 1])  # (bs, z_dim, T)
        mu = self.postprocess(mu)  # (bs, T, z_dim)
        return mu

    def decode(self, mu):
        mu_in = self.preprocess(mu)  # (bs, T, z_dim) -> (bs, z_dim, T)
        x_decoder = self.model.decode(mu_in, scale=[0, 1])  # (bs, z_dim, T)
        x_out = self.postprocess(x_decoder)  # (bs, T, input_dim)
        x_out = x_out * self.std + self.mean
        return x_out

    @torch.no_grad()
    def stream_encode(self, x, first_chunk=True):
        x = (x - self.mean) / self.std
        x_in = self.preprocess(x)  # (bs, input_dim, T)
        mu = self.model.stream_encode(x_in, first_chunk=first_chunk, scale=[0, 1])
        mu = self.postprocess(mu)  # (bs, T, z_dim)
        return mu

    @torch.no_grad()
    def stream_decode(self, mu, first_chunk=True):
        mu_in = self.preprocess(mu)  # (bs, z_dim, T)
        x_decoder = self.model.stream_decode(
            mu_in, first_chunk=first_chunk, scale=[0, 1]
        )
        x_out = self.postprocess(x_decoder)  # (bs, T, input_dim)
        x_out = x_out * self.std + self.mean
        return x_out

    def clear_cache(self):
        self.model.clear_cache()

    def generate(self, x):
        features = x["feature"]
        feature_length = x["feature_length"]
        y_hat = self.decode(self.encode(features))

        y_hat_out = []

        for i in range(y_hat.shape[0]):
            # cut off the padding and align lengths
            valid_len = (
                feature_length[i] - 1
            ) // self.downsample_factor * self.downsample_factor + 1
            # Make sure both have the same length (take minimum)
            y_hat_out.append(y_hat[i, :valid_len, :])

        out = {}
        out["generated"] = y_hat_out
        return out
