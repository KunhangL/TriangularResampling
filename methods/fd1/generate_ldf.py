import os
import time

import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from common.utils.initialize import compare_statedict_and_parameters, instantiate, load_config
from common.utils.motion_process import StreamJointRecovery263
from methods.fd1.utils.render_skeleton import get_humanml3d_chains, render_simple_skeleton_video
from methods.fd1.utils.visualize import render_single_video

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model_from_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)

    vae = instantiate(
        target=cfg.test_vae.target,
        cfg=None,
        hfstyle=False,
        **cfg.test_vae.params,
    )
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae.parameters(), decay=cfg.test_vae.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
        print(f"Loaded VAE model from {cfg.test_vae_ckpt} with EMA")
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        print(f"Loaded VAE model from {cfg.test_vae_ckpt} w/o EMA")

    compare_statedict_and_parameters(
        state_dict=vae.state_dict(),
        named_parameters=vae.named_parameters(),
        named_buffers=vae.named_buffers(),
    )
    vae.to(device)
    vae.eval()

    # model
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if "ema_state" in checkpoint:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(model.parameters())
        print(f"Loaded model from {cfg.test_ckpt} with EMA")
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"Loaded model from {cfg.test_ckpt} w/o EMA")

    compare_statedict_and_parameters(
        state_dict=model.state_dict(),
        named_parameters=model.named_parameters(),
        named_buffers=model.named_buffers(),
    )
    model.to(device)
    model.eval()

    return vae, model


def generate_feature_stream(
    model, feature_length, text, feature_text_end=None, num_denoise_steps=None
):
    """
    Streaming interface for feature generation
    Args:
        model: Loaded model
        feature_length: List[int], generation length for each sample
        text: List[str] or List[List[str]], text prompts
        feature_text_end: List[List[int]], time points where text ends (if text is list of list)
        num_denoise_steps: Number of denoising steps
    Yields:
        dict: Contains "generated" (current generated feature segment)
    """

    # Construct input dict x using model's input_keys mapping
    ik = model.input_keys
    x = {ik["feature_length"]: torch.tensor(feature_length), ik["text"]: text}

    if feature_text_end is not None:
        x[ik["text_end"]] = feature_text_end

    # Call model's stream_generate
    # Note: stream_generate is a generator
    generator = model.stream_generate(x, num_denoise_steps=num_denoise_steps)

    for output in generator:
        yield output


if __name__ == "__main__":
    # Ensure tmp directory exists
    os.makedirs("tmp", exist_ok=True)

    # Example usage
    text_list = ["walk in a circle.", "jump up."]
    text_end = [150, 250]
    length = text_end[-1]

    vae, model = load_model_from_config()

    print("Starting generation...")
    # Simple example: single sample
    text = [text_list]  # For generate/stream_generate, wrap in list for batch
    feature_text_end = [text_end]
    feature_length = [length]

    ik = model.input_keys
    x = {ik["feature_length"]: torch.tensor(feature_length), ik["text"]: text}
    if feature_text_end is not None:
        x[ik["text_end"]] = feature_text_end

    with torch.no_grad():
        # non-streaming generate
        print("Non-streaming generate...")
        torch.manual_seed(42)
        output = model.generate(x)
        nonstream_generated = output["generated"][0]  # (T, C) latent features
        print(f"Non-streaming generated shape: {nonstream_generated.shape}")
        print("Non-streaming generate done")

        # streaming generate step
        print("Streaming generate step...")
        vae.clear_cache()
        torch.manual_seed(42)
        model.init_generated(100)
        text_end_with_zero = [0] + text_end
        durations = [
            t - b for t, b in zip(text_end_with_zero[1:], text_end_with_zero[:-1])
        ]
        first_chunk = True
        stream_decoded_g = []
        stream_raw_features = []  # accumulate raw latent features for comparison

        # Initialize stream joint recovery for converting motion to joint positions
        # Test both without and with smoothing
        stream_recovery_no_smooth = StreamJointRecovery263(
            joints_num=22, smoothing_alpha=1.0
        )
        stream_recovery_smooth = StreamJointRecovery263(
            joints_num=22, smoothing_alpha=0.5
        )
        stream_joints_no_smooth = []
        stream_joints_smooth = []

        for text_item, duration in zip(text_list, durations):
            for i in range(duration):
                start_time = time.time()
                x = {}
                x[model.input_keys["text"]] = [text_item]  # text_item is a string
                output = model.stream_generate_step(x)
                output = output["generated"]
                # print("output shape: ", output[0].shape)
                if output[0].shape[0] == 0:
                    continue
                stream_raw_features.append(output[0])
                decoded_g = vae.stream_decode(
                    output[0][None, :], first_chunk=first_chunk
                )[0]
                # print("decoded_g shape: ", decoded_g.shape)
                first_chunk = False
                stream_decoded_g.append(decoded_g)

                # Convert each frame to joint positions (both smoothed and non-smoothed)
                # decoded_g can have multiple frames (usually 4)
                decoded_g_np = decoded_g.cpu().numpy()
                if decoded_g_np.ndim == 1:
                    # Single frame
                    frame_joints_no_smooth = stream_recovery_no_smooth.process_frame(
                        decoded_g_np
                    )
                    frame_joints_smooth = stream_recovery_smooth.process_frame(
                        decoded_g_np
                    )
                    stream_joints_no_smooth.append(frame_joints_no_smooth)
                    stream_joints_smooth.append(frame_joints_smooth)
                else:
                    # Multiple frames
                    for frame_idx in range(decoded_g_np.shape[0]):
                        frame_data = decoded_g_np[frame_idx]
                        frame_joints_no_smooth = (
                            stream_recovery_no_smooth.process_frame(frame_data)
                        )
                        frame_joints_smooth = stream_recovery_smooth.process_frame(
                            frame_data
                        )
                        stream_joints_no_smooth.append(frame_joints_no_smooth)
                        stream_joints_smooth.append(frame_joints_smooth)

                print(
                    f"generation time for step {i}: {time.time() - start_time:.4f}s, decoded {decoded_g.shape[0] if decoded_g.ndim > 1 else 1} frames"
                )
                start_time = time.time()
        vae.clear_cache()
        stream_decoded_g = torch.cat(stream_decoded_g, dim=0)
        print("stream_decoded_g shape: ", stream_decoded_g.shape)

        # Convert stream joints list to numpy array
        import numpy as np

        stream_joints_no_smooth = np.array(stream_joints_no_smooth)
        stream_joints_smooth = np.array(stream_joints_smooth)
        print(
            f"stream_joints shape: {stream_joints_no_smooth.shape}"
        )  # Should be (num_frames, 22, 3)

        render_single_video(
            motion=stream_decoded_g.cpu().numpy(),
            save_path="tmp/stream_generated_step.mp4",
            dim=263,
            render_setting={},
        )

        # Render joint positions (no smoothing)
        print("Rendering joints (no smoothing)...")
        chains = get_humanml3d_chains()
        render_simple_skeleton_video(
            data=stream_joints_no_smooth,
            chains=chains,
            out_path="tmp/stream_joints_no_smooth.mp4",
            fps=20,
        )

        # Render joint positions (with smoothing)
        print("Rendering joints (with smoothing)...")
        render_simple_skeleton_video(
            data=stream_joints_smooth,
            chains=chains,
            out_path="tmp/stream_joints_smooth.mp4",
            fps=20,
        )

        print("Streaming generate step done")

        # Save smoothed joints for future use
        np.save("tmp/stream_joints_no_smooth.npy", stream_joints_no_smooth)
        np.save("tmp/stream_joints_smooth.npy", stream_joints_smooth)
        print("Saved joint positions to tmp/stream_joints_*.npy")

        # Compare non-streaming vs streaming latent features
        print("\n=== Comparing non-streaming vs streaming latent features ===")
        stream_features = torch.cat(stream_raw_features, dim=0)  # (T_stream, C)
        print(f"Non-streaming shape: {nonstream_generated.shape}")
        print(f"Streaming shape:     {stream_features.shape}")
        min_len = min(nonstream_generated.shape[0], stream_features.shape[0])
        ns = nonstream_generated[:min_len].cpu()
        st = stream_features[:min_len].cpu()
        per_frame_mse = ((ns - st) ** 2).mean(dim=-1)  # (T,)
        per_frame_max = (ns - st).abs().max(dim=-1).values  # (T,)
        # Find first divergent frame
        threshold = 1e-4
        divergent_mask = per_frame_mse > threshold
        if divergent_mask.any():
            first_divergent = divergent_mask.nonzero(as_tuple=True)[0][0].item()
            print(f"First divergent frame (MSE > {threshold}): {first_divergent}")
        else:
            print(f"All {min_len} frames match within MSE threshold {threshold}")
        # Print per-frame stats every 10 frames
        print(f"\nPer-frame MSE (every 10 frames):")
        for i in range(0, min_len, 10):
            print(
                f"  frame {i:4d}: MSE={per_frame_mse[i]:.8f}, MaxAbsDiff={per_frame_max[i]:.8f}"
            )
        print(f"\nOverall MSE: {per_frame_mse.mean():.8f}")
        print(f"Overall MaxAbsDiff: {per_frame_max.max():.8f}")
