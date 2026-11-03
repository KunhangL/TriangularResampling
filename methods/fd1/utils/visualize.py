import os
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from common.utils.motion_process import convert_motion_to_joints, convert_motion_to_smpl_params, recover_smpl_params_201d
from methods.fd1.utils.render_mesh import SMPLMeshRenderer, render_mesh_video
from methods.fd1.utils.render_skeleton import (
    get_humanml3d_chains,
    recover_joint_positions_201d,
    recover_joint_positions_201d_fk,
    render_simple_skeleton_video,
    render_skeleton_video,
)


def mux_audio_to_video(video_path, audio_path):
    """Mux a .wav audio track into an existing .mp4 video using ffmpeg."""
    import shutil
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", tmp_path],
            check=False, capture_output=True, text=True,
        )
        if result.returncode == 0:
            shutil.move(tmp_path, video_path)
        else:
            print(f"Warning: Failed to mux audio to {video_path}: {result.stderr}")
            os.unlink(tmp_path) if os.path.exists(tmp_path) else None
    except Exception as e:
        print(f"Warning: Failed to mux audio to {video_path}: {e}")
        os.unlink(tmp_path) if os.path.exists(tmp_path) else None


def render_single_video(
    motion: np.ndarray,
    save_path: str,
    dim,
    render_setting: dict = {},
    mean_np: np.ndarray = None,
    std_np: np.ndarray = None,
    frames: np.ndarray = None,
    audio_path: str = None,
    feature_layout: dict = None,
    smpl_renderer=None,
):
    fps = render_setting.get("fps", 20)
    render_mode = render_setting.get("render_mode", "skeleton")

    if render_mode == "mesh":
        # Mesh rendering path
        if dim in (138, "138"):
            # MEI138: route through the shared `render_mei138_mesh_video` entry
            # in utils.render_mesh.  One FK pass (`recover_mei138_mesh` →
            # `mei_to_smplh` direct read of root_orient/body_pose/transl) yields
            # mesh + 22 joints together, no IK needed and no twist artefacts.
            from methods.fd1.utils.render_mesh import render_mei138_mesh_video
            motion_138 = motion
            if mean_np is not None and std_np is not None:
                motion_138 = motion_138 * std_np + mean_np
            render_mei138_mesh_video(
                motion_138,
                save_path,
                fps=fps,
                renderer=smpl_renderer,
                frames=frames,
                smpl_model_path=render_setting.get("smpl_model_path"),
                overlay_keypoints=render_setting.get("overlay_keypoints", True),
                image_size=render_setting.get("image_size", 800),
            )
            if audio_path is not None and os.path.exists(audio_path):
                mux_audio_to_video(save_path, audio_path)
            return

        if dim == "333d":
            # 333d: rot6d + trans -> SMPL params directly (no IK needed)
            from methods.fd1.metrics.s2g import convert_333d_to_smpl_params, convert_333d_to_position_165
            smpl_params = convert_333d_to_smpl_params(motion)
            pos_165 = convert_333d_to_position_165(motion)
            joint_positions = pos_165.reshape(-1, 55, 3)[:, :22, :]
        elif dim in ("201d_pos", "201d_fk"):
            # 201d: extract rot6d -> axis-angle directly (no IK needed)
            smpl_params = recover_smpl_params_201d(motion)
            joint_positions = recover_joint_positions_201d(motion)
        elif dim == "201d_world":
            from common.utils.motion_process import recover_smpl_params_201d_world, recover_joint_positions_201d_world
            r6conv = render_setting.get("rot6d_convention", "row")
            smpl_params = recover_smpl_params_201d_world(motion, rot6d_convention=r6conv)
            joint_positions = recover_joint_positions_201d_world(motion)
        else:
            smpl_params = convert_motion_to_smpl_params(
                motion, dim, mean_np, std_np, feature_layout=feature_layout, fps=fps
            )
            joint_positions = convert_motion_to_joints(
                motion, dim, mean_np, std_np, feature_layout=feature_layout, fps=fps
            )

        if joint_positions.shape[1] > 22:
            joint_positions = joint_positions[:, :22, :]

        if smpl_renderer is None:
            smpl_renderer = SMPLMeshRenderer(
                smpl_model_path=render_setting.get("smpl_model_path")
            )

        vertices, faces = smpl_renderer.get_mesh_sequence(
            smpl_params["root_orient"],
            smpl_params["pose_body"],
            smpl_params["trans"],
            jaw_pose=smpl_params.get("jaw_pose"),
            left_hand_pose=smpl_params.get("left_hand_pose"),
            right_hand_pose=smpl_params.get("right_hand_pose"),
        )

        render_mesh_video(
            vertices=vertices,
            faces=faces,
            out_path=save_path,
            fps=fps,
            frames=frames,
            joint_positions=joint_positions,
            overlay_keypoints=render_setting.get("overlay_keypoints", True),
        )
    else:
        # Skeleton rendering (original behavior)
        chains = get_humanml3d_chains()
        if dim == "333d":
            from methods.fd1.metrics.s2g import convert_333d_to_position_165
            pos_165 = convert_333d_to_position_165(motion)
            joint_positions = pos_165.reshape(-1, 55, 3)[:, :22, :]
        elif dim == "201d_pos":
            joint_positions = recover_joint_positions_201d(motion)
        elif dim == "201d_fk":
            joint_positions = recover_joint_positions_201d_fk(motion)
        elif dim == "201d_world":
            from common.utils.motion_process import recover_joint_positions_201d_world
            joint_positions = recover_joint_positions_201d_world(motion)
        else:
            joint_positions = convert_motion_to_joints(
                motion, dim, mean_np, std_np, feature_layout=feature_layout, fps=fps
            )
        if joint_positions.shape[1] > 22:
            joint_positions = joint_positions[:, :22, :]
        if render_setting.get("simple", True):
            render_simple_skeleton_video(
                data=joint_positions,
                chains=chains,
                out_path=save_path,
                fps=fps,
                frames=frames,
            )
        else:
            render_skeleton_video(
                data=joint_positions,
                chains=chains,
                out_path=save_path,
                fps=fps,
                frames=frames,
            )

    if audio_path is not None and os.path.exists(audio_path):
        mux_audio_to_video(save_path, audio_path)


def render_video(
    motion_dir: str,
    save_dir: str,
    render_setting,
    frames_dir: str = None,
    audio_dir: str = None,
    max_count: int = 0,
):
    os.makedirs(save_dir, exist_ok=True)
    motion_path = Path(motion_dir)
    npy_files = sorted(motion_path.glob("*.npy"))
    if max_count > 0:
        npy_files = npy_files[:max_count]

    motion_count = 0
    error_count = 0

    # Get feature_layout for UMR format (None for HumanML3D)
    feature_layout = render_setting.get("feature_layout", None)

    if render_setting.get("mean_path") is not None:
        mean_np = np.load(render_setting["mean_path"])
        std_np = np.load(render_setting["std_path"])
    else:
        mean_np = np.zeros(render_setting["recover_dim"])
        std_np = np.ones(render_setting["recover_dim"])

    # Cache SMPL renderer for mesh mode to avoid reloading model per clip
    smpl_renderer = None
    if render_setting.get("render_mode") == "mesh":
        smpl_renderer = SMPLMeshRenderer(
            smpl_model_path=render_setting.get("smpl_model_path")
        )

    for npy_file in tqdm(npy_files, desc="Rendering"):
        motion_data = np.load(npy_file)
        output_filename = npy_file.stem + ".mp4"
        output_path = os.path.join(save_dir, output_filename)

        if frames_dir is not None and os.path.exists(frames_dir):
            frames_path = os.path.join(frames_dir, npy_file.name)
            frames = np.load(frames_path)
        else:
            frames = None

        audio_path = None
        if audio_dir:
            ap = os.path.join(audio_dir, npy_file.stem + ".wav")
            if os.path.exists(ap):
                audio_path = ap

        try:
            render_single_video(
                motion=motion_data,
                save_path=output_path,
                dim=render_setting["recover_dim"],
                render_setting=render_setting,
                mean_np=mean_np,
                std_np=std_np,
                frames=frames,
                audio_path=audio_path,
                feature_layout=feature_layout,
                smpl_renderer=smpl_renderer,
            )
        except Exception as e:
            print(f"Error rendering {npy_file}: {e}")
            error_count += 1
            continue
        motion_count += 1

    print(
        f"{motion_count} motion clips rendered. {error_count} errors. Saved to {save_dir}"
    )


def render_text_bar(
    text, width, padding=20, font_size=28, bg_color=(0, 0, 0), fg_color=(255, 255, 255)
):
    """Renders a text bar with automatic wrapping using PIL, returns np.uint8(H,W,3)."""
    try:
        font_path = font_manager.findfont("DejaVu Sans")
    except Exception:
        font_path = font_manager.findfont("Arial")
    font = ImageFont.truetype(font_path, font_size)

    # Split text by separator if present
    segments = []
    if "//////////" in text:
        parts = text.split("//////////")
        for part in parts:
            if part.strip():
                segments.append(part.strip())
    else:
        segments.append(text)

    # Define tint colors matching render_skeleton.py
    # Using slightly more saturated colors for text to ensure readability against black background
    tint_colors = [
        (255, 180, 180),  # Reddish
        (180, 255, 180),  # Greenish
        (180, 180, 255),  # Blueish
    ]

    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    max_w = width - 2 * padding

    all_lines = []  # List of (text_content, color) tuples

    for i, segment in enumerate(segments):
        color = fg_color
        if len(segments) > 1:
            color = tint_colors[i % len(tint_colors)]

        # Wrap text for this segment
        cur = ""
        first_word = True
        for w in segment.split():
            test = (cur + " " + w).strip() if cur else w
            if dummy.textlength(test, font=font) <= max_w:
                cur = test
            else:
                all_lines.append((cur, color))
                cur = w
        if cur:
            all_lines.append((cur, color))

    _, top, _, bottom = font.getbbox("A")
    line_h = bottom - top + 4
    bar_h = padding * 2 + line_h * len(all_lines)

    # Ensure height is even for H.264 encoding
    if bar_h % 2 != 0:
        bar_h += 1

    img = Image.new("RGB", (width, bar_h), bg_color)
    draw = ImageDraw.Draw(img)
    y = padding
    for line_text, line_color in all_lines:
        draw.text((padding, y), line_text, font=font, fill=line_color)
        y += line_h
    return np.array(img)


def render_aligned_title_bar(
    total_width,
    widths,
    titles,
    font_size=32,
    bg_color=(255, 255, 255),
    fg_color=(0, 0, 0),
    padding=10,
):
    """Renders a title bar with centered titles aligned to video sections."""
    try:
        font_path = font_manager.findfont("DejaVu Sans")
    except Exception:
        font_path = font_manager.findfont("Arial")
    font = ImageFont.truetype(font_path, font_size)

    # Calculate title bar height
    _, top, _, bottom = font.getbbox("A")
    bar_height = bottom - top + 2 * padding

    # Ensure height is even for H.264 encoding
    if bar_height % 2 != 0:
        bar_height += 1

    # Create image
    img = Image.new("RGB", (total_width, bar_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Calculate positions and draw titles
    x_offset = 0
    for i, (title, width) in enumerate(zip(titles, widths)):
        # Calculate center position for this section
        text_width = draw.textlength(title, font=font)
        x_center = x_offset + width // 2
        x_pos = x_center - text_width // 2
        y_pos = padding

        draw.text((x_pos, y_pos), title, font=font, fill=fg_color)
        x_offset += width

    return np.array(img)


def _ffprobe_bin():
    """Resolve the ffprobe binary (FFPROBE_BIN env var, PATH, then /usr/bin)."""
    import shutil
    p = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
    if p:
        return p
    for candidate in (
        "/usr/bin/ffprobe",
    ):
        if os.path.exists(candidate):
            return candidate
    return "ffprobe"  # last-ditch; will raise the original error


def _get_video_info(video_path: str):
    """Get video width, height, and duration using ffprobe."""
    import subprocess

    cmd = [
        _ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "csv=p=0",
        video_path,
    ]
    output = subprocess.check_output(
        cmd, text=True, stderr=subprocess.DEVNULL
    ).strip().split(",")
    return int(output[0]), int(output[1]), float(output[2])


def _get_fps(video_path: str):
    """Get video frame rate using ffprobe."""
    import subprocess

    cmd = [
        _ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        video_path,
    ]
    fps_str = subprocess.check_output(
        cmd, text=True, stderr=subprocess.DEVNULL
    ).strip()
    num, den = map(int, fps_str.split("/"))
    return num / den


def _build_video_filter(
    input_idx: int,
    video_idx: int,
    target_height: int,
    duration: float,
    max_duration: float,
    target_width: int = -2,
):
    """Build ffmpeg filter for a single video stream."""
    filters = []

    # Scale to target height, ensure even dimensions
    # filters.append(f"[{input_idx}:v]scale=-2:{target_height}[v{video_idx}_scaled]")
    filters.append(
        f"[{input_idx}:v]scale={target_width}:{target_height}[v{video_idx}_scaled]"
    )

    # Handle duration and graying
    if duration < max_duration:
        pad_duration = max_duration - duration
        filters.append(
            f"[v{video_idx}_scaled]tpad=stop_mode=clone:stop_duration={pad_duration}[v{video_idx}_padded]"
        )
        # Apply gray effect after original duration using eq filter with enable
        filters.append(
            f"[v{video_idx}_padded]eq=brightness=-0.5:saturation=0:enable='gte(t,{duration})'[v{video_idx}]"
        )
    else:
        # Video is already long enough, just use scaled version
        filters.append(f"[v{video_idx}_scaled]null[v{video_idx}]")

    return filters


def make_composite_compare_videos(
    result_folder: str,
    save_dir: str,
    text_folder: str = None,
    compare_folders: list = None,
    compare_names: list = None,
    result_name: str = "Ours",
    result_position: str = "left",
):
    """Generates composite videos of (result | compare_folders) with captions and text descriptions.

    Args:
        result_folder: Folder containing result videos (base for comparison)
        save_dir: Directory to save composite videos
        text_folder: Folder containing text descriptions (optional)
        compare_folders: List of folders to compare with result (optional)
        compare_names: List of names for compare folders (optional)

    Uses the longest video duration. Missing videos show black screen.
    Videos that end early show their last frame grayed out.
    Optimized version using ffmpeg directly for much faster processing.
    """
    import subprocess
    import tempfile

    os.makedirs(save_dir, exist_ok=True)
    video_files = list(Path(result_folder).glob("*.mp4"))

    # Handle empty or non-existent compare folders
    if compare_folders is None:
        compare_folders = []
    if compare_names is None:
        compare_names = []

    # Filter out non-existent compare folders
    valid_compare_folders = []
    valid_compare_names = []
    for i, folder in enumerate(compare_folders):
        if folder and os.path.exists(folder):
            valid_compare_folders.append(folder)
            if i < len(compare_names):
                valid_compare_names.append(compare_names[i])
            else:
                valid_compare_names.append(f"Compare {i + 1}")

    compare_folders = valid_compare_folders
    compare_names = valid_compare_names

    for video_file in tqdm(video_files, desc="Creating composite videos"):
        video_id = video_file.stem

        # Build hstack column order. Iteration is always driven by result_folder,
        # but the result column itself goes to the LEFT (default) or RIGHT
        # depending on result_position.
        compare_paths = [
            os.path.join(folder, f"{video_id}.mp4")
            for folder in compare_folders
        ]
        if result_position == "right":
            video_paths = compare_paths + [str(video_file)]
            video_names = list(compare_names) + [result_name]
        else:
            video_paths = [str(video_file)] + compare_paths
            video_names = [result_name] + list(compare_names)

        video_exists = [os.path.exists(p) for p in video_paths]

        # Load text description
        if text_folder:
            text_file = os.path.join(text_folder, f"{video_id}.txt")
            text_content = (
                Path(text_file).read_text().strip()
                if os.path.exists(text_file)
                else f"Motion: {video_id}"
            )
        else:
            text_content = f"Motion: {video_id}"

        # Find reference video for properties (should always have result video)
        reference_video = str(video_file)
        if not os.path.exists(reference_video):
            print(f"Error: Result video not found for {video_id}, skipping")
            continue

        # Get video properties
        try:
            fps = _get_fps(reference_video)
            ref_width, ref_height, _ = _get_video_info(reference_video)
        except Exception as e:
            print(f"Error probing {video_id}: {e}, skipping")
            continue

        # Collect dimensions and durations for all videos
        widths, heights, durations = [], [], []
        for path, exists, name in zip(video_paths, video_exists, video_names):
            if exists:
                try:
                    w, h, d = _get_video_info(path)
                    widths.append(w)
                    heights.append(h)
                    durations.append(d)
                except Exception as e:
                    print(
                        f"Error probing {name} video for {video_id}: {e}, will use black screen"
                    )
                    widths.append(ref_width)
                    heights.append(ref_height)
                    durations.append(0)
            else:
                print(
                    f"Warning: {name} video missing for {video_id}, will use black screen"
                )
                widths.append(ref_width)
                heights.append(ref_height)
                durations.append(0)

        max_duration = max(durations)
        if max_duration == 0:
            print(f"Warning: All videos for {video_id} have zero duration, skipping")
            continue

        target_height = min(h for h in heights if h > 0)
        # Ensure target height is even
        if target_height % 2 != 0:
            target_height += 1

        # Re-calculate widths based on target_height scaling
        new_widths = []
        for w, h in zip(widths, heights):
            if h > 0:
                # Calculate scaled width maintaining aspect ratio
                aspect_ratio = w / h
                scaled_w = int(target_height * aspect_ratio)
                # Ensure even width
                if scaled_w % 2 != 0:
                    scaled_w += 1
                new_widths.append(scaled_w)
            else:
                new_widths.append(w)

        widths = new_widths
        total_width = sum(widths)

        # print(f"DEBUG: video_id={video_id}")
        # print(f"DEBUG: original widths={widths} (after update), heights={heights}")
        # print(f"DEBUG: target_height={target_height}")
        # print(f"DEBUG: total_width={total_width}")

        # Create and save title/text bars
        title_bar = render_aligned_title_bar(
            total_width, widths, video_names, font_size=32
        )
        text_bar = render_text_bar(text_content, width=total_width, font_size=24)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            title_path = f.name
            Image.fromarray(title_bar).save(title_path)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            text_path = f.name
            Image.fromarray(text_bar).save(text_path)

        # Build ffmpeg command
        inputs = []
        filter_parts = []
        input_count = 0

        # Add video inputs and filters
        for i, (path, exists, width, duration) in enumerate(
            zip(video_paths, video_exists, widths, durations)
        ):
            if exists:
                inputs.extend(["-i", path])
                filter_parts.extend(
                    _build_video_filter(
                        input_count,
                        i,
                        target_height,
                        duration,
                        max_duration,
                        target_width=width,
                    )
                )
            else:
                inputs.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c=black:s={width}x{target_height}:d={max_duration}:r={int(fps)}",
                    ]
                )
                filter_parts.append(f"[{input_count}:v]null[v{i}]")
            input_count += 1

        # Add title and text images
        inputs.extend(["-loop", "1", "-i", title_path])
        title_idx = input_count
        input_count += 1

        inputs.extend(["-loop", "1", "-i", text_path])
        text_idx = input_count

        # Compose final layout - horizontally stack all videos
        num_videos = len(video_paths)
        if num_videos == 1:
            filter_parts.append("[v0]null[videos]")
        else:
            video_inputs = "".join([f"[v{i}]" for i in range(num_videos)])
            filter_parts.append(f"{video_inputs}hstack=inputs={num_videos}[videos]")

        filter_parts.append(
            f"[{title_idx}:v][videos][{text_idx}:v]vstack=inputs=3[out]"
        )

        # Execute ffmpeg
        output_path = os.path.join(save_dir, f"{video_id}_composite.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[out]",
            "-t",
            str(max_duration),
            "-r",
            str(int(fps)),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-movflags",
            "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error processing {video_id}: Return code {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Stderr: {result.stderr}")
            elif os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                print(f"Warning: Generated video {output_path} is empty!")
                print(f"Command: {' '.join(cmd)}")
                print(f"Stderr: {result.stderr}")
        except Exception as e:
            print(f"Unexpected error processing {video_id}: {e}")
        finally:
            for path in [title_path, text_path]:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    print(f"Composite videos saved to {save_dir}")
