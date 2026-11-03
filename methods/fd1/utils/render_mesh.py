#!/usr/bin/env python3
"""SMPL mesh rendering for motion visualization.

Renders SMPL body mesh videos from motion features or raw UMR npz files.
Supports multiple rotation representations and analytical IK fallback.

Rendering backend: pytorch3d (CUDA)
"""

import contextlib
import os

import numpy as np
import smplx as smplx_lib
import torch

from common.utils.math.mei import (
    MEI_DEFAULT_FPS,
    mei_to_smplh,
)
from methods.fd1.utils.render_skeleton import (
    compute_camera_params,
    draw_skeleton_on_image,
    get_humanml3d_chains,
)

# SMPL-X body models (https://smpl-x.is.tue.mpg.de/): <SMPLX_MODEL_DIR>/smplx/SMPLX_NEUTRAL.npz etc.
# Defaults to deps/smplx_models relative to the working directory; override with the
# SMPLX_MODEL_DIR environment variable or pass smpl_model_path explicitly.
_DEFAULT_SMPLX_DIR = os.environ.get("SMPLX_MODEL_DIR", os.path.join("deps", "smplx_models"))


def _normalize_smplx_gender(gender):
    gender = str(gender).lower()
    if gender.startswith("b'") and gender.endswith("'"):
        gender = gender[2:-1]
    if gender.startswith('b"') and gender.endswith('"'):
        gender = gender[2:-1]
    return gender if gender in ("male", "female", "neutral") else "neutral"


class SMPLMeshRenderer:
    """Loads an SMPLX body model and runs forward pass for mesh rendering."""

    def __init__(self, smpl_model_path=None, gender="neutral", device=None):
        if smpl_model_path is None:
            smpl_model_path = _DEFAULT_SMPLX_DIR
        requested_gender = _normalize_smplx_gender(gender)
        model_file = os.path.join(
            smpl_model_path, "smplx", f"SMPLX_{requested_gender.upper()}.npz"
        )
        model_file_2020 = os.path.join(
            smpl_model_path, "smplx", f"SMPLX_{requested_gender.upper()}_2020.npz"
        )
        self.gender = requested_gender
        if requested_gender != "neutral" and not (
            os.path.exists(model_file) or os.path.exists(model_file_2020)
        ):
            self.gender = "neutral"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.body_model = smplx_lib.create(
            smpl_model_path, model_type="smplx", gender=self.gender,
            num_betas=10, use_pca=False, flat_hand_mean=True,
        ).to(self.device).eval()
        self.faces = self.body_model.faces.astype(np.int64)

    def get_mesh_sequence(self, root_orient, pose_body, trans, betas=None,
                          jaw_pose=None, left_hand_pose=None, right_hand_pose=None,
                          return_joints=False):
        T = root_orient.shape[0]
        device = self.device
        _t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)

        if betas is None:
            betas_t = torch.zeros(T, 10, device=device)
        else:
            betas_np = np.asarray(betas, dtype=np.float32)
            if betas_np.ndim == 1:
                betas_np = np.tile(betas_np[:10], (T, 1))
            else:
                betas_np = betas_np[:, :10]
            betas_t = _t(betas_np)

        kwargs = dict(
            global_orient=_t(root_orient),
            body_pose=_t(pose_body),
            transl=_t(trans),
            betas=betas_t,
            jaw_pose=_t(jaw_pose) if jaw_pose is not None else torch.zeros(T, 3, device=device),
            leye_pose=torch.zeros(T, 3, device=device),
            reye_pose=torch.zeros(T, 3, device=device),
            left_hand_pose=_t(left_hand_pose) if left_hand_pose is not None else torch.zeros(T, 45, device=device),
            right_hand_pose=_t(right_hand_pose) if right_hand_pose is not None else torch.zeros(T, 45, device=device),
            expression=torch.zeros(T, 10, device=device),
        )

        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False)
            if hasattr(torch, "autocast") and device.type != "cpu"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            output = self.body_model(**kwargs)
        vertices = output.vertices.float().cpu().numpy()
        if return_joints:
            joints = output.joints[:, :22, :].float().cpu().numpy()
            return vertices, self.faces, joints
        return vertices, self.faces

    def get_joints_22_batched(self, root_orient, pose_body, trans, betas=None,
                              left_hand_pose=None, right_hand_pose=None,
                              batch_size=512):
        """Memory-bounded SMPL FK -> joints[:, :22, :] as numpy.

        `get_mesh_sequence` runs once-pass and would OOM on T >> 1k. This
        helper chunks the input over T and runs SMPLX forward in batches,
        used by extract_138d for offline 138d feature extraction on long
        AMASS motions.

        Args:
            root_orient: (T, 3) axis-angle global orient
            pose_body:   (T, 63) axis-angle body 21 joints
            trans:       (T, 3) translation
            betas:       (10,) [tile per frame] or (T, 10) [per frame]
            left_hand_pose / right_hand_pose: (T, 45) optional (default zeros)
            batch_size:  per-forward T slice (default 512)

        Returns:
            np.ndarray (T, 22, 3) float32 joint positions in SMPLX world frame
        """
        T = int(np.asarray(root_orient).shape[0])
        device = self.device
        betas_arr = None if betas is None else np.asarray(betas, dtype=np.float32)
        is_per_frame = betas_arr is not None and betas_arr.ndim == 2
        out = []
        for s in range(0, T, int(batch_size)):
            e = min(s + int(batch_size), T)
            bs = e - s
            _t = lambda x: torch.tensor(x, dtype=torch.float32, device=device) if not isinstance(x, torch.Tensor) else x.to(device).float()
            if betas_arr is None:
                betas_t = torch.zeros(bs, 10, device=device)
            elif is_per_frame:
                betas_t = _t(betas_arr[s:e, :10])
            else:
                betas_t = _t(np.tile(betas_arr[:10], (bs, 1)))
            kwargs = dict(
                global_orient=_t(np.asarray(root_orient)[s:e]),
                body_pose=_t(np.asarray(pose_body)[s:e]),
                transl=_t(np.asarray(trans)[s:e]),
                betas=betas_t,
                jaw_pose=torch.zeros(bs, 3, device=device),
                leye_pose=torch.zeros(bs, 3, device=device),
                reye_pose=torch.zeros(bs, 3, device=device),
                left_hand_pose=_t(np.asarray(left_hand_pose)[s:e]) if left_hand_pose is not None else torch.zeros(bs, 45, device=device),
                right_hand_pose=_t(np.asarray(right_hand_pose)[s:e]) if right_hand_pose is not None else torch.zeros(bs, 45, device=device),
                expression=torch.zeros(bs, 10, device=device),
            )
            autocast_ctx = (
                torch.autocast(device_type=device.type, enabled=False)
                if hasattr(torch, "autocast") and device.type != "cpu"
                else contextlib.nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                output = self.body_model(**kwargs)
            out.append(output.joints[:, :22, :].float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def get_tpose_pelvis(self, betas=None):
        device = self.device
        if betas is None:
            betas_t = torch.zeros(1, 10, device=device)
        else:
            betas_np = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
            betas_t = torch.tensor(betas_np[None], dtype=torch.float32, device=device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False)
            if hasattr(torch, "autocast") and device.type != "cpu"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            output = self.body_model(
                global_orient=torch.zeros(1, 3, device=device),
                body_pose=torch.zeros(1, 63, device=device),
                transl=torch.zeros(1, 3, device=device),
                betas=betas_t,
                jaw_pose=torch.zeros(1, 3, device=device),
                leye_pose=torch.zeros(1, 3, device=device),
                reye_pose=torch.zeros(1, 3, device=device),
                left_hand_pose=torch.zeros(1, 45, device=device),
                right_hand_pose=torch.zeros(1, 45, device=device),
                expression=torch.zeros(1, 10, device=device),
            )
        return output.joints[0, 0].float().cpu().numpy()


def recover_mei138_mesh(motion, fps=MEI_DEFAULT_FPS, smpl_model_path=None,
                        renderer=None):
    """Recover neutral SMPLX mesh vertices and 22-joint overlay from MEI138."""
    params = mei_to_smplh(np.asarray(motion, dtype=np.float64), fps=fps)
    if renderer is None:
        renderer = SMPLMeshRenderer(smpl_model_path=smpl_model_path, gender="neutral")
    # The SMPLX Python model used here is already in the same Y-up frame as the
    # MEI138 features.  Do not apply the PHC/z-up conversion helpers here; doing
    # so swaps the vertical axis and renders the body lying on its back.
    root_orient = params["root_orient"]
    body_pose = params["body_pose"]
    trans_model = params["transl"] - renderer.get_tpose_pelvis()[None, :]
    vertices_raw, faces, joints_raw = renderer.get_mesh_sequence(
        root_orient,
        body_pose,
        trans_model,
        return_joints=True,
    )
    return vertices_raw, faces, joints_raw


def render_mei138_mesh_frames(
    motion,
    fps=MEI_DEFAULT_FPS,
    smpl_model_path=None,
    renderer=None,
    frames=None,
    mesh_color=(0.85, 0.68, 0.58, 1.0),
    align_root_to=None,
    overlay_skeleton=None,
    overlay_keypoints=True,
    image_size=800,
    normalize_ground=None,
):
    """Render a MEI138 feature array as mesh+skeleton frames (no mp4 write).

    Same mesh-recovery + overlay-handling pipeline as
    `render_mei138_mesh_video`, but returns a `list[np.ndarray (H, W, 3)]`
    so callers (e.g. debug grid scripts) can compose multiple panels via
    `np.concatenate` instead of writing one mp4 per panel.

    `align_root_to` and `overlay_skeleton` are independent: pass either,
    both, or neither. See `render_mei138_mesh_video` for parameter docs.
    """
    vertices, faces, joints = recover_mei138_mesh(
        motion, fps=fps, smpl_model_path=smpl_model_path, renderer=renderer,
    )

    # Truncate to a length covered by every external input.
    n = len(vertices)
    if align_root_to is not None:
        n = min(n, len(align_root_to))
    if overlay_skeleton is not None:
        n = min(n, len(overlay_skeleton))
    vertices = vertices[:n]
    joints = joints[:n]

    if align_root_to is not None:
        target = np.asarray(align_root_to, dtype=np.float32)[:n]
        # Accept (T, 3) for an explicit root track, or (T, J, 3) (J >= 1)
        # for a full-skeleton source (root = joint 0). Only joint 0 is used.
        if target.ndim == 3:
            target_root = target[:, 0, :]
        elif target.ndim == 2 and target.shape[-1] == 3:
            target_root = target
        else:
            raise ValueError(
                f"align_root_to must be (T, 3) or (T, J>=1, 3); got {target.shape}"
            )
        delta = (target_root - joints[:, 0, :])[:, None, :]              # (T, 1, 3)
        vertices = vertices + delta
        joints = joints + delta

    if overlay_skeleton is not None:
        joints_to_render = np.asarray(overlay_skeleton, dtype=np.float32)[:n, :22]
    else:
        joints_to_render = joints

    if normalize_ground is None:
        # Auto-ground unless the caller is pinning the mesh to an external
        # root trajectory — in that case keep the overlay's real Y.
        normalize_ground = align_root_to is None
    return _render_mesh_frames_pytorch3d(
        vertices,
        faces,
        frames,
        mesh_color,
        joint_positions=joints_to_render,
        overlay_keypoints=overlay_keypoints,
        image_size=image_size,
        normalize_ground=normalize_ground,
    )


def render_mei138_mesh_video(
    motion,
    out_path,
    fps=MEI_DEFAULT_FPS,
    smpl_model_path=None,
    renderer=None,
    frames=None,
    mesh_color=(0.85, 0.68, 0.58, 1.0),
    align_root_to=None,
    overlay_skeleton=None,
    overlay_keypoints=True,
    image_size=800,
    normalize_ground=None,
):
    """Render a MEI138 feature array as a neutral SMPLX mesh + skeleton video.

    Single-pass FK: `recover_mei138_mesh` returns mesh vertices, faces, and
    22 joint positions in one go.  Both are passed to `render_mesh_video`
    (mesh body + keypoint overlay drawn on top) so panels show the body
    surface and the joint chain together.

    `align_root_to` and `overlay_skeleton` are independent and can be combined:

      - Neither: mesh + FK skeleton, mesh sits at MEI138's origin-integrated
        trajectory; ground is auto-normalized.
      - `align_root_to=R` only: mesh per-frame shifted so root = R; skeleton
        is the mesh's own FK joints (after the shift) — mesh and skeleton
        stay strictly co-sourced. Used by debug-viz row-2 panels.
      - `overlay_skeleton=S` only: mesh sits at MEI138 origin trajectory;
        skeleton drawn on top is the external S (e.g. for comparing an
        external joint source against the mesh body).
      - Both: shift mesh to align_root_to and draw overlay_skeleton on top
        (legacy "anchor mesh to external world + show external bones" mode).

    Args:
        motion: (T, 138) MEI138 feature array.
        out_path: output mp4 path.
        fps: source motion fps; used both for MEI138 integration and as the
            output video framerate.
        smpl_model_path: optional path passed to a freshly-built renderer.
        renderer: optional pre-built `SMPLMeshRenderer` (reuse across many
            panels to avoid rebuilding the heavy pytorch3d state).
        mesh_color: RGBA tuple for the body surface.
        align_root_to: optional root trajectory used to pin the mesh's root
            per-frame. Accepts `(T, 3)` (explicit root) or `(T, J>=1, 3)`
            (joint 0 is taken as root). The per-frame shift is
            `target_root[t] - fk_root[t]`; mesh body shape is untouched
            (translation-only, no rotation). The shifted FK joints are also
            returned as the skeleton unless `overlay_skeleton` is given.
        overlay_skeleton: optional `(T, 22, 3)` joint sequence drawn as the
            skeleton on top of the mesh. When None, the mesh's own FK joints
            are used (so mesh and skeleton are co-sourced).
        overlay_keypoints: forward to `render_mesh_video`.
        image_size: forward to `render_mesh_video`.
        normalize_ground: when None, auto-grounds unless `align_root_to` is
            given (overlay-anchored panels keep the overlay's real Y).
    """
    import imageio
    rendered = render_mei138_mesh_frames(
        motion,
        fps=fps,
        smpl_model_path=smpl_model_path,
        renderer=renderer,
        frames=frames,
        mesh_color=mesh_color,
        align_root_to=align_root_to,
        overlay_skeleton=overlay_skeleton,
        overlay_keypoints=overlay_keypoints,
        image_size=image_size,
        normalize_ground=normalize_ground,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=int(round(fps)))
    try:
        for img in rendered:
            writer.append_data(img)
    finally:
        writer.close()


def render_mesh_video(
    vertices,
    faces,
    out_path,
    fps=20,
    frames=None,
    mesh_color=(0.85, 0.68, 0.58, 1.0),
    joint_positions=None,
    overlay_keypoints=True,
    image_size=800,
    normalize_ground=True,
):
    """Render SMPL mesh video using pytorch3d.

    Args:
        vertices: (T, V, 3) mesh vertices per frame
        faces: (F, 3) face indices
        out_path: output video path
        fps: frame rate
        frames: optional frame segment boundaries for tinting
        mesh_color: RGBA color for the mesh material
        joint_positions: (T, J, 3) joint positions for camera framing.
            If provided, camera parameters are computed from joints (matching
            the skeleton renderer exactly). If None, computed from vertices.
        overlay_keypoints: if True and joint_positions provided, draw skeleton
            keypoints on top of the mesh rendering.
    """
    _render_mesh_video_pytorch3d(
        vertices, faces, out_path, fps, frames, mesh_color, joint_positions,
        overlay_keypoints, image_size=image_size,
        normalize_ground=normalize_ground,
    )


def _create_checkerboard_ground(center_x, center_y, half_extent, n_tiles=8, ground_y=-0.001):
    """Create a checkerboard ground plane as mesh vertices/faces/colors.

    Args:
        center_x, center_y: center of ground in XZ plane
        half_extent: half-size of the ground plane
        n_tiles: number of tiles per side
        ground_y: Y position of the ground plane

    Returns:
        verts: (V, 3) ground vertices
        faces: (F, 3) face indices
        colors: (V, 3) per-vertex colors
    """
    tile_size = 2 * half_extent / n_tiles
    gi, gj = np.meshgrid(np.arange(n_tiles), np.arange(n_tiles), indexing='ij')
    gi, gj = gi.ravel(), gj.ravel()
    n = n_tiles * n_tiles

    x0 = center_x - half_extent + gi * tile_size
    z0 = center_y - half_extent + gj * tile_size
    x1 = x0 + tile_size
    z1 = z0 + tile_size
    y_arr = np.full(n, ground_y)

    # 4 vertices per tile
    verts = np.zeros((n * 4, 3), dtype=np.float32)
    verts[0::4] = np.stack([x0, y_arr, z0], axis=1)
    verts[1::4] = np.stack([x1, y_arr, z0], axis=1)
    verts[2::4] = np.stack([x1, y_arr, z1], axis=1)
    verts[3::4] = np.stack([x0, y_arr, z1], axis=1)

    # Wind faces so normals point up (+Y)
    offsets = np.arange(n) * 4
    faces = np.stack([
        np.stack([offsets, offsets + 2, offsets + 1], axis=1),
        np.stack([offsets, offsets + 3, offsets + 2], axis=1),
    ], axis=1).reshape(-1, 3).astype(np.int64)

    c = np.where((gi + gj) % 2 == 0, 0.93, 0.86)
    colors = np.repeat(c, 4)[:, None] * np.ones((1, 3), dtype=np.float32)

    return verts, faces, colors.astype(np.float32)


def _create_trajectory_mesh(traj_points, half_width=0.017, ground_y=0.0):
    """Create trajectory as a continuous ribbon strip on the ground plane (Y=0).

    Args:
        traj_points: (N, 2) XZ trajectory points
        half_width: half-width of the ribbon

    Returns:
        verts: (V, 3) vertices
        faces: (F, 3) face indices
        colors: (V, 3) per-vertex colors (red)
    """
    N = len(traj_points)
    if N < 2:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
        )

    # Vectorized per-vertex normals (averaged from adjacent segments for smoothness)
    diffs = traj_points[1:] - traj_points[:-1]  # (N-1, 2)
    lengths = np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 1e-6)
    perp = np.stack([-diffs[:, 1], diffs[:, 0]], axis=1) / lengths  # (N-1, 2)

    normals = np.zeros((N, 2), dtype=np.float64)
    normals[:-1] += perp
    normals[1:] += perp
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms

    # Vectorized vertex construction: 2 vertices per trajectory point (left + right)
    offsets = normals * half_width  # (N, 2)
    traj_y = ground_y + 0.002
    verts = np.zeros((N * 2, 3), dtype=np.float32)
    verts[0::2, 0] = traj_points[:, 0] - offsets[:, 0]
    verts[0::2, 1] = traj_y
    verts[0::2, 2] = traj_points[:, 1] - offsets[:, 1]
    verts[1::2, 0] = traj_points[:, 0] + offsets[:, 0]
    verts[1::2, 1] = traj_y
    verts[1::2, 2] = traj_points[:, 1] + offsets[:, 1]

    # Vectorized face construction: 2 triangles per segment
    j = np.arange(N - 1)
    i0 = j * 2
    i1 = i0 + 1
    i2 = (j + 1) * 2
    i3 = i2 + 1
    faces = np.stack([
        np.stack([i0, i2, i1], axis=1),
        np.stack([i1, i2, i3], axis=1),
    ], axis=1).reshape(-1, 3).astype(np.int64)

    colors = np.full((N * 2, 3), [0.85, 0.2, 0.2], dtype=np.float32)

    return verts, faces, colors


def _render_mesh_frames_pytorch3d(
    vertices, faces, frames, mesh_color, joint_positions=None,
    overlay_keypoints=True, image_size=800, normalize_ground=True,
):
    """Render mesh+skeleton frames via pytorch3d, return list[(H,W,3) uint8].

    Same internal pipeline as `_render_mesh_video_pytorch3d` but returns the
    per-frame RGB arrays instead of writing an mp4. Used by the video writer
    wrapper and by grid-composing callers in `utils/debug_viz`.
    """
    import math
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVOrthographicCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        PointLights,
        TexturesVertex,
        BlendParams,
    )

    device = torch.device("cuda:0")
    T = len(vertices)

    # Debug rendering can preserve world height. The old default normalizes each
    # panel independently to the ground, which is useful for inspection but hides
    # PHC failures where the body flies upward or sinks.
    vertices = vertices.copy()
    if joint_positions is not None:
        joint_y_min = float(joint_positions[..., 1].min())
        foot_depth = max(joint_y_min - vertices[..., 1].min(), 0.0)
        if normalize_ground:
            vertices[..., 1] -= joint_y_min
    else:
        joint_y_min = float(vertices[..., 1].min())
        foot_depth = 0.0
        if normalize_ground:
            vertices[..., 1] -= joint_y_min

    # Use root joint position for trajectory (not mesh vertex 0, which is a
    # surface point that wobbles as the body rotates).
    if joint_positions is not None:
        traj = joint_positions[:, 0, [0, 2]]
    else:
        traj = vertices[:, 0, [0, 2]]

    # Camera framing: use joint_positions if provided (matches skeleton renderer),
    # otherwise fall back to mesh vertices.
    if joint_positions is not None:
        cam_ref = joint_positions.copy()
        if normalize_ground:
            cam_ref[..., 1] -= joint_y_min
        cam_pts = cam_ref.reshape(-1, 3)
    else:
        cam_pts = vertices.reshape(-1, 3)

    x_min, x_max = float(cam_pts[:, 0].min()), float(cam_pts[:, 0].max())
    z_min, z_max = float(cam_pts[:, 2].min()), float(cam_pts[:, 2].max())
    if normalize_ground:
        y_min = float(cam_pts[:, 1].min())
        y_range = float(np.ptp(cam_pts[:, 1]))
    else:
        y_min = float(min(cam_pts[:, 1].min(), -foot_depth, 0.0))
        y_max = float(max(cam_pts[:, 1].max(), 1.5))
        y_range = y_max - y_min
    horizontal_range = max(x_max - x_min, z_max - z_min)

    # Camera setup: exactly matching render_skeleton.py (lines 351-388)
    elevation = -math.pi / 10.0   # -18 degrees
    azimuth = -math.pi * 3.0 / 4.0  # -135 degrees

    sk_height = y_range if y_range > 1.0 else 1.5
    motion_ratio = horizontal_range / sk_height
    motion_scale = 1.0 + (motion_ratio - 1.5) * 0.5 if motion_ratio > 1.5 else 1.0

    dist = sk_height * 3.0
    look_at = np.array(
        [(x_min + x_max) / 2, y_min + sk_height * 0.45, (z_min + z_max) / 2],
        dtype=np.float64,
    )

    # Camera vectors: skeleton renderer (no occlusion) places camera behind
    # the person, but front/back look identical for wireframes. For solid mesh
    # rendering we need the camera in front, so we negate the front vector.
    front = -np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.sin(azimuth),
    ])
    front /= np.linalg.norm(front)
    up_init = np.array([0.0, 1.0, 0.0])
    right = np.cross(front, up_init)
    right /= np.linalg.norm(right)
    up = np.cross(right, front)

    cam_pos = look_at + front * dist

    # Build pytorch3d R, T directly from camera vectors.
    # pytorch3d convention: p_cam = p_world @ R + T (row-vector multiply)
    # R columns = [right, up, -front] — matches pytorch3d's look_at_view_transform.
    R = np.stack([right, up, -front], axis=1).astype(np.float32)
    Tv = (-cam_pos @ R).astype(np.float32)
    R = torch.tensor(R, device=device).unsqueeze(0)   # (1,3,3)
    Tv = torch.tensor(Tv, device=device).unsqueeze(0)  # (1,3)

    # Match skeleton renderer framing: skeleton uses screen_scale = 480*0.4/ortho_s
    # with visible half = 240/screen_scale = 1.25*ortho_s where ortho_s=sk_height*0.8*ms.
    # FoVOrthographicCameras visible half = ortho_scale, so use 1.25x.
    ortho_scale = float(sk_height * 0.8 * motion_scale * 1.25)

    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=Tv,
        min_x=-ortho_scale, max_x=ortho_scale,
        min_y=-ortho_scale, max_y=ortho_scale,
        scale_xyz=((1.0, 1.0, 1.0),),
    )

    # Lighting
    lights = PointLights(
        device=device,
        location=((float(cam_pos[0]), float(cam_pos[1]) + 3, float(cam_pos[2]),),),
        ambient_color=((0.4, 0.4, 0.4),),
        diffuse_color=((0.8, 0.8, 0.8),),
        specular_color=((0.05, 0.05, 0.05),),
    )

    # Rasterization settings
    image_size = int(image_size)
    n_faces_total = faces.shape[0] + 200  # body + ground + trajectory
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
        max_faces_per_bin=max(20000, n_faces_total + 5000),
    )

    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights,
            blend_params=blend_params,
        ),
    )

    # Pre-create ground plane (stays constant across frames)
    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2
    extent = max(x_max - x_min, z_max - z_min, sk_height)
    ground_half = float(max(extent * 0.55, horizontal_range * 0.6))
    ground_verts, ground_faces, ground_colors = _create_checkerboard_ground(
        float(center_x), float(center_z), ground_half,
        ground_y=-foot_depth - 0.001,
    )

    faces_tensor = torch.tensor(faces, dtype=torch.int64, device=device)
    n_body_verts = vertices.shape[1]
    body_color_np = np.full((n_body_verts, 3), mesh_color[:3], dtype=np.float32)
    body_color_tensor = torch.tensor(body_color_np, device=device)

    # Pre-upload all frame vertices to GPU at once
    all_body_verts = torch.tensor(vertices, dtype=torch.float32, device=device)  # (T, V, 3)

    # Pre-upload ground tensors to GPU (constant across frames)
    ground_verts_tensor = torch.tensor(ground_verts, dtype=torch.float32, device=device)
    ground_colors_tensor = torch.tensor(ground_colors, dtype=torch.float32, device=device)
    ground_faces_tensor = torch.tensor(ground_faces, dtype=torch.int64, device=device) + n_body_verts

    # Pre-compute the full trajectory mesh once (O(T) instead of O(T^2))
    full_traj_verts, full_traj_faces, full_traj_colors = _create_trajectory_mesh(
        traj, ground_y=-foot_depth,
    )
    traj_offset = n_body_verts + ground_verts_tensor.shape[0]
    if len(full_traj_verts) > 0:
        full_traj_verts_tensor = torch.tensor(full_traj_verts, dtype=torch.float32, device=device)
        full_traj_faces_tensor = torch.tensor(full_traj_faces, dtype=torch.int64, device=device) + traj_offset
        full_traj_colors_tensor = torch.tensor(full_traj_colors, dtype=torch.float32, device=device)
        has_traj = True
    else:
        has_traj = False

    # Pre-allocate combined buffers to avoid per-frame torch.cat
    n_ground = ground_verts_tensor.shape[0]
    n_body_faces = faces_tensor.shape[0]
    n_ground_faces = ground_faces_tensor.shape[0]
    n_traj_verts_max = full_traj_verts_tensor.shape[0] if has_traj else 0
    n_traj_faces_max = full_traj_faces_tensor.shape[0] if has_traj else 0
    n_total_verts_max = n_body_verts + n_ground + n_traj_verts_max
    n_total_faces_max = n_body_faces + n_ground_faces + n_traj_faces_max

    verts_buf = torch.zeros(1, n_total_verts_max, 3, device=device)
    colors_buf = torch.zeros(1, n_total_verts_max, 3, device=device)
    faces_buf = torch.zeros(1, n_total_faces_max, 3, dtype=torch.int64, device=device)

    # Fill in constant regions
    verts_buf[0, n_body_verts:n_body_verts + n_ground] = ground_verts_tensor
    colors_buf[0, :n_body_verts] = body_color_tensor
    colors_buf[0, n_body_verts:n_body_verts + n_ground] = ground_colors_tensor
    faces_buf[0, :n_body_faces] = faces_tensor
    faces_buf[0, n_body_faces:n_body_faces + n_ground_faces] = ground_faces_tensor

    if has_traj:
        traj_v_start = n_body_verts + n_ground
        traj_f_start = n_body_faces + n_ground_faces
        verts_buf[0, traj_v_start:traj_v_start + n_traj_verts_max] = full_traj_verts_tensor
        colors_buf[0, traj_v_start:traj_v_start + n_traj_verts_max] = full_traj_colors_tensor
        faces_buf[0, traj_f_start:traj_f_start + n_traj_faces_max] = full_traj_faces_tensor

    # Skeleton overlay setup: compute camera params at mesh image resolution
    skel_cam = None
    joints_for_draw = None
    skel_chains = None
    if overlay_keypoints and joint_positions is not None:
        skel_cam = compute_camera_params(cam_ref, width=image_size, height=image_size)
        skel_chains = get_humanml3d_chains()
        joints_for_draw = joint_positions.copy()
        if normalize_ground:
            joints_for_draw[..., 1] -= joint_y_min

    frames_out = []

    for i in range(T):
        # Only update body vertices (changes each frame)
        verts_buf[0, :n_body_verts] = all_body_verts[i]

        # Compute active region sizes
        if i > 0 and has_traj:
            n_active_verts = n_body_verts + n_ground + (i + 1) * 2
            n_active_faces = n_body_faces + n_ground_faces + i * 2
        else:
            n_active_verts = n_body_verts + n_ground
            n_active_faces = n_body_faces + n_ground_faces

        # Slice active regions (narrow — no new memory allocation)
        active_verts = verts_buf[:, :n_active_verts]
        active_faces = faces_buf[:, :n_active_faces]
        active_colors = colors_buf[:, :n_active_verts]

        textures = TexturesVertex(verts_features=active_colors)
        mesh = Meshes(verts=active_verts, faces=active_faces, textures=textures)

        # Render
        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False)
            if hasattr(torch, "autocast") and device.type != "cpu"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            images = renderer(mesh)

        img = (
            images[0, ..., :3].float().cpu().numpy() * 255
        ).clip(0, 255).astype(np.uint8)

        # Overlay skeleton keypoints on mesh
        if skel_cam is not None:
            draw_skeleton_on_image(img, joints_for_draw[i], skel_chains, skel_cam)

        # Apply tint for frame segments
        if frames is not None and len(frames) > 1:
            tint_colors = [[255, 220, 220], [220, 255, 220], [220, 220, 255]]
            segment_idx = np.searchsorted(frames, i, side="right")
            tint_factor = np.array(tint_colors[segment_idx % 3]) / 255.0
            img = (img.astype(np.float32) * tint_factor).clip(0, 255).astype(np.uint8)

        frames_out.append(img)

        if (i + 1) % 50 == 0 or i == T - 1:
            print(f"  Rendered frame {i + 1}/{T}")

    # Free GPU memory
    del renderer
    torch.cuda.empty_cache()
    return frames_out


def _render_mesh_video_pytorch3d(
    vertices, faces, out_path, fps, frames, mesh_color, joint_positions=None,
    overlay_keypoints=True, image_size=800, normalize_ground=True,
):
    """Render SMPL mesh video using pytorch3d (writes mp4 file)."""
    import imageio
    rendered = _render_mesh_frames_pytorch3d(
        vertices, faces, frames, mesh_color,
        joint_positions=joint_positions,
        overlay_keypoints=overlay_keypoints,
        image_size=image_size,
        normalize_ground=normalize_ground,
    )
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for img in rendered:
            writer.append_data(img)
    finally:
        writer.close()


