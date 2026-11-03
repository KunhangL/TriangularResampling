#!/usr/bin/env python3
import math
import os

import cv2
import imageio
import numpy as np
import pyrender
import trimesh

from common.utils.motion_process import (  # noqa: F401
    recover_joint_positions_201d,
    recover_joint_positions_201d_fk,
)

os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_humanml3d_chains():
    return [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]


def get_chain_color_table():
    """Normalized RGB palette used to color consecutive bones."""
    return [
        [254 / 255, 178 / 255, 26 / 255],  # orange
        [0 / 255, 170 / 255, 255 / 255],  # cyan
        [19 / 255, 70 / 255, 134 / 255],  # aquamarine
        [255 / 255, 182 / 255, 0 / 255],  # amber
        [0 / 255, 212 / 255, 126 / 255],  # aquamarine
    ]


def compute_camera_pose(look_at, distance, up=(0, 1, 0)):
    elevation = -math.pi / 10.0
    azimuth = -math.pi * 3.0 / 4.0  # 45 degrees
    front = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ]
    )
    front /= np.linalg.norm(front)
    up = np.array(up) / np.linalg.norm(up)
    right = np.cross(front, up)
    right /= np.linalg.norm(right)
    cam_z = -front

    R = np.stack([right, up, cam_z], axis=1)
    t = look_at - front * distance

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def setup_camera(scene, skeleton_height, look_at_x, look_at_z):
    distance = skeleton_height * 3.0
    look_at = np.array([look_at_x, skeleton_height * 0.45, look_at_z])
    camera = pyrender.OrthographicCamera(
        xmag=skeleton_height * 0.8, ymag=skeleton_height * 0.8
    )
    cam_pose = compute_camera_pose(look_at, distance)
    return scene.add(camera, pose=cam_pose, name="iso_cam")


def update_camera(scene, cam_node, skeleton_height, look_at_x, look_at_z):
    distance = skeleton_height * 3.0
    look_at = np.array([look_at_x, skeleton_height * 0.45, look_at_z])
    new_pose = compute_camera_pose(look_at, distance)
    scene.set_pose(cam_node, pose=new_pose)


def create_ground_plane(traj):
    padding = 0.2
    minx, maxx = traj[:, 0].min() - padding, traj[:, 0].max() + padding
    minz, maxz = traj[:, 1].min() - padding, traj[:, 1].max() + padding

    vertices = np.array(
        [
            [minx, 0, minz],
            [maxx, 0, minz],
            [maxx, 0, maxz],
            [minx, 0, maxz],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    ground = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Set ground to pure white to test lighting
    ground.visual.vertex_colors = np.tile(
        [1.0, 1.0, 1.0, 1.0], (len(ground.vertices), 1)
    )

    # Ensure proper normals for lighting
    ground.fix_normals()

    return ground


def create_skeleton_trimesh(joints, chains):
    meshes = []
    # joints as small spheres
    for p in joints:
        sph = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
        sph.apply_translation(p)
        sph.visual.vertex_colors = np.tile(
            [0 / 255, 128 / 255, 157 / 255, 1.0], (len(sph.vertices), 1)
        )
        meshes.append(sph)

    bone_colors = get_chain_color_table()

    color_index = 0
    z_axis = np.array([0, 0, 1])

    # bones as cylinders
    for chain in chains:
        for i in range(len(chain) - 1):
            p1, p2 = joints[chain[i]], joints[chain[i + 1]]
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            if dist < 1e-3:
                continue

            cyl = trimesh.creation.cylinder(radius=0.02, height=dist, sections=16)
            direction = vec / dist
            if np.allclose(direction, z_axis):
                R = np.eye(3)
            elif np.allclose(direction, -z_axis):
                R = np.diag([-1, -1, -1])
            else:
                v = np.cross(z_axis, direction)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, direction)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s * s))

            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = (p1 + p2) / 2.0
            cyl.apply_transform(M)

            color = bone_colors[color_index % len(bone_colors)]
            cyl.visual.vertex_colors = np.tile(
                np.append(color, 1.0), (len(cyl.vertices), 1)
            )
            meshes.append(cyl)
        color_index += 1

    return trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()


def create_trajectory_mesh(pts, radius=0.01, color=[0, 0, 1, 1]):
    """Small‐sphere approximation of a line of pts."""
    spheres = []
    for p in pts:
        sph = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        sph.apply_translation(p)
        sph.visual.vertex_colors = np.tile([0.0, 0.0, 1.0, 1.0], (len(sph.vertices), 1))
        spheres.append(sph)
    return trimesh.util.concatenate(spheres) if spheres else trimesh.Trimesh()


# rendering function for a skeleton structure
# data: [N, J, 3] N for frames, J for joints
# chains: [[J, J, ...], ...] list of chains, each chain is a list of joint indices
def render_skeleton_video(data, chains, out_path, fps=20, frames: np.ndarray = None):
    traj = data[:, 0, [0, 2]]

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],  # White background
        # Higher ambient light to see ground properly
        ambient_light=[0.5, 0.5, 0.5],
    )

    # Primary directional light for main illumination
    main_light = pyrender.DirectionalLight(
        color=[1.0, 1.0, 1.0],
        intensity=3.0,
    )

    # Position main light to illuminate the scene properly
    main_light_pose = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.8, -0.6, 0],  # Light angled down from above-front
            [0, 0.6, 0.6, 0],  # Light direction
            [0, 0, 0, 1],
        ]
    )

    scene.add(main_light, pose=main_light_pose)

    # Add a secondary light from different angle for better illumination
    fill_light = pyrender.DirectionalLight(
        color=[0.8, 0.8, 0.8],
        intensity=2.0,
    )

    fill_light_pose = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.6, 0.8, 0],  # Light from above-back
            [0, -0.8, 0.6, 0],  # Different angle
            [0, 0, 0, 1],
        ]
    )

    scene.add(fill_light, pose=fill_light_pose)

    bone_colors = get_chain_color_table()

    ground = create_ground_plane(traj)

    # Create ground mesh with proper material for shadow reception
    ground_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # White base color
        metallicFactor=0.0,  # Non-metallic
        roughnessFactor=0.8,  # Somewhat rough for better light scattering
    )

    ground_mesh = pyrender.Mesh.from_trimesh(
        ground, material=ground_material, smooth=True
    )
    scene.add(ground_mesh)

    sk_height = np.ptp(data[..., 1])
    sk_mesh = create_skeleton_trimesh(data[0], chains, bone_colors=bone_colors)
    sk_node = scene.add(pyrender.Mesh.from_trimesh(sk_mesh, smooth=True))

    # initial empty trajectory
    traj_mesh = create_trajectory_mesh(np.zeros((0, 3)))
    traj_node = scene.add(pyrender.Mesh.from_trimesh(traj_mesh, smooth=False))

    cam_node = setup_camera(scene, sk_height, traj[0, 0], traj[0, 1])

    # Create renderer with shadow mapping enabled
    renderer = pyrender.OffscreenRenderer(800, 800)

    # Enable shadows by setting shadow mapping parameters
    render_flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    writer = imageio.get_writer(out_path, fps=fps)

    for i in range(len(data)):
        # update skeleton
        scene.remove_node(sk_node)
        sk_mesh = create_skeleton_trimesh(data[i], chains, bone_colors=bone_colors)
        sk_node = scene.add(pyrender.Mesh.from_trimesh(sk_mesh, smooth=True))

        # update trajectory
        scene.remove_node(traj_node)
        pts3 = np.column_stack([traj[: i + 1, 0], np.zeros(i + 1), traj[: i + 1, 1]])
        traj_mesh = create_trajectory_mesh(pts3)
        traj_node = scene.add(pyrender.Mesh.from_trimesh(traj_mesh, smooth=False))

        # update camera
        update_camera(scene, cam_node, sk_height, traj[i, 0], traj[i, 1])

        # Render with shadows enabled
        color, _ = renderer.render(scene, flags=render_flags)

        # Apply color tint based on frames segments
        if frames is not None and len(frames) > 1:
            # Define 3 colors to cycle through (RGB)
            tint_colors = [
                [255, 220, 220],  # Light Red
                [220, 255, 220],  # Light Green
                [220, 220, 255],  # Light Blue
            ]

            # Find which segment the current frame 'i' belongs to
            segment_idx = np.searchsorted(frames, i, side="right")

            # Cycle through the 3 colors
            current_tint = np.array(tint_colors[segment_idx % 3])

            # Multiply original color with tint to keep original colors
            # Normalize tint to 0-1 range
            tint_factor = current_tint / 255.0

            # Apply tint: color * tint_factor
            # This adds the color cast while preserving original hues (bones, joints)
            color = (
                (color[..., :3].astype(np.float32) * tint_factor)
                .clip(0, 255)
                .astype(np.uint8)
            )

        writer.append_data(color)

    writer.close()
    renderer.delete()


def compute_camera_params(data, width=480, height=480):
    """Compute orthographic camera parameters from joint position data.

    These parameters fully describe the camera projection used by the
    skeleton renderer, so that other renderers (e.g. mesh) can produce
    pixel-aligned images for overlay compositing.

    Args:
        data: (T, J, 3) joint positions.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        dict with keys: cam_pos, right, up, screen_scale, width, height.
    """
    all_points = data.reshape(-1, 3)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    y_range = y_max - y_min
    horizontal_range = max(x_max - x_min, z_max - z_min)

    elevation = -math.pi / 10.0
    azimuth = -math.pi * 3.0 / 4.0

    sk_height = y_range if y_range > 1.0 else 1.5
    motion_ratio = horizontal_range / sk_height
    motion_scale = 1.0 + (motion_ratio - 1.5) * 0.5 if motion_ratio > 1.5 else 1.0

    distance = sk_height * 3.0
    look_at = np.array(
        [(x_min + x_max) / 2, y_min + sk_height * 0.45, (z_min + z_max) / 2]
    )

    front = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ]
    )
    front /= np.linalg.norm(front)
    cam_pos = look_at + front * distance
    up = np.array([0, 1, 0])
    right = np.cross(front, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, front)

    ortho_scale = sk_height * 0.8 * motion_scale
    screen_scale = min(width, height) * 0.4 / ortho_scale

    return {
        "cam_pos": cam_pos,
        "right": right,
        "up": up,
        "screen_scale": screen_scale,
        "width": width,
        "height": height,
    }


def _draw_line(img, p1, p2, color, thickness=2):
    """Draw a thick antialiased line on img between two 2D points (cv2 backend)."""
    cv2.line(
        img,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        tuple(int(c) for c in color),
        int(thickness),
        lineType=cv2.LINE_AA,
    )


def _draw_circle(img, center, radius, color):
    """Draw a filled antialiased circle on img (cv2 backend)."""
    cv2.circle(
        img,
        (int(center[0]), int(center[1])),
        int(radius),
        tuple(int(c) for c in color),
        -1,
        lineType=cv2.LINE_AA,
    )


def _world_to_screen_batch(pts_world, cam_params):
    """Project (..., 3) world points to (..., 2) int32 screen coords.

    Camera is fixed across frames, so callers can project all joints +
    trajectory points up front in one matmul instead of looping per-joint.
    """
    cam_pos = cam_params["cam_pos"]
    right = cam_params["right"]
    up = cam_params["up"]
    screen_scale = cam_params["screen_scale"]
    center_x = cam_params["width"] // 2
    center_y = cam_params["height"] // 2

    rel = np.asarray(pts_world) - cam_pos
    x_cam = rel @ right
    y_cam = rel @ up
    sx = (center_x + x_cam * screen_scale).astype(np.int32)
    sy = (center_y - y_cam * screen_scale).astype(np.int32)
    return np.stack([sx, sy], axis=-1)


def _to_uint8_palette(colors):
    """Convert list of float [0,1] RGB colors to uint8 lists."""
    converted = []
    for color in colors:
        arr = np.array(color, dtype=np.float32)
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
        arr = np.clip(arr[:3], 0.0, 1.0)
        converted.append((arr * 255).astype(np.uint8).tolist())
    return converted


def draw_skeleton_on_image(img, joints, chains, cam_params):
    """Draw skeleton keypoints and bones on an existing image (in-place).

    Args:
        img: (H, W, 3) uint8 image to draw on.
        joints: (J, 3) joint positions in world coordinates.
        chains: list of joint chains for bone drawing.
        cam_params: dict from compute_camera_params().
    """
    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2
    cam_pos = cam_params["cam_pos"]
    right = cam_params["right"]
    up = cam_params["up"]
    screen_scale = cam_params["screen_scale"]

    def world_to_screen(point):
        to_point = np.array(point) - cam_pos
        x_cam = np.dot(to_point, right)
        y_cam = np.dot(to_point, up)
        return (int(center_x + x_cam * screen_scale),
                int(center_y - y_cam * screen_scale))

    bone_colors_uint8 = _to_uint8_palette(get_chain_color_table())

    # Draw bones
    color_index = 0
    for chain in chains:
        for i in range(len(chain) - 1):
            if chain[i] < len(joints) and chain[i + 1] < len(joints):
                p1 = world_to_screen(joints[chain[i]])
                p2 = world_to_screen(joints[chain[i + 1]])
                _draw_line(img, p1, p2,
                           bone_colors_uint8[color_index % len(bone_colors_uint8)],
                           thickness=4)
        color_index += 1

    # Draw joints
    for joint in joints:
        _draw_circle(img, world_to_screen(joint), 3, [0, 100, 255])


def render_skeleton_frames(
    data, chains, frames: np.ndarray = None, camera_params: dict = None
):
    """Pure frame generator: returns (T, H, W, 3) uint8 ndarray.

    Splits the writer step out of render_simple_skeleton_video so callers
    can compose multiple panels in memory before encoding once. Uses cv2
    rasterizers and one-shot batch projection for ~50-100x speedup over
    the numpy meshgrid-per-call path.
    """
    traj = data[:, 0, [0, 2]]  # root joint XZ trajectory
    cam = camera_params if camera_params is not None else compute_camera_params(data)
    width, height = cam["width"], cam["height"]

    bone_colors_uint8 = _to_uint8_palette(get_chain_color_table())

    # Pre-project everything to screen once (camera is constant across frames).
    joints_screen = _world_to_screen_batch(data, cam)  # (T, J, 2)
    traj_world = np.stack(
        [traj[:, 0], np.zeros(len(traj), dtype=traj.dtype), traj[:, 1]], axis=-1
    )
    traj_screen = _world_to_screen_batch(traj_world, cam)  # (T, 2)

    LINE_AA = cv2.LINE_AA
    out = np.full((len(data), height, width, 3), 255, dtype=np.uint8)

    for frame_idx in range(len(data)):
        img = out[frame_idx]

        # Trajectory: cumulative polyline up to current frame in one cv2 call,
        # replacing the original O(T^2) per-frame redraw loop.
        if frame_idx > 0:
            pts = traj_screen[: frame_idx + 1].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [pts], False, (255, 0, 0), 3, lineType=LINE_AA)

        # Bones (per-chain palette cycling)
        joints_now = joints_screen[frame_idx]
        n_joints = len(joints_now)
        color_index = 0
        for chain in chains:
            color = tuple(int(c) for c in bone_colors_uint8[color_index % len(bone_colors_uint8)])
            for i in range(len(chain) - 1):
                if chain[i] < n_joints and chain[i + 1] < n_joints:
                    cv2.line(
                        img,
                        tuple(joints_now[chain[i]]),
                        tuple(joints_now[chain[i + 1]]),
                        color,
                        4,
                        lineType=LINE_AA,
                    )
            color_index += 1

        # Joints (blue circles)
        for jp in joints_now:
            cv2.circle(img, tuple(jp), 3, (0, 100, 255), -1, lineType=LINE_AA)

        # Tint by frame segment
        if frames is not None and len(frames) > 1:
            tint_colors = [
                [255, 220, 220],  # Light Red
                [220, 255, 220],  # Light Green
                [220, 220, 255],  # Light Blue
            ]
            segment_idx = np.searchsorted(frames, frame_idx, side="right")
            tint_factor = np.array(tint_colors[segment_idx % 3]) / 255.0
            out[frame_idx] = (
                (img[..., :3].astype(np.float32) * tint_factor)
                .clip(0, 255)
                .astype(np.uint8)
            )

    return out


def render_simple_skeleton_video(
    data, chains, out_path="results_ultra.mp4", fps=20, frames: np.ndarray = None,
    camera_params: dict = None,
):
    """Thin wrapper: generate frames in memory, encode once with libx264 ultrafast."""
    out = render_skeleton_frames(
        data, chains, frames=frames, camera_params=camera_params
    )
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=[
            "-preset", "ultrafast",
            "-crf", "28",
            "-threads", "1",
            "-pix_fmt", "yuv420p",
        ],
    )
    for img in out:
        writer.append_data(img)
    writer.close()


def main():
    data = np.random.rand(60, 22, 3)
    render_skeleton_video(data, get_humanml3d_chains(), "animation.mp4", fps=20)


if __name__ == "__main__":
    main()
