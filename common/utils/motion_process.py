import functools
import os

import numpy as np
import smplx as smplx_lib
import torch
from scipy.spatial.transform import Rotation as ScipyR

from common.utils.math.quaternion import *
from common.utils.math.rotation import matrix_to_axis_angle, rotation_6d_to_matrix

# SMPL-X body models (https://smpl-x.is.tue.mpg.de/): <SMPLX_MODEL_DIR>/smplx/SMPLX_NEUTRAL.npz etc.
# Defaults to deps/smplx_models relative to the working directory; override with the
# SMPLX_MODEL_DIR environment variable or pass smpl_model_path explicitly.
_DEFAULT_SMPLX_DIR = os.environ.get("SMPLX_MODEL_DIR", os.path.join("deps", "smplx_models"))


@functools.lru_cache(maxsize=1)
def _get_tpose_pelvis(smpl_model_path=None):
    """Compute T-pose pelvis (joint 0) position. Cached after first call."""
    if smpl_model_path is None:
        smpl_model_path = _DEFAULT_SMPLX_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx_lib.create(
        smpl_model_path, model_type='smplx', gender='neutral',
        num_betas=10, use_pca=False, flat_hand_mean=True,
    ).to(device).eval()
    with torch.no_grad():
        tpose = model(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            transl=torch.zeros(1, 3, device=device),
            betas=torch.zeros(1, 10, device=device),
        )
    return tpose.joints[0, 0].cpu().numpy()  # (3,)

"""
Motion data structure:
(B: batch size)
root_rot_velocity (B, seq_len, 1)
root_linear_velocity (B, seq_len, 2)
root_y (B, seq_len, 1)
ric_data (B, seq_len, (joint_num - 1)*3)
rot_data (B, seq_len, (joint_num - 1)*6)
local_velocity (B, seq_len, joint_num*3)
foot contact (B, seq_len, 4)
"""


def recover_root_rot_pos(data):
    # recover root rotation and position
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_joint_positions_263(data: np.ndarray, joints_num) -> np.ndarray:
    """
    Recovers 3D joint positions from the rotation-invariant local positions (ric_data).
    This is the most direct way to get the skeleton for animation.
    """
    feature_vec = torch.from_numpy(data).unsqueeze(0).float()
    r_rot_quat, r_pos = recover_root_rot_pos(feature_vec)
    positions = feature_vec[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )
    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    """Concatenate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    joints_np = positions.squeeze(0).detach().cpu().numpy()
    return joints_np


def recover_joint_positions_263_batched(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    """Pure-torch batched version of `recover_joint_positions_263`.

    Stays on whatever device `data` is on (no CPU round-trip), and accepts an
    arbitrary leading batch dim — `recover_root_rot_pos` is already shape-agnostic.

    Args:
        data: (..., T, 263) torch tensor (typically (B, T, 263)).
        joints_num: typically 22 (HML).

    Returns:
        (..., T, joints_num, 3) on the same device/dtype.
    """
    feature_vec = data.float()
    r_rot_quat, r_pos = recover_root_rot_pos(feature_vec)
    positions = feature_vec[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.reshape(positions.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )
    positions = positions.clone()
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


class StreamJointRecovery263:
    """
    Stream version of recover_joint_positions_263 that processes one frame at a time.
    Maintains cumulative state for rotation angles and positions.

    Key insight: The batch version uses PREVIOUS frame's velocity for the current frame,
    so we need to delay the velocity application by one frame.

    Args:
        joints_num: Number of joints in the skeleton
        smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
            - 1.0 = no smoothing (default), output follows input exactly
            - 0.0 = infinite smoothing, output never changes
            - Recommended values: 0.3-0.7 for visible smoothing
            - Formula: smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, joints_num: int, smoothing_alpha: float = 1.0):
        self.joints_num = joints_num
        self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
        self.reset()

    def reset(self):
        """Reset the accumulated state"""
        self.r_rot_ang_accum = 0.0
        self.r_pos_accum = np.array([0.0, 0.0, 0.0])
        # Store previous frame's velocities for delayed application
        self.prev_rot_vel = 0.0
        self.prev_linear_vel = np.array([0.0, 0.0])
        # Store previous smoothed joints for EMA
        self.prev_smoothed_joints = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return joint positions for that frame.

        Args:
            frame_data: numpy array of shape (263,) for a single frame

        Returns:
            joints: numpy array of shape (joints_num, 3) representing joint positions
        """
        # Convert to torch tensor
        feature_vec = torch.from_numpy(frame_data).float()

        # Extract current frame's velocities (will be used in NEXT frame)
        curr_rot_vel = feature_vec[0].item()
        curr_linear_vel = feature_vec[1:3].numpy()

        # Update accumulated rotation angle with PREVIOUS frame's velocity FIRST
        # This matches the batch processing: r_rot_ang[i] uses rot_vel[i-1]
        self.r_rot_ang_accum += self.prev_rot_vel

        # Calculate current rotation quaternion using updated accumulated angle
        r_rot_quat = torch.zeros(4)
        r_rot_quat[0] = np.cos(self.r_rot_ang_accum)
        r_rot_quat[2] = np.sin(self.r_rot_ang_accum)

        # Create velocity vector with Y=0 using PREVIOUS frame's velocity
        r_vel = np.array([self.prev_linear_vel[0], 0.0, self.prev_linear_vel[1]])

        # Apply inverse rotation to velocity using CURRENT rotation
        r_vel_torch = torch.from_numpy(r_vel).float()
        r_vel_rotated = qrot(qinv(r_rot_quat).unsqueeze(0), r_vel_torch.unsqueeze(0))
        r_vel_rotated = r_vel_rotated.squeeze(0).numpy()

        # Update accumulated position with rotated velocity
        self.r_pos_accum += r_vel_rotated

        # Get Y position from data
        r_pos = self.r_pos_accum.copy()
        r_pos[1] = feature_vec[3].item()

        # Extract local joint positions
        positions = feature_vec[4 : (self.joints_num - 1) * 3 + 4]
        positions = positions.view(-1, 3)

        # Apply inverse rotation to local joints
        r_rot_quat_expanded = (
            qinv(r_rot_quat).unsqueeze(0).expand(positions.shape[0], 4)
        )
        positions = qrot(r_rot_quat_expanded, positions)

        # Add root XZ to joints
        positions[:, 0] += r_pos[0]
        positions[:, 2] += r_pos[2]

        # Concatenate root and joints
        r_pos_torch = torch.from_numpy(r_pos).float()
        positions = torch.cat([r_pos_torch.unsqueeze(0), positions], dim=0)

        # Convert to numpy
        joints_np = positions.detach().cpu().numpy()

        # Apply EMA smoothing if enabled
        if self.smoothing_alpha < 1.0:
            if self.prev_smoothed_joints is None:
                # First frame, no smoothing possible
                self.prev_smoothed_joints = joints_np.copy()
            else:
                # EMA: smoothed = alpha * current + (1 - alpha) * previous
                joints_np = (
                    self.smoothing_alpha * joints_np
                    + (1.0 - self.smoothing_alpha) * self.prev_smoothed_joints
                )
                self.prev_smoothed_joints = joints_np.copy()

        # Store current velocities for next frame
        self.prev_rot_vel = curr_rot_vel
        self.prev_linear_vel = curr_linear_vel

        return joints_np


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))

    return np.array(R_total)


def recover_from_local_position(final_x, njoint):
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(
        nfrm, -1, 3
    )  # frames, njoints * 3
    velocities_root_xy_no_heading = final_x[:, :2]  # frames, 2
    global_heading_diff_rot = final_x[:, 2:8]  # frames, 6

    # recover global heading
    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(
        np.repeat(inv_global_heading_rot[:, None, :, :], njoint, axis=1),
        positions_no_heading[..., None],
    ).squeeze(-1)

    # recover root translation
    # add heading to velocities_root_xy_no_heading

    velocities_root_xyz_no_heading = np.zeros(
        (
            velocities_root_xy_no_heading.shape[0],
            3,
        )
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading


def rotations_matrix_to_smpl85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = (
        matrix_to_axis_angle(torch.from_numpy(rotations_matrix))
        .numpy()
        .reshape(nfrm, -1)
    )
    smpl_85 = np.concatenate(
        [axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1
    )
    return smpl_85


def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(
        torch.from_numpy(final_x[:, 8 + 6 * njoint : 8 + 12 * njoint]).reshape(
            nfrm, -1, 6
        )
    ).numpy()
    global_heading_diff_rot = final_x[:, 2:8]
    velocities_root_xy_no_heading = final_x[:, :2]
    positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation
    rotations_matrix[:, 0, ...] = np.matmul(
        inv_global_heading_rot, rotations_matrix[:, 0, ...]
    )
    velocities_root_xyz_no_heading = np.zeros(
        (
            velocities_root_xy_no_heading.shape[0],
            3,
        )
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
    return smpl_85


def recover_joint_positions_272(data: np.ndarray, joints_num) -> np.ndarray:
    return recover_from_local_position(data, joints_num)


def convert_motion_to_joints(
    motion_data: np.ndarray,
    dim,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    fps: float = 30.0,
    joints_num=22,
    feature_layout=None,
):
    """Convert motion data to joint positions.

    Supports:
    - 263-dim HumanML3D format (22 joints)
    - 272-dim format (22 joints)
    - 201D canonical format (22 joints)
    - 138-dim MEI/SMPL format (22 joints)

    Args:
        motion_data: numpy array of shape (K, dim) where K is number of frames
        dim: Feature dimension (int or str like "201d_pos", "201d_fk")
        mean: Optional mean for denormalization
        std: Optional std for denormalization
        fps: Frame rate for velocity conversion (default 30)
        joints_num: Number of joints (default 22)

    Returns:
        joints: numpy array of shape (K, joints_num, 3)
    """
    if mean is not None and std is not None:
        motion_data = motion_data * std + mean

    if dim == 263:
        return recover_joint_positions_263(motion_data, joints_num)
    elif dim == 272:
        return recover_joint_positions_272(motion_data, joints_num)
    elif dim in (201, "201d_pos", "201d_fk"):
        return recover_joint_positions_201d(motion_data, fps=fps)
    elif dim == "201d_world":
        return recover_joint_positions_201d_world(motion_data)
    elif dim in (138, "138"):
        from common.utils.math.mei import recover_joint_positions_138

        return recover_joint_positions_138(motion_data, fps=fps)
    else:
        raise ValueError(f"Unsupported dim: {dim}")


# ============================================================================
# UMR (Universal Motion Representation) Joint Recovery
# ============================================================================


def _recover_root_transform_201d(feature, fps=30):
    """Recover root Y-rotation quaternion and translation from canonical 201D.

    Args:
        feature: (T, 201) canonical 201D feature array (numpy)
        fps: frame rate for velocity-to-displacement conversion

    Returns:
        r_rot_quat: (T, 4) Y-axis rotation quaternion (torch)
        trans: (T, 3) root translation (torch)
    """
    T = feature.shape[0]
    dt = 1.0 / fps

    rot_vel = torch.from_numpy(feature[:, 2].copy()).float()
    vel_xz = torch.from_numpy(feature[:, 0:2].copy()).float()
    root_height = torch.from_numpy(feature[:, 136].copy()).float()

    # Angular velocity -> cumulative Y angle
    r_rot_ang = torch.cumsum(rot_vel, dim=0)
    r_rot_quat = torch.zeros(T, 4)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang)

    # Canonical velocity -> world displacement -> trajectory
    canonical_disp = torch.zeros(T, 3)
    canonical_disp[:, [0, 2]] = vel_xz * dt
    world_disp = qrot(r_rot_quat, canonical_disp)
    trans = torch.cumsum(world_disp, dim=0)
    trans[:, 1] = root_height

    return r_rot_quat, trans


def recover_joint_positions_201d(feature: np.ndarray, fps: float = 30) -> np.ndarray:
    """Recover (T, 22, 3) joint positions from canonical 201D.

    Args:
        feature: (T, 201) canonical 201D feature array
        fps: frame rate for velocity-to-displacement conversion

    Returns:
        positions: (T, 22, 3) joint positions in Y-up world frame
    """
    T = feature.shape[0]
    r_rot_quat, root_pos = _recover_root_transform_201d(feature, fps)

    canonical_pos = torch.from_numpy(feature[:, 135:201].copy()).float().reshape(T, 22, 3)
    world_pos = qrot(r_rot_quat.unsqueeze(1).expand(T, 22, 4), canonical_pos)
    world_pos[:, :, 0] += root_pos[:, 0:1]
    world_pos[:, :, 2] += root_pos[:, 2:3]

    return world_pos.numpy()


@functools.lru_cache(maxsize=1)
def _get_smplx_model_for_fk(smpl_model_path=None):
    """Lazy-load and cache SMPLX model for FK. Cached after first call."""
    if smpl_model_path is None:
        smpl_model_path = _DEFAULT_SMPLX_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx_lib.create(
        smpl_model_path, model_type='smplx', gender='neutral',
        num_betas=10, use_pca=False, flat_hand_mean=True,
    ).to(device).eval()
    return model, device


def recover_joint_positions_201d_fk(feature: np.ndarray, fps=30) -> np.ndarray:
    """Recover (T, 22, 3) world-space joint positions via SMPLX forward kinematics.

    Extracts rot6d from canonical 201D -> rotation matrix -> recover root rotation
    -> axis-angle -> SMPLX FK.  Rotations are stored in SMPLX-compatible Y-up
    frame, so no coordinate conversion is needed.
    """
    T = feature.shape[0]

    # Root trajectory (reuse shared helper)
    _, root_pos = _recover_root_transform_201d(feature, fps)
    root_trans = root_pos.numpy()

    # rot6d -> rotation matrices via rotation.py
    all_rot6d = torch.from_numpy(feature[:, 3:135].copy()).float().reshape(T, 22, 6)
    rotations = rotation_6d_to_matrix(
        all_rot6d.reshape(-1, 6)
    ).reshape(T, 22, 3, 3).numpy()

    # Joint 0: R_root = R_y @ (R_y_inv @ R_root)
    rot_vel = feature[:, 2]
    full_angle = 2.0 * np.cumsum(rot_vel)
    cos_a, sin_a = np.cos(full_angle), np.sin(full_angle)
    R_y = np.zeros((T, 3, 3), dtype=np.float32)
    R_y[:, 0, 0] = cos_a;  R_y[:, 0, 2] = sin_a
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sin_a; R_y[:, 2, 2] = cos_a
    rotations[:, 0] = np.matmul(R_y, rotations[:, 0])

    # axis-angle -> SMPLX FK
    aa = ScipyR.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(T, 22, 3)

    global_orient_aa = aa[:, 0, :]
    body_pose_aa = aa[:, 1:, :].reshape(T, -1)

    # SMPLX forward
    smplx_model, device = _get_smplx_model_for_fk()
    batch_size = 512
    all_pos3d = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        bs = end - start
        go = torch.from_numpy(global_orient_aa[start:end]).float().to(device)
        bp = torch.from_numpy(body_pose_aa[start:end]).float().to(device)
        zeros3 = torch.zeros(bs, 3, device=device)
        zeros10 = torch.zeros(bs, 10, device=device)
        zeros45 = torch.zeros(bs, 45, device=device)
        with torch.no_grad():
            output = smplx_model(
                global_orient=go, body_pose=bp, betas=zeros10,
                jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                left_hand_pose=zeros45, right_hand_pose=zeros45,
                expression=zeros10,
            )
        joints = output.joints[:, :22, :]
        pos3d = (joints - joints[:, 0:1, :]).cpu().numpy()
        all_pos3d.append(pos3d)

    positions = np.concatenate(all_pos3d, axis=0)
    return positions + root_trans[:, None, :]


# ============================================================================
# SMPL Parameter Recovery (for mesh visualization)
# ============================================================================

def global_to_local_rotation(global_rotmat, parents):
    """Convert global rotation matrices to local (parent-relative).

    Args:
        global_rotmat: (T, J, 3, 3) global rotation matrices
        parents: list/array of parent indices, parents[0] = -1 for root

    Returns:
        local_rotmat: (T, J, 3, 3) local rotation matrices
    """
    T, J = global_rotmat.shape[:2]
    local_rotmat = np.zeros_like(global_rotmat)
    local_rotmat[:, 0] = global_rotmat[:, 0]  # root is same
    for j in range(1, J):
        p = parents[j]
        # R_local = R_global_parent^T @ R_global_child
        local_rotmat[:, j] = np.matmul(
            global_rotmat[:, p].transpose(0, 2, 1), global_rotmat[:, j]
        )
    return local_rotmat


def analytical_ik_from_positions(joint_pos, smpl_model_path=None):
    """Recover SMPL parameters from joint positions via optimization-based IK.

    Args:
        joint_pos: (T, 22, 3) joint positions in UMR frame (Y-up)
        smpl_model_path: path to SMPL model directory

    Returns:
        dict with root_orient (T,3), pose_body (T,63), trans (T,3)
    """
    if smpl_model_path is None:
        smpl_model_path = _DEFAULT_SMPLX_DIR

    T = joint_pos.shape[0]
    device = torch.device("cpu")

    # Compute R_y (yaw) from hip/shoulder directions
    r_hip, l_hip, sdr_r, sdr_l = 2, 1, 17, 16
    across = (joint_pos[:, r_hip] - joint_pos[:, l_hip]) + \
             (joint_pos[:, sdr_r] - joint_pos[:, sdr_l])
    across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-8)
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)
    yaw_angle = np.arctan2(forward[:, 0], forward[:, 2]).astype(np.float32)

    # Build R_y matrices (Y-axis rotation)
    cos_y = np.cos(yaw_angle)
    sin_y = np.sin(yaw_angle)
    R_y = np.zeros((T, 3, 3), dtype=np.float32)
    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y

    root_orient_fixed = matrix_to_axis_angle(
        torch.from_numpy(R_y).float()
    ).numpy()

    trans_fixed = joint_pos[:, 0].astype(np.float32)

    with torch.inference_mode(False):
        model = smplx_lib.create(
            smpl_model_path, model_type='smplx', gender='neutral',
            num_betas=10, use_pca=False, flat_hand_mean=True,
        ).to(device).eval()

        target_t = torch.tensor(joint_pos, dtype=torch.float32, device=device)
        root_orient_t = torch.tensor(root_orient_fixed, dtype=torch.float32, device=device)
        trans_t = torch.tensor(trans_fixed, dtype=torch.float32, device=device)
        pose_body = torch.zeros(T, 63, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([pose_body], lr=0.02)

        betas_zero = torch.zeros(T, 10, device=device)
        target_rel = target_t - target_t[:, 0:1]

        for step in range(300):
            optimizer.zero_grad()
            body = model(
                global_orient=root_orient_t,
                body_pose=pose_body,
                transl=trans_t,
                betas=betas_zero,
                jaw_pose=torch.zeros(T, 3, device=device),
                leye_pose=torch.zeros(T, 3, device=device),
                reye_pose=torch.zeros(T, 3, device=device),
                left_hand_pose=torch.zeros(T, 45, device=device),
                right_hand_pose=torch.zeros(T, 45, device=device),
                expression=torch.zeros(T, 10, device=device),
            )
            pred_joints = body.joints[:, :22]
            pred_rel = pred_joints - pred_joints[:, 0:1]

            loss = ((pred_rel - target_rel) ** 2).mean()
            loss = loss + 0.001 * (pose_body ** 2).mean()

            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    joint_loss = ((pred_rel - target_rel) ** 2).mean().item()
                print(f"  IK optim step {step+1}/300, joint_loss={joint_loss:.6f}")

    return {
        "root_orient": root_orient_fixed,
        "pose_body": pose_body.detach().cpu().numpy(),
        "trans": trans_fixed,
    }


def recover_smpl_params_201d(feature: np.ndarray, fps: float = 30,
                              smpl_model_path: str = None) -> dict:
    """Recover SMPL parameters from canonical 201D for mesh rendering.

    Returns trans as SMPLX-compatible transl (pre-subtracted T-pose pelvis offset),
    ready to feed directly to SMPLX forward pass.

    Args:
        feature: (T, 201) canonical 201D feature array
        fps: frame rate
        smpl_model_path: path to SMPLX model .npz file (None for default)

    Returns:
        dict with root_orient (T,3), pose_body (T,63), trans (T,3)
    """
    T = feature.shape[0]

    # Root translation
    _, trans = _recover_root_transform_201d(feature, fps)

    # rot6d -> rotation matrices (row convention, consistent with extraction)
    all_rot6d = torch.from_numpy(feature[:, 3:135].copy()).float().reshape(T, 22, 6)
    rotations = rotation_6d_to_matrix(
        all_rot6d.reshape(-1, 6)
    ).reshape(T, 22, 3, 3)

    # Joint 0: R_root = R_y @ (R_y_inv @ R_root)
    rot_vel = feature[:, 2]
    full_angle = 2.0 * np.cumsum(rot_vel)
    cos_a, sin_a = np.cos(full_angle).astype(np.float32), np.sin(full_angle).astype(np.float32)
    R_y = torch.zeros(T, 3, 3)
    R_y[:, 0, 0] = torch.from_numpy(cos_a)
    R_y[:, 0, 2] = torch.from_numpy(sin_a)
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = torch.from_numpy(-sin_a)
    R_y[:, 2, 2] = torch.from_numpy(cos_a)
    rotations[:, 0] = torch.matmul(R_y, rotations[:, 0])

    # Convert to axis-angle
    root_orient = matrix_to_axis_angle(rotations[:, 0]).numpy()  # (T, 3)
    pose_body = matrix_to_axis_angle(
        rotations[:, 1:].reshape(-1, 3, 3)
    ).reshape(T, -1).numpy()  # (T, 63)

    # Adjust trans: subtract T-pose pelvis offset so SMPLX output
    # pelvis lands at the correct world position.
    tpose_pelvis = _get_tpose_pelvis(smpl_model_path)
    trans_np = trans.numpy() - tpose_pelvis

    return {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans_np,
    }


# ============================================================================
# World 201D Recovery (HY-Motion format)
# ============================================================================


def recover_joint_positions_201d_world(feature: np.ndarray) -> np.ndarray:
    """Recover (T, 22, 3) joint positions from world-frame 201D.

    World 201D layout:
      [0:3]     absolute translation
      [3:9]     root rot6d (world frame)
      [9:135]   body rot6d (local)
      [135:201]  pelvis-relative joint positions (22×3)

    Args:
        feature: (T, 201) world-frame 201D feature array

    Returns:
        positions: (T, 22, 3) joint positions in world frame
    """
    T = feature.shape[0]
    transl = feature[:, 0:3]  # (T, 3)
    pos_relative = feature[:, 135:201].reshape(T, 22, 3)  # pelvis-relative
    positions = pos_relative + transl[:, None, :]  # add pelvis position
    return positions


def recover_smpl_params_201d_world(feature: np.ndarray,
                                    smpl_model_path: str = None,
                                    rot6d_convention: str = "row") -> dict:
    """Recover SMPL parameters from world-frame 201D for mesh rendering.

    Args:
        feature: (T, 201) world-frame 201D feature array
        smpl_model_path: path to SMPLX model directory
        rot6d_convention: 'row' or 'col' — how to decode rot6d to rotation matrix

    Returns:
        dict with root_orient (T,3), pose_body (T,63), trans (T,3)
    """
    if rot6d_convention == "col":
        from common.utils.math.rotation import cont6d_to_matrix
        decode_fn = cont6d_to_matrix
    else:
        decode_fn = rotation_6d_to_matrix

    T = feature.shape[0]

    # Translation: directly from [0:3]
    transl = torch.from_numpy(feature[:, 0:3].copy()).float()

    # Root rotation: [3:9] raw world-frame rot6d
    root_rot6d = torch.from_numpy(feature[:, 3:9].copy()).float()
    root_rotmat = decode_fn(root_rot6d)  # (T, 3, 3)
    root_orient = matrix_to_axis_angle(root_rotmat).numpy()  # (T, 3)

    # Body rotations: [9:135] local rot6d, 21 joints
    body_rot6d = torch.from_numpy(feature[:, 9:135].copy()).float().reshape(T * 21, 6)
    body_rotmat = decode_fn(body_rot6d).reshape(T, 21, 3, 3)
    pose_body = matrix_to_axis_angle(
        body_rotmat.reshape(-1, 3, 3)
    ).reshape(T, -1).numpy()  # (T, 63)

    # Adjust trans: subtract T-pose pelvis offset
    tpose_pelvis = _get_tpose_pelvis(smpl_model_path)
    trans_np = transl.numpy() - tpose_pelvis

    return {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans_np,
    }


def convert_motion_to_smpl_params(
    motion_data: np.ndarray,
    dim,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    fps: float = 30.0,
    smpl_model_path: str = None,
    feature_layout=None,
) -> dict:
    """Convert motion features to SMPL parameters for mesh rendering.

    Args:
        motion_data: (T, D) feature array
        dim: feature dimension (int or str like "201d_pos", "201d_fk")
        mean: optional mean for denormalization
        std: optional std for denormalization
        fps: frame rate
        smpl_model_path: path to SMPLX model .npz file (None for default)

    Returns:
        dict with root_orient (T,3), pose_body (T,63), trans (T,3)
    """
    if mean is not None and std is not None:
        motion_data = motion_data * std + mean

    if dim in (201, "201d_pos", "201d_fk"):
        return recover_smpl_params_201d(motion_data, fps=fps, smpl_model_path=smpl_model_path)
    elif dim in ("201d_world",):
        return recover_smpl_params_201d_world(motion_data, smpl_model_path=smpl_model_path)
    elif dim in (138, "138"):
        # MEI138 carries SMPL params natively (root_orient/body_pose/transl).
        # IK from joint positions would re-introduce twist artefacts, so we
        # forbid this path and direct callers to the shared 138 mesh entry.
        raise NotImplementedError(
            "MEI138 mesh rendering must go through "
            "utils.render_mesh.render_mei138_mesh_video, which uses the "
            "single-FK `recover_mei138_mesh` path (no IK)."
        )
    else:
        # Fallback: recover joints -> analytical IK
        joint_pos = convert_motion_to_joints(motion_data, dim, fps=fps)
        return analytical_ik_from_positions(joint_pos)
