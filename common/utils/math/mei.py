"""MEI 138D representation helpers.

The representation is SE(2)-invariant on the ground plane:
absolute XZ translation and yaw are represented by local velocities, while
the root/body pose is stored as 6D rotations.  All functions here are numpy
implementations so extraction, rendering, and validation can share one
definition without depending on torch.
"""

from __future__ import annotations

import numpy as np

MEI_DIM = 138
MEI_DEFAULT_FPS = 20.0
NUM_JOINTS = 22

# Row-vector point transform from SMPLX/AMASS model coordinates to the
# renderer/HumanML Y-up frame used by this project: (x, y, z) -> (-x, z, y).
SMPLX_TO_MEI_COORDS = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
SMPLX_TO_MEI_BASIS = SMPLX_TO_MEI_COORDS.T

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]

# Source MEI138 currently uses the pelvis as the CoM proxy.
_MASS_WEIGHTS = np.array([1] + [0] * 21, dtype=np.float64)
JOINT_MASS_WEIGHTS = _MASS_WEIGHTS / _MASS_WEIGHTS.sum()


def _sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions: np.ndarray) -> np.ndarray:
    """Return equivalent WXYZ quaternions with non-negative real part."""
    return np.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    nonzero = angles != 0
    safe_angles = np.where(nonzero, angles, np.ones_like(angles))
    sin_half_angles_over_angles = np.where(
        nonzero, np.sin(half_angles) / safe_angles, 0.5,
    )
    return np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        axis=-1,
    )


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    r, i, j, k = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    out = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return out.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.split(
        matrix.reshape(batch_dim + (9,)), 9, axis=-1,
    )
    m00 = m00[..., 0]
    m01 = m01[..., 0]
    m02 = m02[..., 0]
    m10 = m10[..., 0]
    m11 = m11[..., 0]
    m12 = m12[..., 0]
    m20 = m20[..., 0]
    m21 = m21[..., 0]
    m22 = m22[..., 0]

    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], 0.1))
    best = q_abs.argmax(axis=-1)
    flat_candidates = quat_candidates.reshape(-1, 4, 4)
    flat_best = best.reshape(-1)
    out = flat_candidates[np.arange(flat_candidates.shape[0]), flat_best, :]
    return standardize_quaternion(out.reshape(batch_dim + (4,)))


def quaternion_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    nonzero = angles != 0
    safe_angles = np.where(nonzero, angles, np.ones_like(angles))
    sin_half_angles_over_angles = np.where(
        nonzero, np.sin(half_angles) / safe_angles, 0.5,
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode MEI/source interleaved 6D rotations to matrices."""
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Encode the first two matrix columns using MEI/source interleaving."""
    v1 = matrix[..., 0:1]
    v2 = matrix[..., 1:2]
    return np.concatenate([v1, v2], axis=-1).reshape(*matrix.shape[:-2], 6)


def yaw_rotation_matrix(angle: np.ndarray) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    z = np.zeros_like(angle)
    o = np.ones_like(angle)
    return np.stack(
        [c, z, s, z, o, z, -s, z, c],
        axis=-1,
    ).reshape(*angle.shape, 3, 3)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def smplx_to_mei_points(points: np.ndarray) -> np.ndarray:
    """z-up world position -> y-up world position via `pts @ C` (frame change).

    Numerically equivalent to `(C @ pts.T).T`. Apply to AMASS / PHC z-up
    translations + joint positions to bring them into the y-up SMPL canonical
    world frame that the SMPLX model expects.
    """
    return np.asarray(points) @ SMPLX_TO_MEI_COORDS


def mei_to_smplx_points(points: np.ndarray) -> np.ndarray:
    """Reverse of `smplx_to_mei_points`: y-up -> z-up via `pts @ C.T`."""
    return np.asarray(points) @ SMPLX_TO_MEI_COORDS.T


def smplx_to_mei_root_orient(root_axis_angle: np.ndarray) -> np.ndarray:
    """z-up world root rotation -> y-up world root rotation (frame change).

    Per docs/201d_motion_format.md:130 - root_orient is a world-frame rotation
    operator that maps SMPL T-pose (model y-up) to the final pose. To rebase it
    into a different world frame, *left-multiply* by the change-of-basis matrix:

        R_yup = SMPLX_TO_MEI_COORDS @ R_zup

    Apply ONLY to root_orient (and any other world-frame rotation). Do NOT
    apply to body_pose (1-21) / hand_pose: those are parent-local rotations
    in the SMPL kinematic tree, frame-independent — pass them through unchanged.
    """
    R_zup = axis_angle_to_matrix(root_axis_angle)
    R_yup = np.einsum("ij,...jk->...ik", SMPLX_TO_MEI_COORDS, R_zup)
    return matrix_to_axis_angle(R_yup)


def mei_to_smplx_root_orient(root_axis_angle: np.ndarray) -> np.ndarray:
    """Reverse of `smplx_to_mei_root_orient`: y-up -> z-up (left-multiply C.T)."""
    R_yup = axis_angle_to_matrix(root_axis_angle)
    R_zup = np.einsum("ij,...jk->...ik", SMPLX_TO_MEI_COORDS.T, R_yup)
    return matrix_to_axis_angle(R_zup)


def _compute_com(joints: np.ndarray) -> np.ndarray:
    w = JOINT_MASS_WEIGHTS.reshape(1, -1, 1)
    return (joints * w).sum(axis=1)


def compute_com_ground(joints: np.ndarray) -> np.ndarray:
    com = _compute_com(joints)
    com[:, 1] = 0.0
    return com


def compute_heading(joints: np.ndarray) -> np.ndarray:
    r_hip = SMPL_JOINT_NAMES.index("R_Hip")
    l_hip = SMPL_JOINT_NAMES.index("L_Hip")
    sdr_r = SMPL_JOINT_NAMES.index("R_Shoulder")
    sdr_l = SMPL_JOINT_NAMES.index("L_Shoulder")

    across = (joints[:, r_hip] - joints[:, l_hip]) + (
        joints[:, sdr_r] - joints[:, sdr_l]
    )
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    xz_norm = np.linalg.norm(forward, axis=-1)
    forward = forward / np.maximum(xz_norm[:, None], 1e-12)
    raw_heading = np.arctan2(forward[:, 0], forward[:, 2])

    heading = np.empty_like(raw_heading)
    heading[0] = raw_heading[0]
    for t in range(1, len(heading)):
        if xz_norm[t] < 1e-3:
            heading[t] = heading[t - 1]
            continue
        h = raw_heading[t]
        h_flip = h + np.pi
        diff_h = abs(wrap_angle(h - heading[t - 1]))
        diff_flip = abs(wrap_angle(h_flip - heading[t - 1]))
        heading[t] = h if diff_h <= diff_flip else wrap_angle(h_flip)
    return heading


def smplh_to_mei(
    root_orient: np.ndarray,
    body_pose: np.ndarray,
    joints: np.ndarray,
    fps: float = MEI_DEFAULT_FPS,
) -> np.ndarray:
    """Convert SMPL-H root/body pose and 22 joints to MEI138."""
    T = root_orient.shape[0]
    if body_pose.ndim == 2 and body_pose.shape[-1] == 63:
        body_pose = body_pose.reshape(T, 21, 3)

    heading = compute_heading(joints)
    com_ground = compute_com_ground(joints)

    heading_vel = np.zeros(T)
    if T > 1:
        heading_vel[0] = wrap_angle(heading[1] - heading[0]) * fps
        heading_vel[1:] = wrap_angle(heading[1:] - heading[:-1]) * fps

    R_heading = yaw_rotation_matrix(heading)
    R_inv = np.transpose(R_heading, (0, 2, 1))

    com_vel_global = np.zeros_like(com_ground)
    if T > 1:
        com_vel_global[0] = (com_ground[1] - com_ground[0]) * fps
        com_vel_global[1:] = (com_ground[1:] - com_ground[:-1]) * fps

    com_vel_local = np.einsum("tij,tj->ti", R_inv, com_vel_global)
    com_vel_planar = com_vel_local[:, [0, 2]]

    root_offset_global = joints[:, 0] - com_ground
    root_offset_local = np.einsum("tij,tj->ti", R_inv, root_offset_global)

    root_rotmat = axis_angle_to_matrix(root_orient)
    root_rotmat_local = np.einsum("tij,tjk->tik", R_inv, root_rotmat)
    root_rot6d = matrix_to_rotation_6d(root_rotmat_local)

    body_rotmat = axis_angle_to_matrix(body_pose)
    body_rot6d = matrix_to_rotation_6d(body_rotmat).reshape(T, -1)

    mei = np.concatenate(
        [
            com_vel_planar,
            heading_vel[:, None],
            root_offset_local,
            root_rot6d,
            body_rot6d,
        ],
        axis=-1,
    )
    if mei.shape != (T, MEI_DIM):
        raise ValueError(f"Expected ({T}, {MEI_DIM}), got {mei.shape}")
    return mei.astype(np.float32)


def mei_to_smplh(
    mei: np.ndarray,
    fps: float = MEI_DEFAULT_FPS,
    initial_heading: float = 0.0,
    initial_com: np.ndarray | None = None,
) -> dict:
    """Recover SMPL-H root/body pose and pelvis position from MEI138."""
    if initial_com is None:
        initial_com = np.array([0.0, 0.0], dtype=np.float64)

    T = mei.shape[0]
    com_vel_planar = mei[:, 0:2]
    heading_vel = mei[:, 2]
    root_offset_local = mei[:, 3:6]
    root_rot6d = mei[:, 6:12]
    body_rot6d = mei[:, 12:138]

    heading = np.zeros(T)
    heading[0] = initial_heading
    for t in range(1, T):
        heading[t] = heading[t - 1] + heading_vel[t] / fps
    heading = wrap_angle(heading)

    R_heading = yaw_rotation_matrix(heading)
    com_vel_local_3d = np.zeros((T, 3))
    com_vel_local_3d[:, 0] = com_vel_planar[:, 0]
    com_vel_local_3d[:, 2] = com_vel_planar[:, 1]
    com_vel_global = np.einsum("tij,tj->ti", R_heading, com_vel_local_3d)

    com_ground = np.zeros((T, 3))
    com_ground[0, 0] = initial_com[0]
    com_ground[0, 2] = initial_com[1]
    for t in range(1, T):
        com_ground[t] = com_ground[t - 1] + com_vel_global[t] / fps

    root_offset_global = np.einsum("tij,tj->ti", R_heading, root_offset_local)
    transl = com_ground + root_offset_global

    root_rotmat_local = rotation_6d_to_matrix(root_rot6d)
    root_rotmat = np.einsum("tij,tjk->tik", R_heading, root_rotmat_local)
    root_orient = matrix_to_axis_angle(root_rotmat)

    body_rotmat = rotation_6d_to_matrix(body_rot6d.reshape(T, 21, 6))
    body_pose = matrix_to_axis_angle(body_rotmat).reshape(T, 63)

    return {
        "root_orient": root_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "transl": transl.astype(np.float32),
    }


_SMPL_22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

_SMPL_NEUTRAL_REST_JOINTS = np.array(
    [
        [-1.79505953e-03, -2.23333446e-01,  2.82191255e-02],
        [ 6.77246757e-02, -3.14739671e-01,  2.14037877e-02],
        [-6.94655406e-02, -3.13855126e-01,  2.38993038e-02],
        [-4.32792313e-03, -1.14370215e-01,  1.52281192e-03],
        [ 1.02001221e-01, -6.89938274e-01,  1.69079858e-02],
        [-1.07755594e-01, -6.96424140e-01,  1.50492738e-02],
        [ 1.15910534e-03,  2.08102144e-02,  2.61528404e-03],
        [ 8.84055199e-02, -1.08789863e00, -2.67853442e-02],
        [-9.19818258e-02, -1.09483879e00, -2.72625243e-02],
        [ 2.61610388e-03,  7.37324481e-02,  2.80398521e-02],
        [ 1.14763659e-01, -1.14368952e00,  9.25030544e-02],
        [-1.17353574e-01, -1.14298274e00,  9.60854266e-02],
        [-1.62284535e-04,  2.87602804e-01, -1.48171829e-02],
        [ 8.14608431e-02,  1.95481750e-01, -6.04975478e-03],
        [-7.91430834e-02,  1.92565283e-01, -1.05754332e-02],
        [ 4.98955543e-03,  3.52572414e-01,  3.65317875e-02],
        [ 1.72437770e-01,  2.25950646e-01, -1.49179062e-02],
        [-1.75155461e-01,  2.25116450e-01, -1.97185045e-02],
        [ 4.32050017e-01,  2.13178586e-01, -4.23743412e-02],
        [-4.28897421e-01,  2.11787231e-01, -4.11194829e-02],
        [ 6.81283645e-01,  2.22164620e-01, -4.35452575e-02],
        [-6.84195501e-01,  2.19559526e-01, -4.66786778e-02],
    ],
    dtype=np.float64,
)


def smpl_fk_22(
    root_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    """Forward kinematics on the neutral 22-joint SMPL-H body skeleton."""
    T = root_orient.shape[0]
    root_R = axis_angle_to_matrix(root_orient)
    body_R = axis_angle_to_matrix(body_pose.reshape(T, 21, 3))

    joints = np.zeros((T, 22, 3))
    global_R = np.zeros((T, 22, 3, 3))
    global_R[:, 0] = root_R
    joints[:, 0] = transl

    for j in range(1, 22):
        parent = _SMPL_22_PARENTS[j]
        bone = _SMPL_NEUTRAL_REST_JOINTS[j] - _SMPL_NEUTRAL_REST_JOINTS[parent]
        global_R[:, j] = np.einsum("tij,tjk->tik", global_R[:, parent], body_R[:, j - 1])
        joints[:, j] = joints[:, parent] + np.einsum("tij,j->ti", global_R[:, parent], bone)
    return joints.astype(np.float32)


def recover_joint_positions_138(data: np.ndarray, fps: float = MEI_DEFAULT_FPS) -> np.ndarray:
    params = mei_to_smplh(data.astype(np.float64), fps=fps)
    return smpl_fk_22(params["root_orient"], params["body_pose"], params["transl"])


def encode_smpl_pose_to_138d(
    root_orient: np.ndarray,
    body_pose: np.ndarray,
    joints_22: np.ndarray,
    fps: float = MEI_DEFAULT_FPS,
    floor_align: bool = False,
) -> np.ndarray:
    """Standard 138d encoder. Inputs already in y-up SMPL canonical frame.

    Caller is responsible for the z-up -> y-up flip BEFORE calling:
        - root_orient: use `smplx_to_mei_root_orient` (left-multiply C)
        - body_pose:   pass through unchanged (parent-local, frame-agnostic)
        - joints_22:   use `smplx_to_mei_points` (left-multiply C, position)

    See docs/201d_motion_format.md:130 for the protocol.

    Args:
        root_orient: (T, 3) axis-angle, y-up SMPL canonical, root global rotation
        body_pose:   (T, 63) or (T, 21, 3) axis-angle, y-up SMPL canonical, parent-local
        joints_22:   (T, 22, 3) y-up SMPL canonical world position
        fps: motion frame rate (138d's heading_vel / com_vel_planar are scaled by fps)
        floor_align: if True, lift the lowest joint onto y=0 before encoding.
            AMASS extraction uses True (motion floor isn't always at y=0).
            PHC sim path uses False (sim joint positions already grounded by physics).

    Returns:
        (T, 138) float32 138d feature.
    """
    if floor_align:
        joints_22 = joints_22.copy()
        joints_22[..., 1] -= float(joints_22[..., 1].min())
    return smplh_to_mei(
        root_orient=root_orient,
        body_pose=body_pose,
        joints=joints_22,
        fps=fps,
    ).astype(np.float32)
