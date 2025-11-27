from scipy.spatial.transform import Rotation as R
import numpy as np


def convert_quat_wxyz_to_xyzw(q, batch_mode=False):
    if batch_mode:
        return q[:, [1, 2, 3, 0]]
    else:
        return q[[1, 2, 3, 0]]

def convert_quat_xyzw_to_wxyz(q, batch_mode=False):
    if batch_mode:
        return q[:, [3, 0, 1, 2]]
    else:
        return q[[3, 0, 1, 2]]

def convert_state_quat_to_rpy(state_quat):
    """
        state_quat: [object_state(quat, 4), allegro_state(16,)]
    """
    state_rpy = np.zeros(state_quat.shape[0]-1)
    state_rpy[:3] = R.from_quat(convert_quat_wxyz_to_xyzw(state_quat[:4])).as_euler('xyz')
    state_rpy[3:] = state_quat[4:]
    
    return state_rpy

def convert_state_rpy_to_quat(state_rpy):
    """
        state_rpy: [object_state(rpy, 3), allegro_state(16,)]
    """
    state_quat = np.zeros(state_rpy.shape[0]+1)
    state_quat[:4] = convert_quat_xyzw_to_wxyz(R.from_euler('xyz', state_rpy[:3]).as_quat())
    state_quat[4:] = state_rpy[3:]

    return state_quat

def make_skew_symmetric_from_vec(v):
    return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

def CalcNW2Qdot(q):
    E = np.zeros((4, 3))
    E[0, :] = -q[-3:]
    bottom_rows_E = -make_skew_symmetric_from_vec(q[-3:])
    bottom_rows_E[0, 0] = q[0]; bottom_rows_E[1, 1] = q[0]; bottom_rows_E[2, 2] = q[0]
    E[-3:, :] = bottom_rows_E.copy()
    E *= 0.5

    return E

def CalcNQdot2W(q):
    E = np.zeros((3, 4))
    E[:, 0] = -q[-3:]
    right_rows_E = make_skew_symmetric_from_vec(q[-3:])
    right_rows_E[0, 0] = q[0]; right_rows_E[1, 1] = q[0]; right_rows_E[2, 2] = q[0]
    E[:, -3:] = right_rows_E.copy()
    E *= 2

    return E

def convert_Bhat_rpy_to_quat(Bhat_rpy, state_quat):
    ori_slc_ddp = slice(0, 4)
    ori_slc_cqdc = slice(0, 3)

    ndofs_before_ori = 0
    n_dofs_after_ori = 16

    # calc omega <--> qdot projection matrices
    left_mat = np.zeros((20, 19))
    right_mat = np.zeros((19, 20))

    left_mat[ori_slc_ddp, ori_slc_cqdc] = CalcNW2Qdot(state_quat[ori_slc_ddp])
    left_mat[:ori_slc_ddp.start, :ori_slc_cqdc.start] = np.eye(ndofs_before_ori)
    left_mat[ori_slc_ddp.stop:, ori_slc_cqdc.stop:] = np.eye(n_dofs_after_ori)

    right_mat[ori_slc_cqdc, ori_slc_ddp] = CalcNQdot2W(state_quat[ori_slc_ddp])
    right_mat[:ori_slc_cqdc.start, :ori_slc_ddp.start] = np.eye(ndofs_before_ori)
    right_mat[ori_slc_cqdc.stop:, ori_slc_ddp.stop:] = np.eye(n_dofs_after_ori)

    Bhat_quat = left_mat @ Bhat_rpy

    return Bhat_quat

def quat_angle_difference(query_quat, target_quat):
    """
    query_quat: (4,)
    target_quat: (N,4)
    return: (N,) 角度差（弧度）
    """
    q = query_quat / np.linalg.norm(query_quat)
    t = target_quat / np.linalg.norm(target_quat, axis=1, keepdims=True)

    # 点积 (N,)
    dot = np.abs(np.sum(q * t, axis=1))

    # 数值安全裁剪
    dot = np.clip(dot, -1.0, 1.0)

    # 角度差
    angle = 2 * np.arccos(dot)
    return angle
