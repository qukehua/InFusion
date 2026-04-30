import os
import random
import torch
import numpy as np


def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_pad(padding, t_his, t_pred):
    zero_index = None
    if padding == 'Zero':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = max(idx_pad)
    elif padding == 'Repeat':
        idx_pad = list(range(t_his)) * int(((t_pred + t_his) / t_his))
        # [0, 1, 2,....,24, 0, 1, 2,....,24, 0, 1, 2,...., 24...]
    elif padding == 'LastFrame':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        # [0, 1, 2,....,24, 24, 24,.....]
    else:
        raise NotImplementedError(f"unknown padding method: {padding}")
    return idx_pad, zero_index


def padding_traj(traj, padding, idx_pad, zero_index):
    if padding == 'Zero':
        traj_tmp = traj
        traj_tmp[..., zero_index, :] = 0
        traj_pad = traj_tmp[..., idx_pad, :]
    else:
        traj_pad = traj[..., idx_pad, :]

    return traj_pad


def post_process(pred, cfg):
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
    pred = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (pred.shape[0], 1, 1, 1)), pred),
                          axis=2)
    pred[..., :1, :] = 0
    return pred


def get_prediction_traj(traj, cfg):
    """
    Select the joints that the model should predict.

    For Harper3D we can condition on human + robot joints while still predicting
    only the human body. In that case the first 21 joints are the human skeleton.
    """
    if getattr(cfg, 'dataset', None) == 'harper3d' and getattr(cfg, 'predict_human_only', False):
        return traj[..., :cfg.output_total_joints, :]
    if getattr(cfg, 'dataset', None) == 'chico' and getattr(cfg, 'predict_human_only', False):
        return traj[..., :cfg.output_total_joints, :]
    if getattr(cfg, 'dataset', None) == 'comad' and getattr(cfg, 'predict_human_only', False):
        return traj[..., :cfg.output_total_joints, :]
    return traj


def flatten_motion_joints(traj):
    """Drop the root joint and flatten xyz coordinates."""
    return traj[..., 1:, :].reshape(*traj.shape[:-2], -1)


def motion_to_velocity(traj):
    """
    Convert position sequence [B, T, D] (or [T, D]) into frame-to-frame velocity.

    The first frame has no previous frame, so we use zeros there and then
    standard backward differences for the remaining frames.
    """
    if isinstance(traj, torch.Tensor):
        vel = torch.zeros_like(traj)
        vel[..., 1:, :] = traj[..., 1:, :] - traj[..., :-1, :]
    else:
        vel = np.zeros_like(traj)
        vel[..., 1:, :] = traj[..., 1:, :] - traj[..., :-1, :]
    return vel


def get_position_inputs(traj, cfg):
    """
    Build flattened position trajectories for target and condition branches.
    """
    target_joints = get_prediction_traj(traj, cfg)
    if getattr(cfg, 'dataset', None) == 'harper3d' and getattr(cfg, 'use_spot_condition', False):
        cond_joints = traj
    elif getattr(cfg, 'dataset', None) == 'chico' and getattr(cfg, 'use_robot_condition', False):
        cond_joints = traj
    elif getattr(cfg, 'dataset', None) == 'comad' and (
        getattr(cfg, 'use_hr_robot_condition', False) or getattr(cfg, 'use_hh_human_condition', False)
    ):
        # CoMad scene-aware conditioning:
        # - HR scenes: robot joints as condition (controlled by use_hr_robot_condition)
        # - HH scenes: Person_2 joints as condition (controlled by use_hh_human_condition)
        cond_joints = traj.copy() if not isinstance(traj, torch.Tensor) else traj.clone()
        p1 = getattr(cfg, 'comad_p1_joints', 25)
        p2 = getattr(cfg, 'comad_p2_joints', 25)
        rb = getattr(cfg, 'comad_robot_joints', 12)
        p2_slice = slice(p1, p1 + p2)
        rb_slice = slice(p1 + p2, p1 + p2 + rb)

        if rb > 0:
            if isinstance(cond_joints, torch.Tensor):
                robot_energy = torch.sum(torch.abs(cond_joints[..., rb_slice, :]), dim=(-1, -2, -3))
                hr_mask = robot_energy > 1e-6
                hh_mask = ~hr_mask
                if p2 > 0:
                    cond_joints[hr_mask, :, p2_slice, :] = 0
                if not getattr(cfg, 'use_hr_robot_condition', False):
                    cond_joints[hr_mask, :, rb_slice, :] = 0
                cond_joints[hh_mask, :, rb_slice, :] = 0
                if p2 > 0 and not getattr(cfg, 'use_hh_human_condition', False):
                    cond_joints[hh_mask, :, p2_slice, :] = 0
            else:
                robot_energy = np.sum(np.abs(cond_joints[..., rb_slice, :]), axis=(-1, -2, -3))
                hr_mask = robot_energy > 1e-6
                hh_mask = ~hr_mask
                if p2 > 0:
                    cond_joints[hr_mask, :, p2_slice, :] = 0
                if not getattr(cfg, 'use_hr_robot_condition', False):
                    cond_joints[hr_mask, :, rb_slice, :] = 0
                cond_joints[hh_mask, :, rb_slice, :] = 0
                if p2 > 0 and not getattr(cfg, 'use_hh_human_condition', False):
                    cond_joints[hh_mask, :, p2_slice, :] = 0
    else:
        cond_joints = target_joints
    return flatten_motion_joints(target_joints), flatten_motion_joints(cond_joints)


def split_motion_inputs(traj, cfg):
    """
    Build model target and conditioning motion tensors from a raw joint sequence.

    Returns:
        target_traj: flattened trajectory used as diffusion target / prediction.
        cond_traj: flattened trajectory used as conditioning input.
    """
    target_traj, cond_traj = get_position_inputs(traj, cfg)
    if getattr(cfg, 'use_velocity_input', False):
        target_traj = motion_to_velocity(target_traj)
        cond_traj = motion_to_velocity(cond_traj)
    return target_traj, cond_traj


def reconstruct_from_velocity(motion, reference_traj, cfg):
    """
    Convert model output back to position features when the model operates on velocity.

    Args:
        motion: [B, T, D] predicted motion in model space
        reference_traj: raw joint sequence [B_ref, T, J, 3] (or compatible)
    Returns:
        [B, T, D] position features
    """
    if not getattr(cfg, 'use_velocity_input', False):
        return motion

    ref_pos, _ = get_position_inputs(reference_traj, cfg)
    ref_first = ref_pos[:, :1, :]

    if isinstance(motion, torch.Tensor):
        if not isinstance(ref_first, torch.Tensor):
            ref_first = torch.as_tensor(ref_first, device=motion.device, dtype=motion.dtype)
        else:
            ref_first = ref_first.to(device=motion.device, dtype=motion.dtype)
        if ref_first.shape[0] == 1 and motion.shape[0] != 1:
            ref_first = ref_first.repeat(motion.shape[0], 1, 1)
        return torch.cumsum(motion, dim=1) + ref_first

    if isinstance(ref_first, torch.Tensor):
        ref_first = ref_first.detach().cpu().numpy()
    if ref_first.shape[0] == 1 and motion.shape[0] != 1:
        ref_first = np.repeat(ref_first, motion.shape[0], axis=0)
    return np.cumsum(motion, axis=1) + ref_first


def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances


def _pairwise_distances_l1(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    distances = torch.abs(embeddings[None, :, :] - embeddings[:, None, :])
    return distances


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def absolute2relative(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = np.linalg.norm(x0[..., 1:, :] - x0[..., parents[1:], :], axis=-1, keepdims=True)
        xt = x * limb_l
        xt0 = np.zeros_like(xt[..., :1, :])
        xt = np.concatenate([xt0, xt], axis=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def absolute2relative_torch(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = torch.norm(x0[..., 1:, :] - x0[..., parents[1:], :], dim=-1, keepdim=True)
        xt = x * limb_l
        xt0 = torch.zeros_like(xt[..., :1, :])
        xt = torch.cat([xt0, xt], dim=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt
