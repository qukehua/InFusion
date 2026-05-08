"""
Preprocess CoMad dataset to generate multimodal evaluation files.

This script generates:
1. data_candi_*.npz - Candidate trajectories for multimodal evaluation
2. t_his*_filtered.npz - Multimodal indices (same history, different future)

Usage:
    python preprocess_comad.py --data_path ./datasets/CoMad --output_dir ./datasets/CoMad/multimodal
"""

import os
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import torch


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")


def load_json(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def _entity_or_zeros(seq_dict, key, n_frames, n_joints):
    if key in seq_dict:
        arr = np.asarray(seq_dict[key], dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != n_frames or arr.shape[2] != 3:
            raise ValueError(f"Unexpected shape for key '{key}': {arr.shape}")
        return arr
    return np.zeros((n_frames, n_joints, 3), dtype=np.float32)


def _fit_joint_dim(arr, n_frames, target_joints):
    """
    Normalize entity array to (n_frames, target_joints, 3) by padding/truncating.
    """
    if arr.ndim != 3 or arr.shape[0] != n_frames or arr.shape[2] != 3:
        raise ValueError(f"Unexpected entity shape: {arr.shape}")
    cur_joints = arr.shape[1]
    if cur_joints == target_joints:
        return arr.astype(np.float32)
    if cur_joints > target_joints:
        return arr[:, :target_joints].astype(np.float32)
    pad = np.zeros((n_frames, target_joints - cur_joints, 3), dtype=np.float32)
    return np.concatenate([arr.astype(np.float32), pad], axis=1)


def _parse_joint_indices(joint_indices):
    if joint_indices is None:
        return None
    if isinstance(joint_indices, str):
        text = joint_indices.strip()
        if not text or text.lower() in {"none", "auto"}:
            return None
        return [int(x) for x in text.replace(",", " ").split()]
    return [int(x) for x in joint_indices]


def _fit_or_select_joint_dim(arr, n_frames, target_joints, joint_indices=None, fallback_indices=None):
    if arr.ndim != 3 or arr.shape[0] != n_frames or arr.shape[2] != 3:
        raise ValueError(f"Unexpected entity shape: {arr.shape}")
    indices = _parse_joint_indices(joint_indices)
    fallback = _parse_joint_indices(fallback_indices)
    if indices is not None and len(indices) > 0:
        if arr.shape[1] > max(indices):
            return _fit_joint_dim(arr[:, indices], n_frames, target_joints)
        if fallback is not None and len(fallback) > 0 and arr.shape[1] > max(fallback):
            return _fit_joint_dim(arr[:, fallback], n_frames, target_joints)
        if arr.shape[1] == target_joints:
            return arr.astype(np.float32)
    return _fit_joint_dim(arr, n_frames, target_joints)


def load_comad_sequences(
    data_path,
    split="test",
    include_person2=False,
    include_robot=True,
    actions="all",
    interaction_filter=None,
    p1_joints=9,
    p2_joints=0,
    robot_joints=2,
    p1_joint_indices=(2, 9, 16, 7, 14, 13, 20, 8, 15),
    p1_fallback_joint_indices=(0, 1, 2, 3, 4, 5, 6, 9, 10),
    robot_joint_indices=(10, 11),
    robot_fallback_joint_indices=(8, 9),
):
    split_dir = os.path.join(data_path, split)
    json_files = sorted(glob(os.path.join(split_dir, "*", "*", "*", "data.json")))

    all_sequences = []
    sequence_info = []
    for json_file in tqdm(json_files, desc=f"Loading {split} data"):
        rel = os.path.relpath(json_file, split_dir)
        action, interaction, seq_id, _ = rel.split(os.sep)
        if interaction_filter is not None and interaction not in interaction_filter:
            continue

        if actions != "all":
            if isinstance(actions, str):
                if action != actions:
                    continue
            elif action not in actions:
                continue

        try:
            seq_dict = load_json(json_file)
        except Exception as e:
            print(f"[WARN] Skip malformed json: {json_file} ({e})")
            continue

        if "Person_1" not in seq_dict:
            print(f"[WARN] Skip file without Person_1: {json_file}")
            continue

        has_person2 = "Person_2" in seq_dict
        has_robot = "Robot" in seq_dict

        p1 = np.asarray(seq_dict["Person_1"], dtype=np.float32)
        if p1.ndim != 3 or p1.shape[2] != 3:
            print(f"[WARN] Skip file with invalid Person_1 shape {p1.shape}: {json_file}")
            continue
        n_frames = p1.shape[0]
        p1 = _fit_or_select_joint_dim(
            p1,
            n_frames,
            p1_joints,
            p1_joint_indices,
            p1_fallback_joint_indices,
        )
        entities = [p1]
        if include_person2:
            p2 = _entity_or_zeros(seq_dict, "Person_2", n_frames, p2_joints)
            p2 = _fit_joint_dim(p2, n_frames, p2_joints)
            entities.append(p2)
        if include_robot:
            rb = _entity_or_zeros(seq_dict, "Robot", n_frames, robot_joints)
            rb = _fit_or_select_joint_dim(
                rb,
                n_frames,
                robot_joints,
                robot_joint_indices,
                robot_fallback_joint_indices,
            )
            entities.append(rb)
        seq = np.concatenate(entities, axis=1)

        seq[:, 1:] -= seq[:, :1]
        all_sequences.append(seq)
        sequence_info.append(
            {
                "action": action,
                "interaction": interaction,
                "seq_id": seq_id,
                "modality_type": "human_robot" if has_robot else "human_human",
                "file": json_file,
                "length": seq.shape[0],
            }
        )

    return all_sequences, sequence_info


def extract_windows(sequences, t_his, t_pred, skip_rate):
    t_total = t_his + t_pred
    windows = []
    window_origins = []

    for seq_idx, seq in enumerate(tqdm(sequences, desc="Extracting windows")):
        seq_len = seq.shape[0]
        for i in range(0, seq_len - t_total, skip_rate):
            windows.append(seq[i : i + t_total])
            window_origins.append((seq_idx, i))

    return np.array(windows), window_origins


def _compute_indices_single_group(windows, t_his, thre_his=0.5, thre_pred=0.1):
    n_windows = len(windows)
    if n_windows == 0:
        return {}

    history = windows[:, t_his - 1 : t_his, 1:].reshape(n_windows, -1)
    future = windows[:, t_his:, 1:].reshape(n_windows, -1)
    print(f"Computing pairwise distances for {n_windows} windows using {DEVICE}...")

    multimodal_dict = {}
    if USE_CUDA:
        history_t = torch.tensor(history, dtype=torch.float32, device=DEVICE)
        future_t = torch.tensor(future, dtype=torch.float32, device=DEVICE)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        estimated_mem_per_batch = n_windows * future.shape[1] * 4 * 2
        max_batch = max(1, int(gpu_mem * 0.3 / max(estimated_mem_per_batch, 1)))
        batch_size = min(200, max_batch)

        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CUDA)"):
            batch_end = min(i + batch_size, n_windows)
            try:
                hist_i = history_t[i:batch_end]
                fut_i = future_t[i:batch_end]
                dist_his = torch.norm(hist_i[:, None, :] - history_t[None, :, :], dim=2)
                dist_pred = torch.norm(fut_i[:, None, :] - future_t[None, :, :], dim=2)
                mask = (dist_his <= thre_his) & (dist_pred >= thre_pred)
                for j in range(batch_end - i):
                    idx = i + j
                    mask[j, idx] = False
                    idx_multi = torch.where(mask[j])[0].cpu().numpy().tolist()
                    if len(idx_multi) > 0:
                        multimodal_dict[idx] = idx_multi
                del dist_his, dist_pred, mask, hist_i, fut_i
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    hist_i = history[i:batch_end]
                    fut_i = future[i:batch_end]
                    dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
                    dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
                    for j in range(batch_end - i):
                        idx = i + j
                        mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                        mask[idx] = False
                        idx_multi = np.where(mask)[0].tolist()
                        if len(idx_multi) > 0:
                            multimodal_dict[idx] = idx_multi
                else:
                    raise e
        del history_t, future_t
        torch.cuda.empty_cache()
    else:
        batch_size = 500
        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CPU)"):
            batch_end = min(i + batch_size, n_windows)
            hist_i = history[i:batch_end]
            fut_i = future[i:batch_end]
            dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
            dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
            for j in range(batch_end - i):
                idx = i + j
                mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                mask[idx] = False
                idx_multi = np.where(mask)[0]
                if len(idx_multi) > 0:
                    multimodal_dict[idx] = idx_multi.tolist()

    return multimodal_dict


def compute_multimodal_indices(windows, t_his, thre_his=0.5, thre_pred=0.1, group_labels=None):
    """
    Compute multimodal neighbors.
    If group_labels is provided, windows are only matched within the same group.
    """
    n_windows = len(windows)
    if n_windows == 0:
        print("ERROR: No windows to process.")
        return {}

    if group_labels is None:
        return _compute_indices_single_group(windows, t_his, thre_his, thre_pred)

    multimodal_dict = {}
    labels = np.asarray(group_labels)
    unique_labels = sorted(list(set(labels.tolist())))
    print(f"Computing multimodal indices by groups: {unique_labels}")

    for lb in unique_labels:
        idxs = np.where(labels == lb)[0]
        if len(idxs) == 0:
            continue
        sub_windows = windows[idxs]
        sub_dict = _compute_indices_single_group(sub_windows, t_his, thre_his, thre_pred)
        # Map sub indices back to global indices.
        for sub_i, sub_neighbors in sub_dict.items():
            global_i = int(idxs[sub_i])
            global_neighbors = [int(idxs[j]) for j in sub_neighbors]
            multimodal_dict[global_i] = global_neighbors

    return multimodal_dict


def main():
    parser = argparse.ArgumentParser(description="Preprocess CoMad for multimodal evaluation")
    parser.add_argument("--data_path", type=str, default="/data/user/qkh/datasets/CoMad", help="Path to CoMad root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/user/qkh/datasets/CoMad/multimodal",
        help="Output directory for preprocessed files",
    )
    parser.add_argument("--t_his", type=int, default=15, help="History frames")
    parser.add_argument("--t_pred", type=int, default=15, help="Prediction frames")
    parser.add_argument("--skip_rate", type=int, default=20, help="Skip rate for extracting windows")
    parser.add_argument("--thre_his", type=float, default=0.025, help="History similarity threshold")
    parser.add_argument("--thre_pred", type=float, default=0.1, help="Future difference threshold")
    parser.add_argument("--include_person2", action="store_true", help="Include Person_2 joints")
    parser.add_argument("--include_robot", action="store_true", help="Include Robot joints")
    parser.add_argument("--p1_joints", type=int, default=9, help="Person_1 joint count")
    parser.add_argument("--p2_joints", type=int, default=0, help="Person_2 joint count")
    parser.add_argument("--robot_joints", type=int, default=2, help="Robot joint count")
    parser.add_argument("--p1_joint_indices", nargs="+", type=int, default=[2, 9, 16, 7, 14, 13, 20, 8, 15])
    parser.add_argument("--p1_fallback_joint_indices", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 9, 10])
    parser.add_argument("--robot_joint_indices", nargs="+", type=int, default=[10, 11])
    parser.add_argument("--robot_fallback_joint_indices", nargs="+", type=int, default=[8, 9])
    parser.add_argument(
        "--interactions",
        nargs="+",
        default=["HR"],
        help="Interaction folders to include: HR, HH, or all",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Which split to preprocess")
    parser.add_argument("--actions", nargs="+", default=["all"], help="Action filter, e.g., cart react")
    parser.add_argument(
        "--no_group_by_modality",
        action="store_true",
        help="Disable modality-wise grouping; match all windows together",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    actions = "all" if args.actions == ["all"] else args.actions
    interaction_filter = None if args.interactions == ["all"] else {x.upper() for x in args.interactions}

    print("=" * 60)
    print("CoMad Preprocessing for TransFusion")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Output dir: {args.output_dir}")
    print(f"t_his: {args.t_his}, t_pred: {args.t_pred}, skip_rate: {args.skip_rate}")
    print(f"thre_his: {args.thre_his}, thre_pred: {args.thre_pred}")
    print(f"include_person2: {args.include_person2}, include_robot: {args.include_robot}")
    print(f"p1_joints: {args.p1_joints}, p2_joints: {args.p2_joints}, robot_joints: {args.robot_joints}")
    print(f"interactions: {interaction_filter if interaction_filter is not None else 'all'}")
    print(f"group_by_modality: {not args.no_group_by_modality}")
    print(f"actions: {actions}")
    print("=" * 60)

    print("\n[1/4] Loading sequences...")
    sequences, seq_info = load_comad_sequences(
        args.data_path,
        split=args.split,
        include_person2=args.include_person2,
        include_robot=args.include_robot,
        actions=actions,
        interaction_filter=interaction_filter,
        p1_joints=args.p1_joints,
        p2_joints=args.p2_joints,
        robot_joints=args.robot_joints,
        p1_joint_indices=args.p1_joint_indices,
        p1_fallback_joint_indices=args.p1_fallback_joint_indices,
        robot_joint_indices=args.robot_joint_indices,
        robot_fallback_joint_indices=args.robot_fallback_joint_indices,
    )
    print(f"Loaded {len(sequences)} sequences")

    print("\n[2/4] Extracting sliding windows...")
    windows, window_origins = extract_windows(sequences, args.t_his, args.t_pred, args.skip_rate)
    print(f"Extracted {len(windows)} windows")
    print(f"Window shape: {windows.shape}")

    print("\n[3/4] Saving candidate trajectories...")
    tag = []
    tag.append(f"p1-{args.p1_joints}")
    if args.include_person2:
        tag.append(f"p2-{args.p2_joints}")
    if args.include_robot:
        tag.append(f"robot-{args.robot_joints}")
    tag = "_".join(tag)

    candi_file = os.path.join(
        args.output_dir,
        f"data_candi_comad_{tag}_t_his{args.t_his}_t_pred{args.t_pred}_skiprate{args.skip_rate}.npz",
    )
    np.savez_compressed(candi_file, **{"data_candidate.npy": windows})
    print(f"Saved: {candi_file}")

    print("\n[4/4] Computing multimodal indices...")
    group_labels = None
    if not args.no_group_by_modality:
        # Expand sequence-level labels to window-level labels.
        group_labels = [seq_info[s_idx]["modality_type"] for (s_idx, _) in window_origins]
    multimodal_dict = compute_multimodal_indices(
        windows,
        args.t_his,
        args.thre_his,
        args.thre_pred,
        group_labels=group_labels,
    )
    multi_file = os.path.join(
        args.output_dir,
        f"t_his{args.t_his}_comad_{tag}_thre{args.thre_his:.3f}_t_pred{args.t_pred}_thre{args.thre_pred:.3f}_filtered.npz",
    )
    np.savez_compressed(multi_file, data_multimodal=multimodal_dict)
    print(f"Saved: {multi_file}")

    n_multi = len(multimodal_dict)
    avg_multi = np.mean([len(v) for v in multimodal_dict.values()]) if n_multi > 0 else 0
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    if len(windows) > 0:
        print(f"Windows with multimodal futures: {n_multi}/{len(windows)} ({100 * n_multi / len(windows):.1f}%)")
    print(f"Average multimodal count: {avg_multi:.1f}")
    print("=" * 60)

    print("\nUse these paths in your config:")
    print(f"  multimodal_path: {multi_file}")
    print(f"  data_candi_path: {candi_file}")


if __name__ == "__main__":
    main()
