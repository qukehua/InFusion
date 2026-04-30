"""
Preprocess CHICO dataset to generate multimodal evaluation files.

This script generates:
1. data_candi_*.npz - Candidate trajectories for multimodal evaluation
2. t_his*_filtered.npz - Multimodal indices (same history, different future)

Usage:
    python preprocess_chico.py --data_path ./datasets/CHICO --output_dir ./data/chico_multi_modal
"""

import os
import argparse
import numpy as np
from glob import glob
import pickle as pkl
from tqdm import tqdm
import torch


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


def load_chico_sequences(data_path, split_subjects, include_robot=True):
    """
    Load all CHICO sequences.
    CHICO structure: data_path/dataset/Sxx/*.pkl
    """
    dataset_dir = os.path.join(data_path, "dataset")
    all_sequences = []
    sequence_info = []

    for subject in split_subjects:
        subject_dir = os.path.join(dataset_dir, subject)
        if not os.path.exists(subject_dir):
            continue
        pkl_files = sorted(glob(os.path.join(subject_dir, "*.pkl")))
        for pkl_file in tqdm(pkl_files, desc=f"Loading {subject}"):
            sequence_list = load_pkl(pkl_file)
            if len(sequence_list) == 0:
                continue

            human_seq = np.asarray([f[0] for f in sequence_list], dtype=np.float32)
            if include_robot:
                robot_seq = np.asarray([f[1] for f in sequence_list], dtype=np.float32)
                seq = np.concatenate([human_seq, robot_seq], axis=1)
            else:
                seq = human_seq

            # Make relative to root.
            seq[:, 1:] -= seq[:, :1]

            all_sequences.append(seq)
            sequence_info.append(
                {
                    "subject": subject,
                    "action": os.path.splitext(os.path.basename(pkl_file))[0],
                    "file": os.path.basename(pkl_file),
                    "length": len(sequence_list),
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


def compute_multimodal_indices(windows, t_his, thre_his=0.5, thre_pred=0.1):
    n_windows = len(windows)
    if n_windows == 0:
        print("ERROR: No windows to process.")
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
        print(f"Using batch_size={batch_size} for {n_windows} windows")

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
                    print(f"\nCUDA OOM at batch {i}, switching this batch to CPU.")
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHICO for multimodal evaluation")
    parser.add_argument("--data_path", type=str, default="/data/user/qkh/datasets/CHICO", help="Path to CHICO root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/user/qkh/datasets/CHICO/multimodal",
        help="Output directory for preprocessed files",
    )
    parser.add_argument("--t_his", type=int, default=10, help="History frames")
    parser.add_argument("--t_pred", type=int, default=25, help="Prediction frames")
    parser.add_argument("--skip_rate", type=int, default=20, help="Skip rate for extracting windows")
    parser.add_argument("--thre_his", type=float, default=320, help="History similarity threshold")
    parser.add_argument("--thre_pred", type=float, default=1000, help="Future difference threshold")
    parser.add_argument("--include_robot", action="store_true", help="Include robot/tool joints")
    parser.add_argument(
        "--val_subjects",
        nargs="+",
        default=None,
        help="Optional val subjects like S00 S04 (paper default)",
    )
    parser.add_argument(
        "--test_subjects",
        nargs="+",
        default=None,
        help="Optional test subjects like S02 S03 S18 S19 (paper default)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_dir = os.path.join(args.data_path, "dataset")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"CHICO dataset dir not found: {dataset_dir}")

    all_subjects = sorted(
        [d for d in os.listdir(dataset_dir) if d.startswith("S") and os.path.isdir(os.path.join(dataset_dir, d))]
    )
    if len(all_subjects) == 0:
        raise RuntimeError(f"No subject folders found in {dataset_dir}")

    # CHICO paper protocol:
    # val: S00, S04
    # test: S02, S03, S18, S19
    val_default = [s for s in ["S00", "S04"] if s in all_subjects]
    test_default = [s for s in ["S02", "S03", "S18", "S19"] if s in all_subjects]
    val_subjects = args.val_subjects if args.val_subjects is not None else val_default
    test_subjects = args.test_subjects if args.test_subjects is not None else test_default
    train_subjects = [s for s in all_subjects if s not in set(val_subjects + test_subjects)]

    print("=" * 60)
    print("CHICO Preprocessing for TransFusion")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"t_his: {args.t_his}, t_pred: {args.t_pred}, skip_rate: {args.skip_rate}")
    print(f"thre_his: {args.thre_his}, thre_pred: {args.thre_pred}")
    include_robot = args.include_robot
    print(f"include_robot: {include_robot}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test subjects: {test_subjects}")
    print("=" * 60)

    print("\n[1/4] Loading test sequences...")
    sequences, seq_info = load_chico_sequences(args.data_path, test_subjects, include_robot=include_robot)
    print(f"Loaded {len(sequences)} sequences from {len(set([s['subject'] for s in seq_info]))} subjects")

    print("\n[2/4] Extracting sliding windows...")
    windows, window_origins = extract_windows(sequences, args.t_his, args.t_pred, args.skip_rate)
    print(f"Extracted {len(windows)} windows")
    print(f"Window shape: {windows.shape}")

    print("\n[3/4] Saving candidate trajectories...")
    tag = "human_robot" if include_robot else "human_only"
    candi_file = os.path.join(
        args.output_dir,
        f"data_candi_chico_{tag}_t_his{args.t_his}_t_pred{args.t_pred}_skiprate{args.skip_rate}.npz",
    )
    np.savez_compressed(candi_file, **{"data_candidate.npy": windows})
    print(f"Saved: {candi_file}")

    print("\n[4/4] Computing multimodal indices...")
    multimodal_dict = compute_multimodal_indices(windows, args.t_his, args.thre_his, args.thre_pred)
    multi_file = os.path.join(
        args.output_dir,
        f"t_his{args.t_his}_chico_{tag}_thre{args.thre_his:.3f}_t_pred{args.t_pred}_thre{args.thre_pred:.3f}_filtered.npz",
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
