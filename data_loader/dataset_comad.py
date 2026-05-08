import os
import json
from glob import glob
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
from data_loader.comad_kinematics import (
    COMAD_P1_JOINTS_LEFT,
    COMAD_P1_JOINTS_RIGHT,
    COMAD_P1_PARENTS,
    COMAD_PAPER_HUMAN_JOINT_INDICES,
    COMAD_PAPER_HUMAN_JOINT_INDICES_11,
    COMAD_PAPER_ROBOT_JOINT_INDICES,
    COMAD_UPPER_BODY_11_VIS_LINKS,
    COMAD_HH_VIS_LINKS,
    comad_p1_links,
)


def load_json(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def _entity_or_zeros(seq_dict, key, n_frames, n_joints):
    if key in seq_dict:
        seq_raw = seq_dict[key]
        out = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
        usable_frames = min(n_frames, len(seq_raw))
        for t in range(usable_frames):
            try:
                frame_arr = np.asarray(seq_raw[t], dtype=np.float32)
            except Exception:
                continue
            if frame_arr.ndim != 2 or frame_arr.shape[-1] != 3:
                continue
            k = min(n_joints, frame_arr.shape[0])
            out[t, :k] = frame_arr[:k]
        return out
    return np.zeros((n_frames, n_joints, 3), dtype=np.float32)


def _fit_joint_dim(arr, n_frames, target_joints):
    if arr.ndim != 3 or arr.shape[0] != n_frames or arr.shape[2] != 3:
        raise ValueError(f"Unexpected entity shape: {arr.shape}")
    cur = arr.shape[1]
    if cur == target_joints:
        return arr.astype(np.float32)
    if cur > target_joints:
        return arr[:, :target_joints].astype(np.float32)
    pad = np.zeros((n_frames, target_joints - cur, 3), dtype=np.float32)
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
        max_idx = max(indices)
        if arr.shape[1] > max_idx:
            return _fit_joint_dim(arr[:, indices], n_frames, target_joints)
        if fallback is not None and len(fallback) > 0 and arr.shape[1] > max(fallback):
            return _fit_joint_dim(arr[:, fallback], n_frames, target_joints)
        if arr.shape[1] == target_joints:
            return arr.astype(np.float32)

    return _fit_joint_dim(arr, n_frames, target_joints)


def _entity_or_zeros_selected(seq_dict, key, n_frames, n_joints, joint_indices=None, fallback_indices=None):
    if key not in seq_dict:
        return np.zeros((n_frames, n_joints, 3), dtype=np.float32)
    # Some CoMad json files store ragged per-frame marker lists (variable marker
    # count across frames), which cannot be converted by one-shot np.asarray.
    # Parse frame-by-frame to stay robust.
    seq_raw = seq_dict[key]
    out = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
    usable_frames = min(n_frames, len(seq_raw))
    indices = _parse_joint_indices(joint_indices)
    fallback = _parse_joint_indices(fallback_indices)
    for t in range(usable_frames):
        try:
            frame_arr = np.asarray(seq_raw[t], dtype=np.float32)
        except Exception:
            continue
        if frame_arr.ndim != 2 or frame_arr.shape[-1] != 3:
            continue
        selected = frame_arr
        if indices is not None and len(indices) > 0:
            max_idx = max(indices)
            if frame_arr.shape[0] > max_idx:
                selected = frame_arr[indices]
            elif fallback is not None and len(fallback) > 0 and frame_arr.shape[0] > max(fallback):
                selected = frame_arr[fallback]
        k = min(n_joints, selected.shape[0])
        out[t, :k] = selected[:k]
    return out


class DatasetCoMad(Dataset):
    """
    Data loader for CoMad dataset.

    CoMad sequence json format can vary across samples.
    We normalize to configured fixed joint counts. The paper setup used by
    cfg/comad.yml is Person_1=(T, 9, 3), Robot=(T, 2, 3).
    """

    def __init__(
        self,
        mode,
        t_his=25,
        t_pred=100,
        actions="all",
        use_vel=False,
        data_path="./datasets/CoMad",
        include_person2=False,
        include_robot=True,
        use_data_aug=False,
        aug_rotate_prob=0.5,
        aug_reverse_prob=0.3,
        eval_interaction_filter=None,
        p1_joints=9,
        p2_joints=0,
        robot_joints=2,
        p1_joint_indices=COMAD_PAPER_HUMAN_JOINT_INDICES,
        p1_fallback_joint_indices=COMAD_PAPER_HUMAN_JOINT_INDICES_11,
        robot_joint_indices=COMAD_PAPER_ROBOT_JOINT_INDICES,
        robot_fallback_joint_indices=(8, 9),
    ):
        # Subset of {'HH', 'HR'} from path .../<action>/<HH|HR>/<id>/; train should pass None.
        self.eval_interaction_filter = eval_interaction_filter
        self.use_vel = use_vel
        self.data_path = data_path
        self.include_person2 = include_person2
        self.include_robot = include_robot
        self.actions_filter = actions
        self.use_data_aug = use_data_aug and mode == "train"
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_reverse_prob = aug_reverse_prob
        self.p1_joints = int(p1_joints)
        self.p2_joints = int(p2_joints)
        self.robot_joints = int(robot_joints)
        self.p1_joint_indices = _parse_joint_indices(p1_joint_indices)
        self.p1_fallback_joint_indices = _parse_joint_indices(p1_fallback_joint_indices)
        self.robot_joint_indices = _parse_joint_indices(robot_joint_indices)
        self.robot_fallback_joint_indices = _parse_joint_indices(robot_fallback_joint_indices)
        if self.p1_joints <= 0:
            raise ValueError(f"p1_joints must be positive, got {self.p1_joints}")
        if self.include_person2 and self.p2_joints <= 0:
            raise ValueError(f"p2_joints must be positive when include_person2=True, got {self.p2_joints}")
        if self.include_robot and self.robot_joints <= 0:
            raise ValueError(f"robot_joints must be positive when include_robot=True, got {self.robot_joints}")

        super().__init__(mode, t_his, t_pred, actions)

        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        # Person_1 / Person_2 topology for visualization metadata.
        if self.p1_joints == 9:
            p1_parents = [-1, 0, 0, 1, 2, 3, 4, 5, 6]
            p1_links = list(COMAD_HH_VIS_LINKS)
        elif self.p1_joints == 25:
            p1_parents = list(COMAD_P1_PARENTS)
            p1_links = comad_p1_links()
        elif self.p1_joints == 11:
            p1_parents = [-1, 0, 0, 2, 6, 6, 3, 3, 7, 8, 9]
            p1_links = list(COMAD_UPPER_BODY_11_VIS_LINKS)
        else:
            p1_parents = [-1] + list(range(self.p1_joints - 1))
            p1_links = [(j, p) for j, p in enumerate(p1_parents) if p != -1]

        if self.p2_joints == self.p1_joints:
            p2_parents = list(p1_parents)
            p2_links = list(p1_links)
        elif self.p2_joints == 25:
            p2_parents = list(COMAD_P1_PARENTS)
            p2_links = comad_p1_links()
        else:
            p2_parents = [-1] + list(range(max(self.p2_joints - 1, 0)))
            p2_links = [(j, p) for j, p in enumerate(p2_parents) if p != -1]
        # Robot: keep sequential chain (Franka-like); visualization-only topology.
        robot_parents = [-1] + list(range(self.robot_joints - 1))
        robot_links = [(j, p) for j, p in enumerate(robot_parents) if p != -1]

        all_parents = list(p1_parents)
        all_links = list(p1_links)
        self.num_p1_joints = self.p1_joints
        self.num_p2_joints = 0
        self.num_robot_joints = 0

        if self.include_person2:
            shift = len(all_parents)
            all_parents += [p + shift if p >= 0 else -1 for p in p2_parents]
            all_links += [(a + shift, b + shift) for a, b in p2_links]
            self.num_p2_joints = self.p2_joints

        if self.include_robot:
            shift = len(all_parents)
            all_parents += [p + shift if p >= 0 else -1 for p in robot_parents]
            all_links += [(a + shift, b + shift) for a, b in robot_links]
            self.num_robot_joints = self.robot_joints

        self.total_joints = len(all_parents)
        # Symmetric L/R lists for Skeleton (same length); arm coloring for P1/P2, coarse bands for robot.
        if self.p1_joints == 25:
            joints_left = list(COMAD_P1_JOINTS_LEFT)
            joints_right = list(COMAD_P1_JOINTS_RIGHT)
        else:
            joints_left = []
            joints_right = []
        if self.include_person2 and self.num_p2_joints == self.p1_joints and self.p1_joints == 25:
            shift = self.p1_joints
            joints_left.extend(j + shift for j in COMAD_P1_JOINTS_LEFT)
            joints_right.extend(j + shift for j in COMAD_P1_JOINTS_RIGHT)
        if self.include_robot and self.num_robot_joints > 0:
            rb0 = self.p1_joints + self.num_p2_joints
            half = self.num_robot_joints // 2
            joints_left.extend(range(rb0, rb0 + half))
            joints_right.extend(range(rb0 + half, rb0 + self.num_robot_joints))
        if len(joints_left) != len(joints_right):
            raise ValueError(f"CoMad L/R joint lists length mismatch: {len(joints_left)} vs {len(joints_right)}")

        self.skeleton = Skeleton(
            parents=all_parents,
            joints_left=joints_left,
            joints_right=joints_right,
            links=all_links,
        )

        self.kept_joints = np.arange(self.total_joints)
        self.removed_joints = set()
        self.process_data()

    def process_data(self):
        split_dir = os.path.join(self.data_path, self.mode)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"CoMad split folder {split_dir} not found. "
                f"Expected structure: {self.data_path}/train|test/<action>/<HH|HR>/<id>/data.json"
            )

        json_files = sorted(glob(os.path.join(split_dir, "*", "*", "*", "data.json")))
        if len(json_files) == 0:
            raise FileNotFoundError(f"No data.json files found under {split_dir}")

        self.data = {}
        self.subjects = []
        for json_file in json_files:
            rel = os.path.relpath(json_file, split_dir)
            parts = rel.split(os.sep)
            if len(parts) < 4:
                continue
            action, interaction, seq_id = parts[0], parts[1], parts[2]

            if self.eval_interaction_filter is not None and interaction not in self.eval_interaction_filter:
                continue

            if self.actions_filter != "all":
                if isinstance(self.actions_filter, str):
                    if action != self.actions_filter:
                        continue
                elif action not in self.actions_filter:
                    continue

            try:
                seq_dict = load_json(json_file)
            except Exception as e:
                print(f"[WARN] Skip malformed json: {json_file} ({e})")
                continue

            if "Person_1" not in seq_dict:
                print(f"[WARN] Skip file without Person_1: {json_file}")
                continue

            p1 = np.asarray(seq_dict["Person_1"], dtype=np.float32)
            if p1.ndim != 3 or p1.shape[2] != 3:
                print(f"[WARN] Skip file with invalid Person_1 shape {p1.shape}: {json_file}")
                continue
            n_frames = p1.shape[0]
            p1 = _fit_or_select_joint_dim(
                p1,
                n_frames,
                self.p1_joints,
                self.p1_joint_indices,
                self.p1_fallback_joint_indices,
            )
            entities = [p1]
            if self.include_person2:
                p2 = _entity_or_zeros(seq_dict, "Person_2", n_frames, self.p2_joints)
                p2 = _fit_joint_dim(p2, n_frames, self.p2_joints)
                entities.append(p2)
            if self.include_robot:
                rb = _entity_or_zeros_selected(
                    seq_dict,
                    "Robot",
                    n_frames,
                    self.robot_joints,
                    self.robot_joint_indices,
                    fallback_indices=self.robot_fallback_joint_indices,
                )
                entities.append(rb)

            seq = np.concatenate(entities, axis=1)
            if self.use_vel:
                v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                v = np.append(v, v[[-1]], axis=0)

            # Root-relative representation.
            seq[:, 1:] -= seq[:, :1]

            if self.use_vel:
                seq = np.concatenate((seq, v), axis=1)

            subject_key = interaction
            if subject_key not in self.data:
                self.data[subject_key] = {}

            action_key = f"{action}_{interaction}_{seq_id}"
            self.data[subject_key][action_key] = seq

        self.subjects = sorted(list(self.data.keys()))
        if len(self.subjects) == 0:
            raise RuntimeError(
                f"No valid CoMad sequences loaded for mode={self.mode}. "
                f"Check data path ({self.data_path}), action filter, and eval_interaction_filter "
                f"({self.eval_interaction_filter!r})."
            )

    def _apply_scene_rotation(self, sample):
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=sample.dtype,
        )
        return np.matmul(sample, rot.T)

    def _apply_sequence_reverse(self, sample):
        return sample[:, ::-1].copy()

    def augment_sample(self, sample):
        if np.random.uniform() < self.aug_rotate_prob:
            sample = self._apply_scene_rotation(sample)
        if np.random.uniform() < self.aug_reverse_prob:
            sample = self._apply_sequence_reverse(sample)
        return sample

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for _ in range(num_samples // batch_size):
            sample = []
            for _ in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug and self.use_data_aug:
                sample = self.augment_sample(sample)
            yield sample


if __name__ == "__main__":
    np.random.seed(0)
    dataset = DatasetCoMad("train", t_his=25, t_pred=100, data_path="./datasets/CoMad")
    print(f"Dataset loaded with {len(dataset.data)} interaction groups")
    for sub in dataset.data:
        print(f"  Group {sub}: {len(dataset.data[sub])} sequences")
    generator = dataset.sampling_generator()
    for data in generator:
        print(f"Sample shape: {data.shape}")
        break
