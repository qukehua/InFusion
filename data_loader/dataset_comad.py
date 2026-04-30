import os
import json
from glob import glob
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


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


class DatasetCoMad(Dataset):
    """
    Data loader for CoMad dataset.

    CoMad sequence json format can vary across samples.
    We normalize to fixed joint counts:
      - Person_1: (T, 25, 3)
      - Person_2: (T, 25, 3)
      - Robot:    (T, 12, 3)
    """

    def __init__(
        self,
        mode,
        t_his=25,
        t_pred=100,
        actions="all",
        use_vel=False,
        data_path="./datasets/CoMad",
        include_person2=True,
        include_robot=True,
        use_data_aug=False,
        aug_rotate_prob=0.5,
        aug_reverse_prob=0.3,
    ):
        self.use_vel = use_vel
        self.data_path = data_path
        self.include_person2 = include_person2
        self.include_robot = include_robot
        self.actions_filter = actions
        self.use_data_aug = use_data_aug and mode == "train"
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_reverse_prob = aug_reverse_prob

        super().__init__(mode, t_his, t_pred, actions)

        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        # Fixed joint layout used for model input.
        self.p1_joints = 25
        self.p2_joints = 25
        self.robot_joints = 12

        # Minimal tree-compatible definitions (chains) for each entity.
        p1_parents = [-1] + list(range(self.p1_joints - 1))
        p2_parents = [-1] + list(range(self.p2_joints - 1))
        robot_parents = [-1] + list(range(self.robot_joints - 1))

        p1_links = [(j, p) for j, p in enumerate(p1_parents) if p != -1]
        p2_links = [(j, p) for j, p in enumerate(p2_parents) if p != -1]
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
        joints_left = [5, 6, 7]
        joints_right = [2, 3, 4]

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
            p1 = _fit_joint_dim(p1, n_frames, self.p1_joints)
            entities = [p1]
            if self.include_person2:
                p2 = _entity_or_zeros(seq_dict, "Person_2", n_frames, self.p2_joints)
                p2 = _fit_joint_dim(p2, n_frames, self.p2_joints)
                entities.append(p2)
            if self.include_robot:
                rb = _entity_or_zeros(seq_dict, "Robot", n_frames, self.robot_joints)
                rb = _fit_joint_dim(rb, n_frames, self.robot_joints)
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
                f"Check data path ({self.data_path}) and action filter."
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
