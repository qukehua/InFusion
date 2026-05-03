import os
from glob import glob
import pickle as pkl
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


class DatasetCHICO(Dataset):
    """
    Data loader for CHICO dataset.
    CHICO pkl frame format:
      frame = [human_joints_3d(15, 3), robot_joints_3d(9, 3)]
    """

    def __init__(
        self,
        mode,
        t_his=25,
        t_pred=100,
        actions="all",
        use_vel=False,
        data_path="./datasets/CHICO",
        include_robot=True,
        train_subjects=None,
        val_subjects=None,
        test_subjects=None,
        exclude_crash=True,
    ):
        self.use_vel = use_vel
        self.data_path = data_path
        self.include_robot = include_robot
        self.actions_filter = actions
        self.exclude_crash = exclude_crash
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects

        super().__init__(mode, t_his, t_pred, actions)

        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        # CHICO 15-joint layout and edges match the authors' visualization utils:
        # https://github.com/federicocunico/human-robot-collaboration/blob/master/datasets/chico_dataset.py
        # Index: 0 hip, 1 r_hip, 2 r_knee, 3 r_foot, 4 l_hip, 5 l_knee, 6 l_foot,
        #        7 nose, 8 c_shoulder, 9 r_shoulder, 10 r_elbow, 11 r_wrist,
        #        12 l_shoulder, 13 l_elbow, 14 l_wrist
        # The published graph is not a tree (torso uses diagonals 1–9 and 4–12); we pass explicit links for drawing.
        # (j, j_parent): first index is used for left/right bone color in utils/visualization.py
        human_links = [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 0),
            (5, 4),
            (6, 5),
            (9, 1),
            (12, 4),
            (7, 8),
            (9, 8),
            (12, 8),
            (10, 9),
            (11, 10),
            (13, 12),
            (14, 13),
        ]
        # Spanning tree compatible with the same indices (for Skeleton.parents / metadata only).
        human_parents = [-1, 0, 1, 2, 0, 4, 5, 8, 0, 8, 9, 10, 8, 12, 13]

        if self.include_robot:
            # Robot/tool joint graph as a chain (same as reference KUKA links, 9 joints).
            robot_parents = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
            robot_links = [(j, p) for j, p in enumerate(robot_parents) if p != -1]
            all_parents = human_parents + [p + 15 if p >= 0 else -1 for p in robot_parents]
            all_links = list(human_links) + [(a + 15, b + 15) for a, b in robot_links]
            self.num_human_joints = 15
            self.num_robot_joints = 9
            self.total_joints = 24
            joints_left = [4, 5, 6, 12, 13, 14]
            joints_right = [1, 2, 3, 9, 10, 11]
        else:
            all_parents = human_parents
            all_links = list(human_links)
            self.num_human_joints = 15
            self.num_robot_joints = 0
            self.total_joints = 15
            joints_left = [4, 5, 6, 12, 13, 14]
            joints_right = [1, 2, 3, 9, 10, 11]

        self.skeleton = Skeleton(
            parents=all_parents,
            joints_left=joints_left,
            joints_right=joints_right,
            links=all_links,
        )

        self.kept_joints = np.arange(self.total_joints)
        self.removed_joints = set()

        dataset_dir = os.path.join(self.data_path, "dataset")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(
                f"CHICO dataset folder {dataset_dir} not found. "
                f"Expected structure: {self.data_path}/dataset/Sxx/*.pkl"
            )

        all_subjects = sorted(
            [d for d in os.listdir(dataset_dir) if d.startswith("S") and os.path.isdir(os.path.join(dataset_dir, d))]
        )
        if len(all_subjects) == 0:
            raise FileNotFoundError(f"No subject folders found in {dataset_dir}")

        if self.train_subjects is None or self.val_subjects is None or self.test_subjects is None:
            # CHICO paper protocol:
            # val:  subjects 0 and 4
            # test: subjects 2, 3, 18 and 19
            val_default = ["S00", "S04"]
            test_default = ["S02", "S03", "S18", "S19"]
            val_subjects = [s for s in val_default if s in all_subjects]
            test_subjects = [s for s in test_default if s in all_subjects]
            train_subjects = [s for s in all_subjects if s not in set(val_subjects + test_subjects)]
            self.subjects_split = {
                "train": train_subjects,
                "val": val_subjects,
                "test": test_subjects,
            }
        else:
            self.subjects_split = {
                "train": self.train_subjects,
                "val": self.val_subjects,
                "test": self.test_subjects,
            }

        if self.mode not in self.subjects_split:
            raise ValueError(
                f"Unsupported mode '{self.mode}'. "
                f"Available modes: {list(self.subjects_split.keys())}"
            )
        self.subjects = self.subjects_split[self.mode]
        self.process_data()

    def process_data(self):
        self.data = {}
        for subject in self.subjects:
            subject_dir = os.path.join(self.data_path, "dataset", subject)
            if not os.path.exists(subject_dir):
                continue

            pkl_files = sorted(glob(os.path.join(subject_dir, "*.pkl")))
            if len(pkl_files) == 0:
                continue

            self.data[subject] = {}
            for pkl_file in pkl_files:
                action = os.path.splitext(os.path.basename(pkl_file))[0]
                if self.exclude_crash and "_CRASH" in action:
                    continue
                if self.actions_filter != "all":
                    if isinstance(self.actions_filter, str):
                        if action != self.actions_filter:
                            continue
                    elif action not in self.actions_filter:
                        continue

                sequence_list = load_pkl(pkl_file)
                if len(sequence_list) == 0:
                    continue

                human_seq = np.asarray([f[0] for f in sequence_list], dtype=np.float32)
                if self.include_robot:
                    robot_seq = np.asarray([f[1] for f in sequence_list], dtype=np.float32)
                    seq = np.concatenate([human_seq, robot_seq], axis=1)
                else:
                    seq = human_seq

                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)

                # Root-relative body: non-root human joints and robot become offsets from human pelvis (joint 0).
                # This preserves inter-joint distances (bone lengths) in each frame; it does not by itself
                # destroy kinematic structure—that comes from unconstrained diffusion outputs.
                seq[:, 1:] -= seq[:, :1]
                # Pelvis at origin every frame (remove global translation from the stored tensor).
                seq[:, :1, :] = 0

                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)

                action_key = action
                counter = 1
                while action_key in self.data[subject]:
                    action_key = f"{action}_{counter}"
                    counter += 1
                self.data[subject][action_key] = seq

            if len(self.data[subject]) == 0:
                self.data.pop(subject)

        self.subjects = sorted(list(self.data.keys()))
        if len(self.subjects) == 0:
            raise RuntimeError(
                f"No valid CHICO sequences loaded for mode={self.mode}. "
                f"Check data path ({self.data_path}) and subject split."
            )


def gen_velocity(m):
    dm = np.zeros_like(m)
    dm[:, 1:] = m[:, 1:] - m[:, :-1]
    dm[:, 0] = dm[:, 1]
    return dm


if __name__ == "__main__":
    np.random.seed(0)
    dataset = DatasetCHICO("train", t_his=25, t_pred=100, data_path="./datasets/CHICO")
    print(f"Dataset loaded with {len(dataset.data)} subjects")
    for sub in dataset.data:
        print(f"  Subject {sub}: {len(dataset.data[sub])} sequences")
    generator = dataset.sampling_generator()
    for data in generator:
        print(f"Sample shape: {data.shape}")
        break
