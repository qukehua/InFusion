import numpy as np


class Dataset:
    def __init__(self, mode, t_his, t_pred, actions='all'):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()
        self.std, self.mean = None, None
        self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        self.sample_ind = None

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / self.std
        self.normalized = True

    def sample(self):
        # Some datasets may contain sequences shorter than (t_his + t_pred).
        # If we sample them blindly, `np.random.randint(seq_len - t_total)` can crash with:
        # ValueError: high <= 0
        #
        # So we resample until we find a sequence with enough length.
        for _ in range(200):
            subject = np.random.choice(self.subjects)
            dict_s = self.data[subject]
            action = np.random.choice(list(dict_s.keys()))
            seq = dict_s[action]
            seq_len = seq.shape[0]
            if seq_len > self.t_total:
                fr_start = np.random.randint(seq_len - self.t_total)
                fr_end = fr_start + self.t_total
                traj = seq[fr_start: fr_end]
                return traj[None, ...]

        # If we reach here, it means no (or almost no) sequences are long enough.
        # Provide a helpful error message.
        min_len = None
        for subject in self.subjects:
            for _, seq in self.data[subject].items():
                l = seq.shape[0]
                min_len = l if min_len is None else min(min_len, l)
        raise ValueError(
            f"No sequences long enough for sampling: t_total={self.t_total}, "
            f"min_seq_len={min_len}. Please verify dataset FPS / preprocessing / t_his+t_pred."
        )
    
    def sample_all_action(self):
        dict_s = self.data['S9']

        action_list = []
        sample = []

        for i in range(0, len(list(dict_s.keys()))):
            type = list(dict_s.keys())[i].split(' ')[0]
            if type == 'Discussion':
                type = 'Discussion 1'
            action_list.append(type)

        action_list = list(set(action_list))
        
        for i in range(0, len(action_list)):
            action = action_list[i]
            seq = dict_s[action]
            fr_start = np.random.randint(seq.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            traj = seq[fr_start: fr_end]
            sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample
    
    def sample_iter_action(self, action_category, dataset_type):
        if dataset_type == 'harper3d':
            # HARPER stores data as self.data[subject][action_key], where action_key
            # may include suffixes like "act1_0_1" for uniqueness.
            candidate = []
            for subject, dict_s_sub in self.data.items():
                for action_key, seq in dict_s_sub.items():
                    if action_key == action_category or action_key.startswith(action_category):
                        if seq.shape[0] > self.t_total:
                            candidate.append(seq)
            if len(candidate) == 0:
                raise ValueError(
                    f"No HARPER sequence found for action '{action_category}' with length > t_total ({self.t_total})."
                )
            seq = candidate[np.random.randint(len(candidate))]
            fr_start = np.random.randint(seq.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            traj = seq[fr_start: fr_end]
            return traj[None, ...]
        elif dataset_type == 'chico':
            candidate = []
            for _, dict_s_sub in self.data.items():
                for action_key, seq in dict_s_sub.items():
                    if action_key == action_category or action_key.startswith(action_category):
                        if seq.shape[0] > self.t_total:
                            candidate.append(seq)
            if len(candidate) == 0:
                raise ValueError(
                    f"No CHICO sequence found for action '{action_category}' with length > t_total ({self.t_total})."
                )
            seq = candidate[np.random.randint(len(candidate))]
            fr_start = np.random.randint(seq.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            traj = seq[fr_start: fr_end]
            return traj[None, ...]
        else:
            raise ValueError(f"Unsupported dataset_type '{dataset_type}' in sample_iter_action.")
    
    def prepare_iter_action(self, dataset_type):
        if dataset_type == 'harper3d':
            # Collect action names across all subjects; strip duplicate suffixes.
            action_list = []
            for _, dict_s_sub in self.data.items():
                for action_key in dict_s_sub.keys():
                    base_action = action_key.rsplit('_', 1)[0]
                    action_list.append(base_action)
            return sorted(list(set(action_list)))
        elif dataset_type == 'chico':
            action_list = []
            for _, dict_s_sub in self.data.items():
                for action_key in dict_s_sub.keys():
                    base_action = action_key.rsplit('_', 1)[0]
                    action_list.append(base_action)
            return sorted(list(set(action_list)))
        else:
            raise ValueError(f"Unsupported dataset_type '{dataset_type}' in prepare_iter_action.")

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug is True:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(sample.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    sample[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    sample[..., 0] = - sample[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    sample[..., 1] = - sample[..., 1]
            yield sample

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj
