import numpy as np
from data_loader.dataset_chico import DatasetCHICO


class DatasetCHICO_multi(DatasetCHICO):
    """
    Multi-modal CHICO dataset for evaluation.
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
        **kwargs,
    ):
        self.multimodal_path = kwargs.get("multimodal_path", None)
        self.data_candi_path = kwargs.get("data_candi_path", None)
        super().__init__(
            mode=mode,
            t_his=t_his,
            t_pred=t_pred,
            actions=actions,
            use_vel=use_vel,
            data_path=data_path,
            include_robot=include_robot,
            train_subjects=kwargs.get("train_subjects", None),
            val_subjects=kwargs.get("val_subjects", None),
            test_subjects=kwargs.get("test_subjects", None),
            exclude_crash=kwargs.get("exclude_crash", True),
        )

    def sample(self, n_modality=5):
        """
        Keep API compatible with other multimodal datasets.
        Current implementation returns trajectory with optional placeholder multimodal target.
        """
        traj = super().sample()
        return traj, None

    def sampling_generator(self, num_samples=1000, batch_size=8, n_modality=5):
        for _ in range(num_samples // batch_size):
            sample = []
            sample_multi = []
            for _ in range(batch_size):
                sample_i, sample_multi_i = self.sample(n_modality=n_modality)
                sample.append(sample_i)
                if sample_multi_i is not None:
                    sample_multi.append(sample_multi_i[None, ...])
            sample = np.concatenate(sample, axis=0)
            if len(sample_multi) > 0:
                sample_multi = np.concatenate(sample_multi, axis=0)
            else:
                sample_multi = None
            yield sample, sample_multi

    def iter_generator(self, step=25, n_modality=10):
        for sub in self.data.keys():
            data_s = self.data[sub]
            for act in data_s.keys():
                seq = data_s[act]
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i : i + self.t_total]
                    yield traj, None


if __name__ == "__main__":
    np.random.seed(0)
    dataset = DatasetCHICO_multi("test", t_his=25, t_pred=100, data_path="./datasets/CHICO")
    print(f"Dataset loaded with {len(dataset.data)} subjects")
    for sub in dataset.data:
        print(f"  Subject {sub}: {len(dataset.data[sub])} sequences")
    gen = dataset.iter_generator()
    for traj, traj_multi in gen:
        print(f"Traj shape: {traj.shape}, multi: {None if traj_multi is None else traj_multi.shape}")
        break
