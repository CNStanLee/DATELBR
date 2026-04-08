import numpy as np
import torch
from torch.utils.data import Dataset


class RealHarmonicDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        split: str = "train",  # "train", "val", "test"
        cycle: str = "half",  # "half", "quarter", "full"
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        assert split in ("train", "val", "test")
        assert cycle in ("half", "quarter", "full")
        assert 0.0 < train_ratio < 1.0
        assert 0.0 <= val_ratio < 1.0
        assert train_ratio + val_ratio < 1.0

        data = np.load(npz_path, allow_pickle=True)
        signals = data["signals"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
        group_ids = data["group_ids"].astype(np.int64)

        if signals.ndim != 2:
            raise ValueError(f"signals should be 2D, got {signals.shape}")
        if labels.ndim != 2:
            raise ValueError(f"labels should be 2D, got {labels.shape}")
        if signals.shape[0] != labels.shape[0] or signals.shape[0] != group_ids.shape[0]:
            raise ValueError(
                f"mismatched sample dims: signals={signals.shape}, labels={labels.shape}, group_ids={group_ids.shape}"
            )

        num_samples, samples_per_cycle = signals.shape
        if num_samples == 0:
            raise ValueError(f"Empty dataset: {npz_path}")

        if cycle == "quarter":
            input_len = samples_per_cycle // 4
        elif cycle == "half":
            input_len = samples_per_cycle // 2
        else:
            input_len = samples_per_cycle

        signals = signals[:, :input_len]

        max_abs = np.max(np.abs(signals))
        if max_abs > 0:
            signals = signals / max_abs

        label_scale = 100.0
        labels = labels / label_scale

        unique_groups = np.unique(group_ids)
        rng = np.random.default_rng(split_seed)
        groups_shuffled = unique_groups.copy()
        rng.shuffle(groups_shuffled)

        num_groups = len(groups_shuffled)
        num_train = int(num_groups * train_ratio)
        num_val = int(num_groups * val_ratio)

        # Keep split robust on small datasets.
        if num_groups >= 1 and num_train == 0:
            num_train = 1
        if num_train + num_val >= num_groups and num_groups >= 2:
            num_val = max(0, num_groups - num_train - 1)

        train_groups = groups_shuffled[:num_train]
        val_groups = groups_shuffled[num_train : num_train + num_val]
        test_groups = groups_shuffled[num_train + num_val :]

        if split == "train":
            sel_groups = train_groups
        elif split == "val":
            sel_groups = val_groups
        else:
            sel_groups = test_groups

        sample_mask = np.isin(group_ids, sel_groups)
        self.signals = signals[sample_mask]
        self.labels = labels[sample_mask]
        self.group_ids = group_ids[sample_mask]

        self.input_len = input_len
        self.label_scale = label_scale
        self.split = split
        self.cycle = cycle
        self.n_total_groups = num_groups
        self.n_split_groups = len(np.unique(self.group_ids))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def create_real_harmonic_datasets(
    npz_path: str,
    cycle: str = "half",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_seed: int = 42,
):
    train_ds = RealHarmonicDataset(
        npz_path=npz_path,
        split="train",
        cycle=cycle,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    val_ds = RealHarmonicDataset(
        npz_path=npz_path,
        split="val",
        cycle=cycle,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    test_ds = RealHarmonicDataset(
        npz_path=npz_path,
        split="test",
        cycle=cycle,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    return train_ds, val_ds, test_ds
