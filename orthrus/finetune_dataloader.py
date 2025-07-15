import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from typing import Union, List, Tuple

from orthrus.eval_utils import get_unpadded_seq_lens

from mrna_bench import load_dataset
from mrna_bench.datasets import DATASET_INFO
from mrna_bench.datasets.dataset_utils import str_to_ohe
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

def assemble_input(
    df: pd.DataFrame,
    n_tracks: int,
    subset_start: int | None = None,
    subset_end: int | None = None,
    channels_last: bool = True
) -> np.ndarray:
    """
    Build a padded numpy array of shape (N, max_len, n_tracks), i.e. channels-last.
      - The last dimension is the channel dimension.
      - If n_tracks=4, we store only the one-hot.
      - If n_tracks=6, we append 'cds' and 'splice' columns as the 5th and 6th channels.

    Steps:
      1. Determine the max sequence length (max_len) across the entire DataFrame.
      2. For each row:
         a) One-hot encode to shape (seq_len, 4).
         b) If n_tracks=6, append 'cds' & 'splice' as shape (seq_len, 1) each.
         c) Pad to (max_len, n_tracks).
      3. Stack to shape (N, max_len, n_tracks).
    """
    assert n_tracks in [4, 6], "We only support 4 or 6 channels."

    seqs_encoded = []
    seq_lens = []

    # Precompute max sequence length
    max_len = df["sequence"].str.len().max()

    if subset_start is not None or subset_end is not None:
        start = int(subset_start) if subset_start is not None else 0
        end = int(subset_end) if subset_end is not None else max_len

        if start < 0:
            start += max_len
        if end < 0:
            end += max_len

        if start >= end:
            raise ValueError(f"subset_start ({start}) must be < subset_end ({end}) after normalization")

        max_len = end - start

    print(f"Max sequence length in the dataset: {max_len}")

    for _, row in df.iterrows():
        # One-hot => shape (seq_len, 4)
        seq_oh = str_to_ohe(row["sequence"])

        if n_tracks == 4:
            # (seq_len, 4)
            sample_arr_unpadded = seq_oh
        else:
            # n_tracks=6 => 4 channels of one-hot + 1 channel of CDS + 1 channel of splice
            # Both 'cds' and 'splice' should each be shape (seq_len,)
            # We'll reshape => (seq_len, 1), then concatenate
            cds_track = row["cds"].astype(np.float32).reshape(-1, 1)    # (seq_len, 1)
            splice_track = row["splice"].astype(np.float32).reshape(-1, 1)  # (seq_len, 1)
            # Combine => (seq_len, 6)
            sample_arr_unpadded = np.concatenate([seq_oh, cds_track, splice_track], axis=1)

        # slice the unpadded array to the subset range
        sample_arr_unpadded = sample_arr_unpadded[subset_start:subset_end, :]

        seq_len = sample_arr_unpadded.shape[0]

        # Now we pad to (max_len, n_tracks)
        padded_sample = np.zeros((max_len, n_tracks), dtype=np.float32)
        padded_sample[:seq_len, :] = sample_arr_unpadded

        seqs_encoded.append(padded_sample)
        seq_lens.append(seq_len)

    # Final shape => (N, max_len, n_tracks)
    X = np.stack(seqs_encoded, axis=0)
    return X


class RNATaskDataset(Dataset):
    def __init__(
        self,
        data: Union[None, np.ndarray],
        seed: int,
        task: str,
        num_classes: int,
        subset_fraction: float = 1.0,
        subset_n_samples: Union[None, int] = None,
        channel_last: bool = False,
    ) -> None:
        """RNA task dataset for supervised training and evaluation.

        Args:
            data: Tuple of (X, y) where X is the input data and y is the target.
            seed: Random seed for reproducibility.
            task: The type of task (e.g., 'classification', 'regression').
            num_classes: The number of classes in the dataset.
            subset_fraction: Fraction of the dataset to use.
            subset_n_samples: Number of samples to use.
            channel_last: If True, data will be of shape (N, L, C).
        """
        self.seed = seed
        self.task = task
        self.num_classes = num_classes

        self.data = data
        self.X, self.y = data

        if subset_fraction < 1.0:
            self.subset_to_fraction(subset_fraction)
        elif subset_n_samples is not None:
            self.subset_to_n_samples(subset_n_samples)

        self.lengths = get_unpadded_seq_lens(self.X) # assumes channel last

        if not channel_last:
            self.X = np.transpose(self.X, (
                0, 2, 1
            ))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], self.lengths[idx]

    def subset_to_fraction(self, fraction: float) -> None:
        """Subset the dataset to a fraction of the original size."""
        if not (0 < fraction < 1):
            return

        n_samples = int(len(self.X) * fraction)
        self.subset_to_n_samples(n_samples)

    def subset_to_n_samples(self, n_samples: int) -> None:
        """Subset the dataset to a fixed number of samples."""
        if n_samples >= len(self.X):
            return

        np.random.seed(self.seed)
        
        if self.task in ["classification", "multilabel"]:
            i = 0
            while True:
                idx = np.random.choice(len(self.X), n_samples, replace=False)
                y_subset = self.y[idx]

                # For multilabel, we check that each class has at least one positive sample
                if self.task == "multilabel":
                    all_present = y_subset.sum(axis=0).min() > 0
                # For multiclass, we check that each class is present
                else:
                    all_present = len(np.unique(y_subset)) >= self.num_classes
                
                if all_present:
                    break
                i += 1
                if i > 1_000_000:
                    raise ValueError("Could not find a subset with all classes present after 1,000,000 tries.")
        else:
            idx = np.random.choice(len(self.X), n_samples, replace=False)

        self.X = self.X[idx]
        self.y = self.y[idx]
        self.data = self.X, self.y

def get_split_dataframes(data_config: dict, train_config: dict):
    """
    Get train, val, and test dataframes for the given dataset and split type.
    Args:
        data_config: Dictionary containing data configuration.
        train_config: Dictionary containing training configuration.
    Returns:
        results: List of tuples containing (seed, train_df, val_df, test_df).
    """
    dataset_name = data_config["dataset_name"]
    split_type = data_config["split_type"]
    data_df = load_dataset(DATASET_INFO[dataset_name]["dataset"]).data_df

    # drop rows with sequences longer than X nucleotides
    original_size = len(data_df)

    # used 50k for other datasets, but 15k for eCLIP

    if 'eclip' in dataset_name:
        drop_length = 15_000
    else:
        drop_length = 30_000

    data_df = data_df[data_df["sequence"].str.len() <= drop_length]
    data_df = data_df.reset_index(drop=True)

    print(f"Removed {original_size - len(data_df)} sequences longer than {drop_length} nucleotides")

    multi_seed = data_config.get("multi_seed", [train_config["rand_seed"]])

    if split_type == "homology":

        species = data_config["species"]
        splitter = SPLIT_CATALOG[split_type](species=species)

    elif split_type == "kmer":

        kmer = data_config.get("kmer", 3)
        n_clusters = data_config.get("n_clusters", None)

        splitter = SPLIT_CATALOG[split_type](
            kmer=data_config.get("kmer", 3),
            n_clusters=data_config.get("n_clusters", None),
        )
    elif split_type == "chromosome":
        splitter = SPLIT_CATALOG["chromosome"]()
    elif split_type == "default":
        splitter = SPLIT_CATALOG["default"]()
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    results = []
    for mseed in multi_seed:
        train_df, val_df, test_df = splitter.get_all_splits_df(
            df=data_df,
            split_ratios=(0.7, 0.15, 0.15),
            random_seed=mseed
        )
        results.append((mseed, train_df, val_df, test_df))
    return results

def get_data_loaders_for_target(
    target_col: str,
    task: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_config: dict,
    train_config: dict,
    seed: int,
):
    n_tracks = data_config["n_tracks"]
    subset_fraction = data_config["subset_fraction"]
    subset_n_samples = data_config["subset_n_samples"]
    batch_sizes = train_config["gpu_batch_sizes"]
    verbose = train_config.get("verbose", False)
    subset_start = data_config.get("subset_start", None)
    subset_end = data_config.get("subset_end", None)

    # Drop NaNs in target_col
    train_mask = ~pd.isnull(train_df[target_col])
    val_mask = ~pd.isnull(val_df[target_col])
    test_mask = ~pd.isnull(test_df[target_col])

    dropped_train = len(train_df) - train_mask.sum()
    dropped_val = len(val_df) - val_mask.sum()
    dropped_test = len(test_df) - test_mask.sum()

    train_df = train_df[train_mask]
    val_df = val_df[val_mask]
    test_df = test_df[test_mask]

    train_y = train_df[target_col].values
    val_y = val_df[target_col].values
    test_y = test_df[target_col].values

    train_X = assemble_input(
        train_df,
        n_tracks=n_tracks,
        subset_start=subset_start,
        subset_end=subset_end,
    )
    val_X = assemble_input(
        val_df,
        n_tracks=n_tracks,
        subset_start=subset_start,
        subset_end=subset_end,
    )
    test_X = assemble_input(
        test_df,
        n_tracks=n_tracks,
        subset_start=subset_start,
        subset_end=subset_end,
    )

    if task in ("regression", "reg_ridge"):
        train_y = train_y.astype(np.float32).reshape(-1, 1)
        val_y = val_y.astype(np.float32).reshape(-1, 1)
        test_y = test_y.astype(np.float32).reshape(-1, 1)
    elif task == "multilabel":
        train_y = np.vstack(train_y).astype(np.float32)
        val_y = np.vstack(val_y).astype(np.float32)
        test_y = np.vstack(test_y).astype(np.float32)
    elif task == "classification":
        dtype = np.float32 if data_config["num_classes"] == 1 else np.int64
        train_y = train_y.astype(dtype).reshape(-1, 1)
        val_y = val_y.astype(dtype).reshape(-1, 1)
        test_y = test_y.astype(dtype).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown task type: {task}")

    if verbose:
        print(f"\tDropped {dropped_train} NaN values from train set")
        print(f"\tDropped {dropped_val} NaN values from val set")
        print(f"\tDropped {dropped_test} NaN values from test set")
        print(f"\tTrain X shape: {train_X.shape}, Train y shape: {train_y.shape}")
        print(f"\tVal X shape: {val_X.shape}, Val y shape: {val_y.shape}")
        print(f"\tTest X shape: {test_X.shape}, Test y shape: {test_y.shape}\n")

    if subset_fraction < 1.0:
        n_samples = int(len(train_X) * subset_fraction)
        if n_samples < batch_sizes[0]:
            raise ValueError("subset_fraction is too small for the given train_batch_size.")
    if subset_n_samples is not None:
        if subset_n_samples < batch_sizes[0]:
            raise ValueError("subset_n_samples is too small for the given train_batch_size.")

    train_dataset = RNATaskDataset((train_X, train_y), seed=seed,
                                   task=task,
                                   num_classes=data_config["num_classes"],
                                   subset_fraction=subset_fraction,
                                   subset_n_samples=subset_n_samples)
    val_dataset = RNATaskDataset((val_X, val_y), seed=seed,
                                   task=task,
                                   num_classes=data_config["num_classes"])
    test_dataset = RNATaskDataset((test_X, test_y), seed=seed,
                                   task=task,
                                   num_classes=data_config["num_classes"])

    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[1], shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader