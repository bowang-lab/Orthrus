import os


import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Union

from orthrus.util import train_test_split_homologous, load_homology_df
from orthrus.eval_utils import get_unpadded_seq_lens


class RNATaskDataset(Dataset):
    def __init__(
        self,
        data: Union[None, np.ndarray],
        data_dir: str,
        dataset_name: str,
        split: str,
        split_type: str,
        seed: int,
        n_tracks: int,
        verbose: int,
        species: str,
        subset_fraction: float = 1.0,
        subset_n_samples: Union[None, int] = None,
    ) -> None:
        """RNA task dataset for training and evaluation.

        Args:
            data: can provide the numpy df directly without loading from disk.
            data_dir: Directory containing the data.
            dataset_name: Name of the dataset.
            split: Split of the data.
            split_type: Type of split.
            seed: Random seed for reproducibility.
            n_tracks: Number of tracks.
            verbose: Verbosity level.
            species: Species for homology split.
            subset_fraction: Fraction of the dataset to use.
            subset_n_samples: Number of samples to use.
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.split_type = split_type
        self.seed = seed
        self.n_tracks = n_tracks
        self.verbose = verbose
        self.species = species

        assert split in ["train", "val", "test"]

        if data is None:
            data = self.load_data(data_dir, dataset_name)
            train_X, train_y, val_X, val_y, test_X, test_y = self.split_data(
                data, split_type, seed, n_tracks, species, verbose
            )
            if split == "train":
                self.data = train_X, train_y
                self.X = train_X
                self.y = train_y
            elif split == "val":
                self.data = val_X, val_y
                self.X = val_X
                self.y = val_y
            elif split == "test":
                self.data = test_X, test_y
                self.X = test_X
                self.y = test_y

        elif data is not None:
            self.data = data
            self.X, self.y = data

        if subset_fraction < 1.0:
            self.subset_to_fraction(subset_fraction)
        elif subset_n_samples is not None:
            self.subset_to_n_samples(subset_n_samples)

        if verbose > 0:
            print(f"Loaded {split} data: {self.X.shape}, {self.y.shape}")

        self.lengths = get_unpadded_seq_lens(self.X, channels_last=False)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], self.lengths[idx]

    def subset_to_fraction(self, fraction: float) -> None:
        """Subset the dataset to a fraction of the original size."""
        assert 0 < fraction < 1
        n_samples = int(len(self.X) * fraction)
        # make sure fraction depends on seed
        np.random.seed(self.seed)
        idx = np.random.choice(len(self.X), n_samples, replace=False)
        all_class_present = False

        i = 0
        while not all_class_present:
            i+=1
            all_class_present = self.y[idx].sum(axis=0).min() > 0
            if not all_class_present:
                idx = np.random.choice(len(self.X), n_samples, replace=False)                
                
            if i > 10_000_000:
                raise ValueError("Could not find a subset with all classes present")
            
        self.X = self.X[idx]
        self.y = self.y[idx]
        self.data = self.X, self.y

    def subset_to_n_samples(self, n_samples: int) -> None:
        """Subset the dataset to a fixed number of samples."""
        assert n_samples < len(self.X)
        np.random.seed(self.seed)
        idx = np.random.choice(len(self.X), n_samples, replace=False)
        
        all_class_present = False
        i = 0
        while not all_class_present:
            i+=1
            all_class_present = self.y[idx].sum(axis=0).min() > 0
            if not all_class_present:
                idx = np.random.choice(len(self.X), n_samples, replace=False)                
                
            if i > 10_000_000:
                raise ValueError("Could not find a subset with all classes present")

        self.X = self.X[idx]
        self.y = self.y[idx]
        self.data = self.X, self.y

    @classmethod
    def load_data(cls, data_dir: str, dataset_name: str) -> np.ndarray:
        """Load the dataset from the data directory."""
        dataset_path = os.path.join(data_dir, dataset_name)
        data = np.load(dataset_path)
        return data

    @classmethod
    def convert_seq_to_one_hot(cls, seq):
        seq = seq.upper()
        one_hot = np.zeros((len(seq), 4))
        for i, char in enumerate(seq):
            if char == 'A':
                one_hot[i, 0] = 1
            elif char == 'C':
                one_hot[i, 1] = 1
            elif char == 'G':
                one_hot[i, 2] = 1
            elif char == 'T':
                one_hot[i, 3] = 1
            elif char == 'U':
                one_hot[i, 3] = 1
        return one_hot

    @classmethod
    def split_data(
        cls,
        data: np.ndarray,
        split_type: str,
        seed: int,
        n_tracks: int,
        species: str,
        verbose: int,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> tuple:
        """Split the data into train, validation, and test sets."""
        assert split_type in ["random", "homology"]
        assert n_tracks in [4, 6]

        if n_tracks == 4:
            X = data["X"][:, :, :4].astype(np.float32)
        else:
            X = data["X"].astype(np.float32)

        assert X.ndim == 3

        # Channels last is false
        if X.shape[2] in [4, 6]:
            X = X.transpose(0, 2, 1)

        y = data["y"].astype(np.float32)

        np.random.seed(seed)
        first_split_size = test_size + val_size
        assert first_split_size < 1.0
        second_split_size = val_size / first_split_size

        if split_type == "random":
            first_split_size = test_size + val_size
            assert first_split_size < 1.0

            train_X, vt_X, train_y, vt_y = train_test_split(
                X, y, test_size=first_split_size
            )
            val_X, test_X, val_y, test_y = train_test_split(
                vt_X, vt_y, test_size=second_split_size
            )

            if verbose > 1:
                print("Random split:")
                print(
                    f"Train: {train_X.shape}, "
                    f"Val: {val_X.shape}, "
                    f"Test: {test_X.shape}"
                )

        elif split_type == "homology":
            gene_names = data["genes"]
            homology_df = load_homology_df(species)

            split = train_test_split_homologous(
                gene_names, homology_df, test_size=first_split_size, random_state=seed
            )

            train_X = X[split["train_indices"]]
            train_y = y[split["train_indices"]]
            val_X = X[split["test_indices"]]
            val_y = y[split["test_indices"]]
            val_genes = gene_names[split["test_indices"]]

            # Second split
            split = train_test_split_homologous(
                val_genes, homology_df, test_size=second_split_size, random_state=seed
            )
            test_X = val_X[split["test_indices"]]
            test_y = val_y[split["test_indices"]]

            val_X = val_X[split["train_indices"]]
            val_y = val_y[split["train_indices"]]
            val_genes = val_genes[split["train_indices"]]
            if verbose > 1:
                print("Homology split:")
                print(
                    f"Train: {train_X.shape},"
                    f"Val: {val_X.shape},"
                    f"Test: {test_X.shape}"
                )
        else:
            raise ValueError(f"Invalid split type: {split_type}")

        return train_X, train_y, val_X, val_y, test_X, test_y


def create_data_loader(dataset, batch_size, shuffle, num_workers, drop_last=True):
    """Create a data loader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def create_dataloaders(
    split_type,
    dataset_name,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    species,
    n_tracks,
    seed,
    data_dir="/fs01/projects/isoclr/linear_probe_data2",
    verbose=0,
    subset_fraction=1.0,
    subset_n_samples=None,
):
    data = RNATaskDataset.load_data(data_dir=data_dir, dataset_name=dataset_name)
    train_X, train_y, val_X, val_y, test_X, test_y = RNATaskDataset.split_data(
        data,
        seed=seed,
        split_type=split_type,
        n_tracks=n_tracks,
        species=species,
        verbose=verbose,
    )

    # Make sure that the subset fraction is not too small since last batch gets dropped
    if subset_fraction < 1.0:
        n_samples = int(len(train_X) * subset_fraction)
        assert n_samples > train_batch_size
    if subset_n_samples is not None:
        assert subset_n_samples > train_batch_size

    train = RNATaskDataset(
        data=(train_X, train_y),
        split_type=split_type,
        data_dir=data_dir,
        dataset_name=dataset_name,
        species=species,
        n_tracks=n_tracks,
        split="train",
        seed=seed,
        verbose=verbose,
        subset_fraction=subset_fraction,
        subset_n_samples=subset_n_samples,
    )
    val = RNATaskDataset(
        data=(val_X, val_y),
        split_type=split_type,
        data_dir=data_dir,
        dataset_name=dataset_name,
        species=species,
        n_tracks=n_tracks,
        split="val",
        seed=seed,
        verbose=verbose,
    )
    test = RNATaskDataset(
        data=(test_X, test_y),
        split_type=split_type,
        data_dir=data_dir,
        dataset_name=dataset_name,
        species=species,
        n_tracks=n_tracks,
        split="test",
        seed=seed,
        verbose=verbose,
    )
    train_loader = create_data_loader(
        train, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = create_data_loader(
        val, batch_size=val_batch_size, shuffle=False, num_workers=4, drop_last=False
    )
    test_loader = create_data_loader(
        test, batch_size=test_batch_size, shuffle=False, num_workers=4, drop_last=False
    )
    return train_loader, val_loader, test_loader
