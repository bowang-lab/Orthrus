import pickle
import sys

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from orthrus.data import Transcript
from orthrus.util import split_dict

sys.path.append("../data_prep/")
from data_utils import get_transcript_path


class TranscriptTranscriptDataset(Dataset):
    """A PyTorch Dataset that operates over Transcript objects."""

    def __init__(
        self,
        transcript_mapping: dict[str, list[str]],
        npz_save_dir: str,
        mask_percentage: float = 0.0,
    ) -> None:
        """Initialize a transcript dataset.

        Args:
            transcript_mapping: Map of transcript to associated transcripts.
            npz_save_dir: Path where processed transcripts are stored.
            mask_percentage: Percentage of tokens to randomly mask.
        """
        self.transcript_mapping = transcript_mapping
        self.npz_save_dir = npz_save_dir
        self.mask_percentage = mask_percentage

        self.index_to_transcript = {}
        for i, transcript in enumerate(self.transcript_mapping.keys()):
            self.index_to_transcript[i] = transcript

    def load_transcript(self, transcript_id: str) -> np.ndarray:
        """Load transcript from disk.

        Assumes data has shape (C x L), i.e., channel is not last.

        Args:
            transcript_id: ID of transcript to load.

        Returns:
            Sequence data associated with transcript.
        """
        t_path_dirs = get_transcript_path(transcript_id)
        t_path = "{}/{}/{}.npz".format(self.npz_save_dir, t_path_dirs, transcript_id)

        t_data = np.load(t_path)["six_track_data"]

        if self.mask_percentage > 0:
            num_elements_to_mask = int(t_data.shape[1] * self.mask_percentage)
            mask_indices = np.random.choice(
                t_data.shape[1], num_elements_to_mask, replace=False
            )
            t_data[:, mask_indices] = 0

        return t_data

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        :return: Number of items in dataset
        """
        return len(self.transcript_mapping)

    def __getitem__(self, idx: int) -> tuple[Transcript, Transcript, int, int]:
        """Get Transcript object and a randomly sampled associated Transcript.

        :param idx: The index of the item.
        :return: A tuple containing the one-hot encoded RNA sequences.
        """
        base_transcript_id = self.index_to_transcript[idx]
        associated_transcripts = self.transcript_mapping[base_transcript_id]
        associated_transcripts = list(associated_transcripts)

        # Cannot be empty
        if len(associated_transcripts) > 0:
            pair_transcript_idx = np.random.choice(len(associated_transcripts))
            pair_transcript_id = associated_transcripts[pair_transcript_idx]
        else:
            pair_transcript_id = base_transcript_id

        encoded_base_transcript = self.load_transcript(base_transcript_id)
        encoded_pair_transcript = self.load_transcript(pair_transcript_id)

        len_t1 = encoded_base_transcript.shape[1]
        len_t2 = encoded_pair_transcript.shape[1]

        return encoded_base_transcript, encoded_pair_transcript, len_t1, len_t2


class WeightedTranscriptTranscriptDataset(TranscriptTranscriptDataset):

    def __init__(
        self,
        transcript_mapping: dict[str, list[str]],
        npz_save_dir: str,
        mask_percentage: float = 0,
        weight: float = 0.5,
    ) -> None:
        super().__init__(transcript_mapping, npz_save_dir, mask_percentage)
        self.weight = weight

    def __getitem__(self, idx: int) -> tuple[Transcript, Transcript, int, int]:
        """Get Transcript object and a randomly sampled associated Transcript.

        :param idx: The index of the item.
        :return: A tuple containing the one-hot encoded RNA sequences.
        """
        base_transcript_id = self.index_to_transcript[idx]
        associated_transcripts = self.transcript_mapping[base_transcript_id]
        associated_transcripts = list(associated_transcripts)

        def is_ortho(t_id):
            return len(t_id.split("_")) == 3

        splice = [t for t in associated_transcripts if not is_ortho(t)]
        ortho = [t for t in associated_transcripts if is_ortho(t)]

        if len(splice) == 0 and len(ortho) == 0:
            pair_transcript_id = base_transcript_id
        elif len(splice) == 0 or len(ortho) == 0:
            pair_transcript_idx = np.random.choice(len(associated_transcripts))
            pair_transcript_id = associated_transcripts[pair_transcript_idx]
        else:
            if np.random.random() > self.weight:
                pair_transcript_idx = np.random.choice(len(splice))
                pair_transcript_id = splice[pair_transcript_idx]
            else:
                pair_transcript_idx = np.random.choice(len(ortho))
                pair_transcript_id = ortho[pair_transcript_idx]

        encoded_base_transcript = self.load_transcript(base_transcript_id)
        encoded_pair_transcript = self.load_transcript(pair_transcript_id)

        len_t1 = encoded_base_transcript.shape[1]
        len_t2 = encoded_pair_transcript.shape[1]

        return encoded_base_transcript, encoded_pair_transcript, len_t1, len_t2


def collate_fn(
    batch: list[tuple[np.array, np.array]],
    mixed_precison=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcript encoding collation function for DataLoader.

    This function dynamically pads the batch sequences based on the longest
    sequence in the batch.

    :param batch:
        List of tuples (encoded_transcript1, encoded_transcript2) as returned
        by __getitem__. Each sequence has shape (C, L) where C is channel and
        L is length.
    :param mixed_precision: Uses 16-bit float when possible.
    :return:
        Two tensors representing padded sequences of transcript1 and
        transcript2 in the batch. Each tensor has shape (B, C, L) where
        B is batch size, C is channel and L is length.
    """
    # Find the longest sequence in the batch for both transcripts
    # Assumes sequences are (C, L) where C is channel and L is length
    max_length1 = max([seq1.shape[1] for seq1, _, _, _ in batch])
    max_length2 = max([seq2.shape[1] for _, seq2, _, _ in batch])

    # Determine the overall max length for padding
    max_length = max(max_length1, max_length2)

    # Pad sequences and prepare the batch
    padded_batch1 = []
    padded_batch2 = []
    # conditional on mixed precision set to float16 or 32
    if mixed_precison:
        precision = torch.float16
    else:
        precision = torch.float32

    t_lengths1 = torch.zeros(len(batch), dtype=torch.int32)
    t_lengths2 = torch.zeros(len(batch), dtype=torch.int32)

    for i, (seq1, seq2, t_length1, t_length2) in enumerate(batch):
        # Pad each sequence to max_length
        padded_seq1 = np.pad(
            seq1,
            ((0, 0), (0, max_length - seq1.shape[1])),
            "constant",
            constant_values=0,
        )
        padded_seq2 = np.pad(
            seq2,
            ((0, 0), (0, max_length - seq2.shape[1])),
            "constant",
            constant_values=0,
        )

        padded_batch1.append(padded_seq1)
        padded_batch2.append(padded_seq2)
        t_lengths1[i] = t_length1
        t_lengths2[i] = t_length2

    # Convert lists of arrays to tensors
    padded_batch1 = torch.tensor(np.stack(padded_batch1), dtype=precision)
    padded_batch2 = torch.tensor(np.stack(padded_batch2), dtype=precision)

    return padded_batch1, padded_batch2, t_lengths1, t_lengths2


def init_data_loader(
    data_config: dict,
    train_config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Initialize train and validation dataloaders split by transcript length.

    Returns two training dataloaders that contain subsets of transcripts based
    on sequence length. A sequence length of 3900 provides a good 3:1 split
    in the current transcript set.

    Args:
        data_config: Dictionary of data configuration values.
        train_config: Dictionary of training config values.
        save_datasets: Whether processed transcripts are cached to disk.
        save_maps: Whether to save the generated transcript maps.
        mini: Specifies whether a subset of transcripts are used.

    Returns:
        Train and validation dataloaders.
    """
    with open(data_config["short_transcript_map_path"], "rb") as f:
        short_tt_map = pickle.load(f)

    with open(data_config["long_transcript_map_path"], "rb") as f:
        long_tt_map = pickle.load(f)

    short_train_tt_map, short_val_tt_map = split_dict(short_tt_map, 0.95)

    train_dataset_shorter = TranscriptTranscriptDataset(
        short_train_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        mask_percentage=data_config["proportion_to_mask"],
    )

    train_dataset_longer = TranscriptTranscriptDataset(
        long_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        mask_percentage=data_config["proportion_to_mask"],
    )

    val_dataset = TranscriptTranscriptDataset(
        short_val_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
    )

    train_data_short_loader = DataLoader(
        train_dataset_shorter,
        batch_size=train_config["gpu_batch_sizes"][1],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )

    train_data_long_loader = DataLoader(
        train_dataset_longer,
        batch_size=train_config["gpu_batch_sizes"][0],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=train_config["gpu_batch_sizes"][2],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
    )

    return train_data_short_loader, train_data_long_loader, val_data_loader


def init_weighted_data_loader(
    data_config: dict,
    train_config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Initialize train and validation dataloaders split by transcript length.

    Returns two training dataloaders that contain subsets of transcripts based
    on sequence length. A sequence length of 3900 provides a good 3:1 split
    in the current transcript set.

    Args:
        data_config: Dictionary of data configuration values.
        train_config: Dictionary of training config values.
        save_datasets: Whether processed transcripts are cached to disk.
        save_maps: Whether to save the generated transcript maps.
        mini: Specifies whether a subset of transcripts are used.

    Returns:
        Train and validation dataloaders.
    """
    with open(data_config["short_transcript_map_path"], "rb") as f:
        short_tt_map = pickle.load(f)

    with open(data_config["long_transcript_map_path"], "rb") as f:
        long_tt_map = pickle.load(f)

    short_train_tt_map, short_val_tt_map = split_dict(short_tt_map, 0.95)

    train_dataset_shorter = WeightedTranscriptTranscriptDataset(
        short_train_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        mask_percentage=data_config["proportion_to_mask"],
    )

    train_dataset_longer = WeightedTranscriptTranscriptDataset(
        long_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        mask_percentage=data_config["proportion_to_mask"],
    )

    val_dataset = WeightedTranscriptTranscriptDataset(
        short_val_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
    )

    train_data_short_loader = DataLoader(
        train_dataset_shorter,
        batch_size=train_config["gpu_batch_sizes"][1],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )

    train_data_long_loader = DataLoader(
        train_dataset_longer,
        batch_size=train_config["gpu_batch_sizes"][0],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=train_config["gpu_batch_sizes"][2],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
    )

    return train_data_short_loader, train_data_long_loader, val_data_loader
