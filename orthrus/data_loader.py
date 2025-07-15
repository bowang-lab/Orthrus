import pickle
import sys
from typing import Callable
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from orthrus.data import Transcript
from orthrus.util import split_dict

from orthrus.util import get_transcript_path


# if a transcript is orthologous relative to the base
def is_ortho(t_id: str, base_species: str = None) -> bool:
    """
    Check if a transcript is orthologous.

    If we are using lncRNA data, we compare the species name in the transcript ID
    to the base species name. If they are different, we consider it orthologous.

    If we are using any other type of transcript, we check if the transcript ID
    has three parts when split by "_". If it does, we consider it orthologous.

    # TODO: Check whether this second case works when comparing between the key species.
        Like does this work between human and mouse, or does this only apply to Zoonomia?

    Args:
        t_id (str): Transcript ID.
        base_species (str): Base species name.
    Returns:
        bool: True if the transcript is orthologous, False otherwise.
    """
    if "TCONS" in t_id:
        return (
            base_species != t_id.split("_")[0]
        )  # different species means ortholog for lncRNA
    else:
        return len(t_id.split("_")) == 3


class MaskSequenceOnlyTransform:
    """
    A transform that, given a fraction of positions to mask, sets the first four channels
    (the sequence channels) at those positions to a specified mask_value.

    For an input transcript tensor of shape [channels, length],
    a fraction (mask_percentage) of positions (columns) are randomly chosen, and only
    the first four channels at those positions are replaced with mask_value.
    """

    def __init__(
        self, mask_percentage: float, mask_value: float = 0.0, n_channels: int = 4
    ):
        """
        :param mask_percentage: Fraction of positions to mask (between 0 and 1).
        :param mask_value: The value to assign to masked positions.
        :param n_channels: The number of channels to mask.
        """
        self.mask_percentage = mask_percentage
        self.mask_value = mask_value
        self.n_channels = n_channels

    def __call__(self, transcript_tensor: np.ndarray) -> np.ndarray:
        # If a torch.Tensor is provided, convert it to a NumPy array.
        if isinstance(transcript_tensor, torch.Tensor):
            transcript_tensor = transcript_tensor.numpy()

        # Get the sequence length (number of columns).
        seq_length = transcript_tensor.shape[1]

        # Determine how many positions to mask.
        num_to_mask = int(seq_length * self.mask_percentage)

        # If no positions are to be masked, return the original.
        if num_to_mask == 0:
            return transcript_tensor

        # Randomly choose indices to mask (without replacement).
        mask_indices = np.random.choice(seq_length, num_to_mask, replace=False)

        # Mask only the specified channels.
        transcript_tensor[: self.n_channels, mask_indices] = self.mask_value

        return transcript_tensor


class DropTracksTransform:
    """
    A transform that, with probability `drop_probability` per track,
    sets the values of the 5th and/or 6th channels to zero.
    """

    def __init__(self, drop_probability: float, i_of_track, mask_value: float = 0.0):
        """
        :param mask_probability: Probability of masking each track independently.
        """
        self.drop_probability = drop_probability
        self.mask_value = mask_value
        self.i_of_track = i_of_track - 1
        assert i_of_track in [5, 6], "Only 5th and 6th channels can be masked"

    def __call__(self, transcript_tensor: np.ndarray) -> np.ndarray:
        """
        :param transcript_tensor: A numpy array of shape [channels, length], for example.
        :return: The same array with the 5th and/or 6th channels masked with probability
        """
        # Convert to torch tensor if you prefer to do random ops in torch, but you can do it
        # in NumPy as well.
        if not isinstance(transcript_tensor, torch.Tensor):
            transcript_tensor = torch.from_numpy(transcript_tensor)

        if (
            torch.rand(1).item() < self.drop_probability
            and self.i_of_track
            < transcript_tensor.shape[
                0
            ]  # needed to fix in case num channels is less than 6
        ):
            # channel index 4 in 0-based indexing
            transcript_tensor[self.i_of_track, :] = self.mask_value

        # Convert back to numpy if your pipeline expects numpy arrays
        return transcript_tensor.numpy()


class CombineUtrCdsTransform:
    """
    A transform that, with probability `combine_probability`, replaces the base transcript's
    coding region with the pair transcript's coding region. Specifically:
      1) Identify the base transcript's first and last coding positions (channel 4).
      2) Take the base 5′ UTR up to `first_cds_pos`.
      3) Take the base 3′ UTR after `last_cds_pos + 2`.
      4) Identify the pair transcript's first and last coding positions (channel 4).
         Take that region [pair_cds_start : pair_cds_end + 3].
      5) Concatenate [base_5′-UTR, pair-CDS, base-3′-UTR].
    Otherwise, if combining is skipped or invalid, returns `pair_transcript_tensor`.
    """

    def __init__(self, combine_probability: float = 1.0, max_length: int = 12288):
        """
        :param combine_probability: Probability of performing the combination.
        :param max_length: Maximum length of the combined transcript.
        """
        self.combine_probability = combine_probability
        self.max_length = max_length

    def __call__(
        self, base_transcript_tensor: np.ndarray, pair_transcript_tensor: np.ndarray
    ) -> np.ndarray:
        # With some probability, we skip combining and just return the pair transcript
        if np.random.rand() > self.combine_probability:
            return pair_transcript_tensor

        # If shapes differ, just return the pair transcript
        if base_transcript_tensor.shape[0] != pair_transcript_tensor.shape[0]:
            return pair_transcript_tensor

        # Channel 4 is the coding track for each
        base_coding = base_transcript_tensor[4, :]  # shape = [length_base]
        pair_coding = pair_transcript_tensor[4, :]  # shape = [length_pair]

        # Find first and last coding positions in base
        base_cds_positions = np.where(base_coding == 1)[0]
        if len(base_cds_positions) == 0:
            # No coding region in base => return pair
            return pair_transcript_tensor

        first_cds_pos = base_cds_positions[0]
        last_cds_pos = base_cds_positions[-1]

        # 5′ UTR = everything up to `first_cds_pos`
        base_5p_utr = base_transcript_tensor[:, :first_cds_pos]

        # 3′ UTR = everything after (last_cds_pos + 2)
        # We assume last_cds_pos is the start of the final codon, so +2 includes the wobble
        base_3p_utr = base_transcript_tensor[:, (last_cds_pos + 3) :]

        # Identify the pair transcript's first and last CDS positions
        pair_cds_positions = np.where(pair_coding == 1)[0]
        if len(pair_cds_positions) == 0:
            # No coding region in pair => return pair
            return pair_transcript_tensor

        pair_first_cds = pair_cds_positions[0]
        pair_last_cds = pair_cds_positions[-1]

        # Slice the pair transcript's CDS region from pair_first_cds to pair_last_cds+2
        # +2 to include the entire final codon; in Python slicing it's +3
        pair_cds = pair_transcript_tensor[:, pair_first_cds : (pair_last_cds + 3)]

        # Concatenate [base_5′UTR, pair_CDS, base_3′UTR]
        merged = np.concatenate([base_5p_utr, pair_cds, base_3p_utr], axis=1)

        # If the merged transcript is shorter than the pair transcript, return the pair transcript
        if merged.shape[1] > self.max_length:
            return pair_transcript_tensor

        return merged


class CdsOnlyOrUtrDropTransform:
    """
    Applies CDS-only or UTR-dropping logic.

    This transform expects transcript IDs to determine whether to apply the logic.

    If `cds_only` is True, all transcripts are sliced to their coding region (channel 4).
    If `utr_dropping` is True, and:
        - the base is a splice isoform (not ortholog)
        - the pair is an ortholog
    then UTRs are removed by slicing both transcripts to their coding region.

    We apply it if only retaining CDS, or in cases where we are pairing a splice isoform
    with an orthologous transcript. The second case is to make sure when comparing a transcript
    with a UTR, we are not comparing it with a transcript that does not have a UTR.

    Args:
        utr_dropping (bool): Whether to apply UTR dropping logic.
        cds_only (bool): Whether to always slice to CDS.
        is_ortho_fn (Callable): Function that returns True if a transcript ID is an ortholog.
    """

    def __init__(self, utr_dropping: bool, cds_only: bool, is_ortho_fn):
        self.utr_dropping = utr_dropping
        self.cds_only = cds_only
        self.is_ortho = is_ortho_fn

    def __call__(
        self,
        base_id,
        base_species,
        pair_id,
        encoded_base_transcript,
        encoded_pair_transcript,
    ):
        # Check if base and pair are orthologs
        base_is_ortho = self.is_ortho(base_id, base_species)
        pair_is_ortho = self.is_ortho(pair_id, base_species)

        # Apply CDS-only or UTR-dropping logic
        if self.cds_only or (
            self.utr_dropping and (not base_is_ortho) and pair_is_ortho
        ):
            # cds is at index 4
            base_cds = encoded_base_transcript[4, :]
            pair_cds = encoded_pair_transcript[4, :]

            base_indices = np.where(base_cds == 1)[0]
            pair_indices = np.where(pair_cds == 1)[0]

            if base_indices.size and pair_indices.size:
                base_first, base_last = base_indices[0], base_indices[-1]
                pair_first, pair_last = pair_indices[0], pair_indices[-1]

                # Adjust for cases where adding 3 overshoots the transcript length
                base_last = base_last + min(
                    3, encoded_base_transcript.shape[1] - base_last
                )
                pair_last = pair_last + min(
                    3, encoded_pair_transcript.shape[1] - pair_last
                )

                # Slice to remove UTRs
                encoded_base_transcript = encoded_base_transcript[
                    :, base_first:base_last
                ]
                encoded_pair_transcript = encoded_pair_transcript[
                    :, pair_first:pair_last
                ]

        return encoded_base_transcript, encoded_pair_transcript


class MaskTrackTransform:
    """
    A transform that masks a percentage of positions in the `i_of_track`-th track.
    """

    def __init__(
        self, i_of_track: int, mask_percentage: float = 0.15, mask_value: float = 0.0
    ):
        """
        :param i_of_track: Index of the track to mask (0-indexed).
        :param mask_percentage: Percentage of positions to mask within the track.
        :param mask_value: Value to use for masking.
        """
        self.i_of_track = i_of_track - 1
        self.mask_percentage = mask_percentage
        self.mask_value = mask_value

    def __call__(self, transcript_tensor: np.ndarray) -> np.ndarray:
        # If a torch.Tensor is provided, convert it to a NumPy array
        if isinstance(transcript_tensor, torch.Tensor):
            transcript_tensor = transcript_tensor.numpy()

        # Get the sequence length
        seq_length = transcript_tensor.shape[1]

        # Determine how many positions to mask
        num_to_mask = int(seq_length * self.mask_percentage)

        # If no positions are to be masked, return the original
        if num_to_mask == 0:
            return transcript_tensor

        # Randomly choose indices to mask (without replacement)
        mask_indices = np.random.choice(seq_length, num_to_mask, replace=False)

        # Mask only the specified track at the chosen positions
        transcript_tensor[self.i_of_track, mask_indices] = self.mask_value

        return transcript_tensor


class TranscriptTranscriptDataset(Dataset):
    """A PyTorch Dataset that operates over Transcript objects."""

    def __init__(
        self,
        transcript_mapping: dict[str, list[str]],
        npz_save_dir: str,
        # mask_percentage: float = 0.0,
        weight: float = 0.5,
        transforms: list[Callable] = [],
        no_splice_isoforms: bool = False,
        no_ortho_isoforms: bool = False,
    ) -> None:
        """Initialize a transcript dataset.

        Args:
            transcript_mapping: Map of transcript to associated transcripts.
            npz_save_dir: Path where processed transcripts are stored.
            weight: Weight for splice/ortho sampling. Larger weight means less
                chance of sampling splice isoform.
            transforms: List of transformations to apply to the transcripts.
        """
        self.transcript_mapping = transcript_mapping
        self.npz_save_dir = npz_save_dir
        self.weight = weight
        self.transforms = transforms
        self.no_splice_isoforms = no_splice_isoforms
        self.no_ortho_isoforms = no_ortho_isoforms
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

        return t_data

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        :return: Number of items in dataset
        """
        return len(self.transcript_mapping)

    def __getitem__(self, idx: int) -> dict:
        """Get Transcript object and a randomly sampled associated Transcript.

        :param idx: The index of the item.
        :return: A dictionary containing the one-hot encoded RNA sequences.
        """
        base_transcript_id = self.index_to_transcript[idx]
        associated_transcripts = self.transcript_mapping[base_transcript_id]

        # Precompute base transcript properties and classify associated transcripts
        base_species = base_transcript_id.split("_")[0]

        if self.weight == 1.0 or self.no_splice_isoforms:
            candidate_splice = []
        else:
            candidate_splice = [t for t in associated_transcripts if not is_ortho(t, base_species)]

        if self.weight == 0.0 or self.no_ortho_isoforms:
            candidate_orthos = []
        else:
            candidate_orthos = [t for t in associated_transcripts if is_ortho(t, base_species)]

        # Sample the pair transcript based on the candidates and weight
        pair_transcript_id = None
        orthology = False
        can_sample_splice = len(candidate_splice) > 0
        can_sample_ortho = len(candidate_orthos) > 0

        # Determine sampling strategy
        if can_sample_splice and can_sample_ortho:
            # If both types are available, use weight to decide
            # self.weight is the probability of sampling an ortholog
            if np.random.random() < self.weight:
                pair_transcript_id = np.random.choice(candidate_orthos)
                orthology = True
            else:
                pair_transcript_id = np.random.choice(candidate_splice)
        elif can_sample_ortho:
            # Only orthologs are available
            pair_transcript_id = np.random.choice(candidate_orthos)
            orthology = True
        elif can_sample_splice:
            # Only splice isoforms are available
            pair_transcript_id = np.random.choice(candidate_splice)
        else:
            # Fallback: if no valid candidates, pair with self
            pair_transcript_id = base_transcript_id

        # Determine the pairing type for logging/analysis
        if pair_transcript_id == base_transcript_id:
            pairing_type = "self"
        elif orthology:
            pairing_type = "ortho"
        else:
            pairing_type = "splice"

        # Load the transcripts
        encoded_base_transcript = self.load_transcript(base_transcript_id)
        encoded_pair_transcript = self.load_transcript(pair_transcript_id)

        # Apply all transforms in the list
        for i, transform in enumerate(self.transforms):
            if isinstance(transform, CombineUtrCdsTransform):
                # only apply the combine transform if the transcripts are orthologous
                if orthology:
                    assert i == 0, "CombineUtrCdsTransform must be the first transform"
                    encoded_pair_transcript = transform(
                        encoded_base_transcript, encoded_pair_transcript
                    )
            elif isinstance(transform, CdsOnlyOrUtrDropTransform):
                # Apply the transform to both transcripts
                encoded_base_transcript, encoded_pair_transcript = transform(
                    base_transcript_id,
                    base_species,
                    pair_transcript_id,
                    encoded_base_transcript,
                    encoded_pair_transcript,
                )
            else:
                encoded_base_transcript = transform(encoded_base_transcript)
                encoded_pair_transcript = transform(encoded_pair_transcript)

        len_t1 = encoded_base_transcript.shape[1]
        len_t2 = encoded_pair_transcript.shape[1]

        return {
            "base_seq": encoded_base_transcript,
            "pair_seq": encoded_pair_transcript,
            "base_len": len_t1,
            "pair_len": len_t2,
            "pairing_type": pairing_type,
        }


def collate_fn(
    batch: list[dict],
    mixed_precison=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transcript encoding collation function for DataLoader.

    This function dynamically pads the batch sequences based on the longest
    sequence in the batch.

    :param batch:
        List of dictionaries as returned by __getitem__. Each dict contains
        sequences, lengths, and pairing type.
    :param mixed_precision: Uses 16-bit float when possible.
    :return:
        A tuple of tensors:
        - Padded sequences for the batch, shape (B, C, L).
        - Sequence lengths for the batch, shape (B,).
        - Pairing types for the batch, shape (B,).
    """
    # Find the longest sequence in the batch for both transcripts
    # Assumes sequences are (C, L) where C is channel and L is length
    max_length1 = max([item["base_seq"].shape[1] for item in batch])
    max_length2 = max([item["pair_seq"].shape[1] for item in batch])

    # Determine the overall max length for padding
    max_length = max(max_length1, max_length2)

    # Pad sequences and prepare the batch
    padded_batch1 = []
    padded_batch2 = []
    t_lengths1 = []
    t_lengths2 = []
    pairing_types = []
    type_map = {"self": 0, "splice": 1, "ortho": 2}

    # conditional on mixed precision set to float16 or 32
    if mixed_precison:
        precision = torch.float16
    else:
        precision = torch.float32

    for item in batch:
        seq1 = item["base_seq"]
        seq2 = item["pair_seq"]
        t_lengths1.append(item["base_len"])
        t_lengths2.append(item["pair_len"])
        pairing_types.append(type_map[item["pairing_type"]])

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

    # Convert lists of arrays to tensors
    padded_batch1 = torch.tensor(np.stack(padded_batch1), dtype=precision)
    padded_batch2 = torch.tensor(np.stack(padded_batch2), dtype=precision)
    t_lengths1 = torch.tensor(t_lengths1, dtype=torch.int32)
    t_lengths2 = torch.tensor(t_lengths2, dtype=torch.int32)

    # combine into one tensor
    padded_batch = torch.cat((padded_batch1, padded_batch2), dim=0)
    padded_batch_lengths = torch.cat((t_lengths1, t_lengths2), dim=0)
    pairing_types = torch.tensor(pairing_types, dtype=torch.long)

    return padded_batch, padded_batch_lengths, pairing_types


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

    Returns:
        Train and validation dataloaders.
    """
    with open(data_config["short_transcript_map_path"], "rb") as f:
        short_tt_map = pickle.load(f)

    with open(data_config["long_transcript_map_path"], "rb") as f:
        long_tt_map = pickle.load(f)

    short_train_tt_map, short_val_tt_map = split_dict(
        short_tt_map, 5
    )  # 95% train, 5% val

    short_transforms = []
    long_transforms = []

    if data_config["combine_probability"] > 0:
        combine_transform_long = CombineUtrCdsTransform(
            combine_probability=data_config["combine_probability"], max_length=12288
        )
        combine_transform_short = CombineUtrCdsTransform(
            combine_probability=data_config["combine_probability"], max_length=3900
        )
        short_transforms.append(combine_transform_short)
        long_transforms.append(combine_transform_long)

    if data_config["utr_dropping"] or data_config["cds_only"]:
        cds_or_utr_dropping_transform = CdsOnlyOrUtrDropTransform(
            utr_dropping=data_config["utr_dropping"],
            cds_only=data_config["cds_only"],
            is_ortho_fn=is_ortho,
        )
        short_transforms.append(cds_or_utr_dropping_transform)
        long_transforms.append(cds_or_utr_dropping_transform)

    if data_config["mask_track_probability_5"] > 0:
        mask_track_transform5 = MaskTrackTransform(
            i_of_track=5,
            mask_percentage=data_config["mask_track_probability_5"],
            mask_value=data_config["mask_value"],
        )
        short_transforms.append(mask_track_transform5)
        long_transforms.append(mask_track_transform5)

    if data_config["mask_track_probability_6"] > 0:
        mask_track_transform6 = MaskTrackTransform(
            i_of_track=6,
            mask_percentage=data_config["mask_track_probability_6"],
            mask_value=data_config["mask_value"],
        )
        short_transforms.append(mask_track_transform6)
        long_transforms.append(mask_track_transform6)

    if data_config["drop_track_probability_5"] > 0:
        drop_track_transform5 = DropTracksTransform(
            drop_probability=data_config["drop_track_probability_5"], i_of_track=5
        )
        short_transforms.append(drop_track_transform5)
        long_transforms.append(drop_track_transform5)

    if data_config["drop_track_probability_6"] > 0:
        drop_track_transform6 = DropTracksTransform(
            drop_probability=data_config["drop_track_probability_6"], i_of_track=6
        )
        short_transforms.append(drop_track_transform6)
        long_transforms.append(drop_track_transform6)

    if data_config["proportion_to_mask"] > 0:
        mask_sequence_transform = MaskSequenceOnlyTransform(
            mask_percentage=data_config["proportion_to_mask"],
            mask_value=data_config["mask_value"],
            n_channels=4,
        )
        short_transforms.append(mask_sequence_transform)
        long_transforms.append(mask_sequence_transform)

    train_dataset_shorter = TranscriptTranscriptDataset(
        short_train_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        weight=data_config["weight"],
        transforms=short_transforms,
        no_splice_isoforms=data_config["no_splice_isoforms"],
        no_ortho_isoforms=data_config["no_ortho_isoforms"],
    )

    train_dataset_longer = TranscriptTranscriptDataset(
        long_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        weight=data_config["weight"],
        transforms=long_transforms,
        no_splice_isoforms=data_config["no_splice_isoforms"],
        no_ortho_isoforms=data_config["no_ortho_isoforms"],
    )

    val_dataset = TranscriptTranscriptDataset(
        short_val_tt_map,
        npz_save_dir=data_config["transcript_save_dir"],
        weight=data_config["weight"],
        no_splice_isoforms=data_config["no_splice_isoforms"],
        no_ortho_isoforms=data_config["no_ortho_isoforms"],
    )

    train_data_short_loader = DataLoader(
        train_dataset_shorter,
        batch_size=train_config["gpu_batch_sizes"][1],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    train_data_long_loader = DataLoader(
        train_dataset_longer,
        batch_size=train_config["gpu_batch_sizes"][0],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=train_config["gpu_batch_sizes"][2],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_data_short_loader, train_data_long_loader, val_data_loader


class GeneSpeciesTranscriptDataset(Dataset):
    """A PyTorch Dataset that operates over Transcript objects."""

    def __init__(
        self,
        gene_mapping: dict[str, list[str]],
        npz_save_dir: str,
        mask_percentage: float = 0.0,
        transcript_type: str = None,
        weight: float = 0.5,
        num_globals: int = 2,
        num_locals: int = 8,
    ) -> None:
        """Initialize a transcript dataset.

        Args:
            gene_mapping: Map of Gene to SpeciesGene objects (containing transcripts)
            npz_save_dir: Path where processed transcripts are stored.
            mask_percentage: Percentage of tokens to randomly mask.
        """
        self.gene_mapping = gene_mapping
        self.npz_save_dir = npz_save_dir
        self.mask_percentage = mask_percentage
        self.transcript_type = transcript_type
        self.weight = weight
        self.num_globals = num_globals
        self.num_locals = num_locals

        self.index_to_gene = {}
        for i, gene in enumerate(self.gene_mapping.keys()):
            self.index_to_gene[i] = gene

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
