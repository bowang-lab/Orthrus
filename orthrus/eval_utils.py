import json
import os

import numpy as np

import torch
import torch.nn as nn

from orthrus.mamba import MixerModel


def load_model(run_path: str, checkpoint_name: str) -> nn.Module:
    """Load trained model located at specified path.

    Args:
        run_path: Path where run data is located.
        checkpoint_name: Name of model checkpoint to load.

    Returns:
        Model with loaded weights.
    """
    model_config_path = os.path.join(run_path, "model_config.json")
    data_config_path = os.path.join(run_path, "data_config.json")

    with open(model_config_path, "r") as f:
        model_params = json.load(f)

    # TODO: Temp backwards compatibility
    if "n_tracks" not in model_params:
        with open(data_config_path, "r") as f:
            data_params = json.load(f)
        n_tracks = data_params["n_tracks"]
    else:
        n_tracks = model_params["n_tracks"]

    model_path = os.path.join(run_path, checkpoint_name)

    model = MixerModel(
        d_model=model_params["ssm_model_dim"],
        n_layer=model_params["ssm_n_layers"],
        input_dim=n_tracks,
    )
    checkpoint = torch.load(model_path)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict[k.lstrip("model")[1:]] = v

    model.load_state_dict(state_dict)
    return model


def get_unpadded_seq_lens(
    seq_arr_batch: np.ndarray,
    channels_last: bool = True,
) -> np.ndarray:
    """Return lengths of unpadded sequences.

    Assumes data is of shape (B x L x C).

    Args:
        seq_arr_batch: Sequences right-padded with zeros.
    Returns:
        Array of length of sequences without padding.
    """
    if not channels_last:
        seq_arr_batch = np.transpose(seq_arr_batch, (0, 2, 1))
    summed = (np.sum(seq_arr_batch, axis=-1) >= 1).astype(int)
    reverse_sums = np.flip(summed, axis=-1)
    first_occurs = np.argmax(reverse_sums, axis=1)

    return seq_arr_batch.shape[1] - first_occurs


def get_representations(
    model: MixerModel,
    seq_array: np.ndarray,
    batch_size: int = 1,
    channel_last: bool = False,
    aggregation: str = "mean",
) -> np.ndarray:
    """Convert input tokens into embeddings using Mamba model.

    Input data is expected to be of shape (B x L x C) if channel_last is true,
    else it is of shape (B x C x L). Automatically handles pad tokens.

    Batches inputs by length to reduce padding overhead.

    Args:
        model: Model used to perform inference.
        seq_array: Data to convert to embedded representations.
        batch_size: Batch size to use when processing conversions.
        channel_last: Position of channel dimension.
        aggregation: Aggregation method to use for sequence embeddings.

    Returns:
        Embedded representations. Numpy array of shape (B x H).
    """
    if not channel_last:
        seq_array = np.transpose(seq_array, (0, 2, 1))

    s_lens = get_unpadded_seq_lens(seq_array)

    dataset = [(d, l, i) for i, (d, l) in enumerate(zip(seq_array, s_lens))]
    sorted_dataset = sorted(dataset, key=lambda x: x[1])

    representations = [None] * len(s_lens)

    for i in range(0, len(sorted_dataset), batch_size):
        b_data = [d[0] for d in sorted_dataset[i : i + batch_size]]
        b_lens = [d[1] for d in sorted_dataset[i : i + batch_size]]
        orig_inds = [d[2] for d in sorted_dataset[i : i + batch_size]]

        seq_tt = torch.Tensor(np.array(b_data))
        seq_lens = torch.Tensor(b_lens)

        seq_tt = seq_tt[:, : max(b_lens)]

        if torch.cuda.is_available():
            seq_tt = seq_tt.cuda()
            seq_lens = seq_lens.cuda()

        with torch.no_grad():
            rep = model.representation(
                seq_tt, seq_lens, channel_last=True, aggregation=aggregation,
            )
            rep_np = rep.cpu().numpy()

            for r, i in zip(rep_np, orig_inds):
                representations[i] = r

    return np.stack(representations)
