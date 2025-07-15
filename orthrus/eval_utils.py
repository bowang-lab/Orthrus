import json
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import pytorch_lightning as pl

from orthrus.mamba import MixerModel
from orthrus.saluki import SalukiModel
from orthrus.dilated_resnet import dilated_small, not_dilated_small


def load_lightning_model(run_path: str, checkpoint_name: str) -> pl.LightningModule:
    """Load a PyTorch Lightning model from a checkpoint.

    This function is designed to restore a complete LightningModule from a file,
    including all its components like projection heads. It dynamically determines
    which model class to load based on the 'mask_prop' in train_config.json.

    Args:
        run_path: The directory where the checkpoint and configs are stored.
        checkpoint_name: The name of the checkpoint file (e.g., 'model.ckpt').

    Returns:
        The loaded LightningModule instance.
    """
    # Load all config files
    config_names = ["model", "projector", "optimizer", "train", "data"]
    configs = {}
    for name in config_names:
        config_path = os.path.join(run_path, f"{name}_config.json")
        with open(config_path, "r") as f:
            configs[f"{name}_config"] = json.load(f)

    if configs["train_config"]["mask_prop"] > 0:
        from orthrus.mlm_model import ContrastiveLearningModel
        print("Loading Contrastive + MLM model")
    else:
        from orthrus.model import ContrastiveLearningModel
        print("Loading Contrastive only model")

    checkpoint_path = os.path.join(run_path, checkpoint_name)
    model = ContrastiveLearningModel.load_from_checkpoint(
        checkpoint_path,
        **configs
    )
    return model


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
        model_config = json.load(f)

    if "n_tracks" not in model_config:
        with open(data_config_path, "r") as f: # this means we are loading a finetuned model
            data_config = json.load(f)
        n_tracks = data_config["n_tracks"]
    else:
        n_tracks = model_config["n_tracks"]

    model_path = os.path.join(run_path, checkpoint_name)

    # get model name from run_path
    model_name = os.path.basename(run_path)

    if model_config['model_class'] == "ssm":
        model = MixerModel(
            d_model = model_config["ssm_model_dim"],
            n_layer = model_config["ssm_n_layers"],
            input_dim = n_tracks,
            bidirectional = "caduceus" if "caduceus" in model_name else "hydra" if "hydra" in model_name else None,
            bidirectional_strategy = model_config["bidirectional_strategy"] if "bidirectional_strategy" in model_config else "add",
            bidirectional_weight_tie = model_config["bidirectional_weight_tie"] if "bidirectional_weight_tie" in model_config else True,
            gradient_checkpointing = model_config["gradient_checkpointing"] if "gradient_checkpointing" in model_config else False,
        )
    elif model_config['model_class'] == "resnet":
        # choose a small dilated or not, based on config
        if model_config["resnet"] == "dilated_small":
            ResNet = dilated_small
        elif model_config["resnet"] == "not_dilated_small":
            ResNet = not_dilated_small
        else:
            raise ValueError(
                f"Unknown resnet variant: {model_config['resnet']}"
            )

        model = ResNet(
            num_classes=data_config["num_classes"],
            dropout_prob=model_config["dropout_prob"],
            norm_type=model_config["norm_type"],
            kernel_size=model_config["kernel_size"],
            pooling_layer=model_config["pooling_layer"],
            add_shift=model_config["add_shift"],
            increase_dilation=model_config["increase_dilation"],
            global_pooling_layer=model_config["global_pooling_layer"],
        )    
    
    elif model_config['model_class'] == "saluki":
        model = SalukiModel(
            seq_depth=data_config["n_tracks"],
            add_shift=model_config["add_shift"]
        )
    else:
        raise NotImplementedError(f"Model class {model_config['model_class']} not implemented")

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict[k.lstrip("model")[1:]] = v

    model.load_state_dict(state_dict)
    return model


def get_unpadded_seq_lens(seq_arr_batch: np.ndarray) -> np.ndarray:
    """Return lengths of unpadded sequences.

    Assumes data is of shape (B x L x C).

    Args:
        seq_arr_batch: Sequences right-padded with zeros.
    Returns:
        Array of length of sequences without padding.
    """
    summed = (np.sum(seq_arr_batch, axis=-1) >= 1).astype(int)
    reverse_sums = np.flip(summed, axis=-1)
    first_occurs = np.argmax(reverse_sums, axis=1)

    return seq_arr_batch.shape[1] - first_occurs


def get_representations(
    model: MixerModel,
    seq_array: np.ndarray,
    batch_size: int = 1,
    channel_last: bool = False,
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

    Returns:
        Embedded representations. Numpy array of shape (B x H).
    """
    if not channel_last:
        seq_array = np.transpose(seq_array, (0, 2, 1))

    s_lens = get_unpadded_seq_lens(seq_array)

    dataset = [(d, l, i) for i, (d, l) in enumerate(zip(seq_array, s_lens))]
    sorted_dataset = sorted(dataset, key=lambda x: x[1])

    representations = [None] * len(s_lens)

    for i in tqdm(range(0, len(sorted_dataset), batch_size)):
        b_data = [d[0] for d in sorted_dataset[i:i + batch_size]]
        b_lens = [d[1] for d in sorted_dataset[i:i + batch_size]]
        orig_inds = [d[2] for d in sorted_dataset[i:i + batch_size]]

        seq_tt = torch.Tensor(np.array(b_data))
        seq_lens = torch.Tensor(b_lens)

        seq_tt = seq_tt[:, :max(b_lens)]

        if torch.cuda.is_available():
            seq_tt = seq_tt.cuda()
            seq_lens = seq_lens.cuda()

        with torch.no_grad():
            rep = model.representation(seq_tt, seq_lens, channel_last=True)
            rep_np = rep.cpu().numpy()

            for r, i in zip(rep_np, orig_inds):
                representations[i] = r

    return np.stack(representations)

class PearsonR(Metric):
    def __init__(self, num_targets=1, summarize=True):
        super().__init__()
        self.num_targets = num_targets
        self.summarize = summarize
        self.add_state("count", default=torch.zeros(num_targets), dist_reduce_fx="sum")
        self.add_state(
            "product", default=torch.zeros(num_targets), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_sum", default=torch.zeros(num_targets), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_sumsq", default=torch.zeros(num_targets), dist_reduce_fx="sum"
        )
        self.add_state(
            "pred_sum", default=torch.zeros(num_targets), dist_reduce_fx="sum"
        )
        self.add_state(
            "pred_sumsq", default=torch.zeros(num_targets), dist_reduce_fx="sum"
        )

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.squeeze().float()
        y_pred = y_pred.squeeze().float()

        reduce_axes = 0 if len(y_true.shape) == 1 else [0]

        product = torch.sum(y_true * y_pred, dim=reduce_axes)
        true_sum = torch.sum(y_true, dim=reduce_axes)
        true_sumsq = torch.sum(y_true**2, dim=reduce_axes)
        pred_sum = torch.sum(y_pred, dim=reduce_axes)
        pred_sumsq = torch.sum(y_pred**2, dim=reduce_axes)
        count = torch.sum(torch.ones_like(y_true), dim=reduce_axes)

        self.product += product.unsqueeze(0)
        self.true_sum += true_sum.unsqueeze(0)
        self.true_sumsq += true_sumsq.unsqueeze(0)
        self.pred_sum += pred_sum.unsqueeze(0)
        self.pred_sumsq += pred_sumsq.unsqueeze(0)
        self.count += count.unsqueeze(0)

    def compute(self):
        true_mean = self.true_sum / self.count
        true_mean2 = true_mean**2
        pred_mean = self.pred_sum / self.count
        pred_mean2 = pred_mean**2

        term1 = self.product
        term2 = -true_mean * self.pred_sum
        term3 = -pred_mean * self.true_sum
        term4 = self.count * true_mean * pred_mean
        covariance = term1 + term2 + term3 + term4

        true_var = self.true_sumsq - self.count * true_mean2
        pred_var = self.pred_sumsq - self.count * pred_mean2
        pred_var = torch.where(
            pred_var > 1e-12, pred_var, torch.full_like(pred_var, float("inf"))
        )

        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var

        if self.summarize:
            return torch.mean(correlation)
        else:
            return correlation


class SpearmanR(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.eps = 1e-8

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds.squeeze())
        self.target.append(target.squeeze())

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        # some intermediate computation...
        r_preds, r_target = self._rank_data(preds), self._rank_data(target)
        preds_diff = r_preds - r_preds.mean(0)
        target_diff = r_target - r_target.mean(0)
        cov = (preds_diff * target_diff).mean(0)
        preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
        target_std = torch.sqrt((target_diff * target_diff).mean(0))
        # finalize the computations
        corrcoef = cov / (preds_std * target_std + self.eps)
        return torch.clamp(corrcoef, -1.0, 1.0)

    def _rank_data(self, data: torch.Tensor) -> torch.Tensor:
        return data.argsort().argsort().to(torch.float32)