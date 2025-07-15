import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW

from orthrus.layers import ProjectionHead, SequenceProjectionHead
from orthrus.mamba import MixerModel, mean_unpadded
from orthrus.losses import DCLLoss, PairingTypeCounter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from orthrus.dilated_resnet import dilated_small, dilated_medium, DilatedResnet
from orthrus.saluki import saluki_small, saluki_medium, SalukiModel


def compute_loss_weights(number_steps, weight_start, weight_end, anneal_steps):
    linear_part = torch.linspace(weight_start, weight_end, steps=anneal_steps)
    constant_part = torch.full((number_steps - anneal_steps,), weight_end)
    loss_weights = torch.cat([linear_part, constant_part])
    return loss_weights


class ContrastiveLearningModel(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        projector_config: dict,
        optimizer_config: dict,
        train_config: dict,
        data_config: dict,
    ):
        """Initialize ContrastiveLearningModel.

        Args:
            model_config: Config for model architecture.
            projector_config: Config for projector architecture.
            optimizer_config: Config for optimizer hyperparameters.
            train_config: Config for training hyperparameters.
            data_config: Config used for dataset generation.
        """
        super().__init__()

        self.opt_config = optimizer_config
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config

        self.use_dual_heads = optimizer_config.get("use_dual_heads", False)

        try:
            self.predict_splice = model_config["predict_splice_codon"]
        except KeyError:
            self.predict_splice = False

        try:
            self.predict_masked = model_config["predict_masked"]
        except KeyError:
            self.predict_masked = False

        self.model = self._create_model()

        # If not using MLM only, we need the contrastive projection head
        if self.opt_config["loss_fn"] != "mlm":
            if self.use_dual_heads:
                self.splice_self_head = ProjectionHead(
                    input_features=projector_config["representation_dim"],
                    projection_body=projector_config["projection_body"],
                    projection_head_size=projector_config["projection_head_size"],
                    norm_type=projector_config["projection_norm_type"],
                )
                self.ortho_head = ProjectionHead(
                    input_features=projector_config["representation_dim"],
                    projection_body=projector_config["projection_body"],
                    projection_head_size=projector_config["projection_head_size"],
                    norm_type=projector_config["projection_norm_type"],
                )
            else:
                self.contrastive_projection_head = ProjectionHead(
                    input_features=projector_config["representation_dim"],
                    projection_body=projector_config["projection_body"],
                    projection_head_size=projector_config["projection_head_size"],
                    norm_type=projector_config["projection_norm_type"],
                )

        if self.predict_splice:
            self.splice_head = SequenceProjectionHead(
                input_features=projector_config["representation_dim"],
                projection_body=projector_config["projection_body"],
                projection_head_size=1,
                norm_type=projector_config["projection_norm_type"],
            )

            self.codon_head = SequenceProjectionHead(
                input_features=projector_config["representation_dim"],
                projection_body=projector_config["projection_body"],
                projection_head_size=1,
                norm_type=projector_config["projection_norm_type"],
            )

        if self.predict_masked:
            self.sequence_head = SequenceProjectionHead(
                input_features=projector_config["representation_dim"],
                projection_body=projector_config["projection_body"],
                projection_head_size=4,
                norm_type=projector_config["projection_norm_type"],
            )
            self.masked_loss = nn.CrossEntropyLoss()

        if self.opt_config["loss_fn"] == "dcl":
            if self.use_dual_heads:
                self.ortho_loss_fn = DCLLoss(
                    self_weight=0.0,
                    splice_weight=0.0,
                    ortho_weight=optimizer_config.get("ortho_weight", 1.0),
                    temperature=self.opt_config.get("temperature", 0.1),
                )
                self.splice_self_loss_fn = DCLLoss(
                    self_weight=optimizer_config.get("self_weight", 1.0),
                    splice_weight=1.0,
                    ortho_weight=0.0,
                    temperature=self.opt_config.get("temperature", 0.1),
                )
            else:
                self.c_loss_fn = DCLLoss(
                    self_weight=optimizer_config.get("self_weight", 1.0),
                    ortho_weight=optimizer_config.get("ortho_weight", 1.0),
                    temperature=self.opt_config.get("temperature", 0.1),
                )
        elif (
            self.opt_config["loss_fn"] == "mlm"
        ):  # this means we are not using contrastive learning
            self.ortho_loss_fn = None
            self.splice_self_loss_fn = None
            self.c_loss_fn = None
        else:
            raise NotImplementedError(
                f"Loss function {self.opt_config['loss_fn']} not implemented"
            )

        # Default is no weight annealing
        self.mlm_start_weight = train_config.get("mlm_start_weight", 1)
        self.mlm_anneal_steps = train_config.get("mlm_anneal_steps", 0)
        self.mlm_end_weight = train_config.get("mlm_end_weight", 1)

        self.cl_start_weight = train_config.get("cl_start_weight", 1)
        self.cl_anneal_steps = train_config.get("cl_anneal_steps", 0)
        self.cl_end_weight = train_config.get("cl_end_weight", 1)

        self.train_pairing_counter = PairingTypeCounter()
        self.val_pairing_counter = PairingTypeCounter()

    def _create_model(self):
        model_class = self.model_config.get("model_class", "ssm")
        if model_class == "ssm":
            return MixerModel(
                self.model_config["ssm_model_dim"],
                self.model_config["ssm_n_layers"],
                self.model_config["n_tracks"],
                self.model_config["bidirectional"],
                self.model_config["bidirectional_strategy"],
                self.model_config["bidirectional_weight_tie"],
                self.model_config["gradient_checkpointing"],
            )
        else:
            raise ValueError("mlm_model.py only supports ssm models")

    def _log_metrics(self, stage: str, metrics: dict, batch_size: int):
        on_step = stage == "train"
        for key, value in metrics.items():
            self.log(
                f"{stage}/{key}",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar="loss" in key,
                logger=True,
                batch_size=batch_size,
                sync_dist=(stage == "validation"),
            )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute forward pass for contrastive model.

        Representations are stored in a dictionary with the following keys:
            "global": Mean of unpadded tokens
            "sequence": 4-D logits used for sequence prediction.
            "splice": 1-D logit used for splice site prediction.
            "codon": 1-D logit used for codon prediction.

        Args:
            x: Input tensor of shape (B x L x C).
            lengths: Unpadded length of each input. Has shape (B,).

        Returns:
            Dictionary of all computed representations.
        """
        seq_repr = self.model.forward(x)

        out = {}
        # Only compute the contrastive ("global") branch if not in MLM-only mode.
        if self.opt_config["loss_fn"] != "mlm":
            mean_repr = mean_unpadded(seq_repr, lengths)
            if self.use_dual_heads:
                out["splice_self_global"] = self.splice_self_head(mean_repr)
                out["ortho_global"] = self.ortho_head(mean_repr)
            else:
                out["global"] = self.contrastive_projection_head(mean_repr)

        if self.predict_splice:
            codon_out = self.codon_head(seq_repr)
            splice_out = self.splice_head(seq_repr)

            out["codon"] = codon_out
            out["splice"] = splice_out

        if self.predict_masked:
            seq_out = self.sequence_head(seq_repr)
            out["sequence"] = seq_out

        return out

    def mask_sequence(
        self, x: torch.Tensor, lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask sequence and return label.

        Args:
            x: Sequence to mask. Has shape (B x C x L).
            lens: Unpadded length of each sequence.

        Returns:
            Masked input with shape (B x C x L) and the masked labels for
            cross-entropy loss with shape (B x L).
        """
        B, C, L = x.size()

        # Compute class labels, excludes sequence padding tokens
        _, class_inds = torch.max(x, dim=1)
        labels = torch.where(
            torch.all(x[:, :4, :] == 0, dim=1), torch.tensor(-100), class_inds
        )

        # Remove pad tokens
        ind_arr = torch.arange(L).unsqueeze(0).to(x.device)
        p_mask_flat = ind_arr >= lens.unsqueeze(1)
        p_mask = p_mask_flat.unsqueeze(1).expand(-1, C, -1)

        # Random masking
        r_mask_flat = torch.rand(B, L).to(x.device)
        r_mask_flat = r_mask_flat < self.train_config["mask_prop"]
        r_mask = r_mask_flat.unsqueeze(1).expand(-1, C, -1)

        masked_x = x.clone()
        masked_x = masked_x.masked_fill(r_mask, 0)
        masked_x = masked_x.masked_fill(p_mask, 0)

        # Initially, only physically masked positions are considered for loss.
        loss_mask = r_mask_flat

        # Check for the new hyperparameter to also reconstruct some unmasked tokens.
        reconstruct_unmasked_prop = self.train_config.get(
            "reconstruct_unmasked_prop", 0.0
        )
        if reconstruct_unmasked_prop > 0:
            # Identify positions that are NOT masked and NOT padded.
            unmasked_candidates = ~r_mask_flat & ~p_mask_flat

            # From these candidates, randomly select a fraction to also include in the loss.
            unmasked_to_reconstruct = (
                torch.rand_like(x[:, 0, :]) < reconstruct_unmasked_prop
            ) & unmasked_candidates

            # Add these positions to our loss mask.
            loss_mask = loss_mask | unmasked_to_reconstruct

        # Set the label to -100 for any position NOT in our final loss mask.
        labels[~loss_mask] = -100
        # Ensure padding is always ignored.
        labels[p_mask_flat] = -100

        return masked_x, labels

    def compute_mask_loss(
        self,
        x1_seq_rep: torch.Tensor,
        x2_seq_rep: torch.Tensor,
        lab_x1: torch.Tensor,
        lab_x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mask loss for sequences in batch.

        Args:
            x1_seq_rep: Sequence representations for first positive pairs.
            x2_seq_rep: Sequence representations for second positive pairs.
            lab_x1: Labels for masked tokens in first positive pairs.
            lab_x2: Labels for masked tokens in second positive pairs.

        Returns:
            Cross entropy loss at all masked positions.
        """
        x1_seq = x1_seq_rep.contiguous().view(-1, x1_seq_rep.size(2))
        x2_seq = x2_seq_rep.contiguous().view(-1, x2_seq_rep.size(2))

        lab_x1 = lab_x1.contiguous().view(-1)
        lab_x2 = lab_x2.contiguous().view(-1)

        masked_loss_1 = self.masked_loss(x1_seq, lab_x1)
        masked_loss_2 = self.masked_loss(x2_seq, lab_x2)
        masked_loss = masked_loss_1 + masked_loss_2

        return masked_loss

    def compute_bce_loss(
        self, x: torch.Tensor, lab: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross entropy between prediction and target.

        Masks out padding tokens that are after specified sequence length.

        Args:
            x: Predicted logits of sequences with shape (B x L x C).
            lab: Ground truth sequences with shape (B x L x C).
            lens: Unpadded length of each sequence. Has shape (B,).

        Returns:
            Binary cross entropy loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(x, lab, reduction="none")

        mask = torch.arange(x.size(1)).to(x.device)[None, :] < lens[:, None]
        mask = mask.unsqueeze(-1).expand_as(x)

        masked_bce_loss = bce_loss * mask.float()
        return masked_bce_loss.sum() / lens.sum().float()

    def compute_single_track_bce_loss(
        self,
        x1_rep: torch.Tensor,
        x2_rep: torch.Tensor,
        x1_lab: torch.Tensor,
        x2_lab: torch.Tensor,
        x1_len: torch.Tensor,
        x2_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the BCE loss for single track output.

        All inputs have shape (B x C x L), except lengths which has shape (B,).

        Args:
            x1_rep: Sequence representations for first positive pairs.
            x2_rep: Sequence representations for second positive pairs.
            x1_lab: Labels for masked tokens in first positive pairs.
            x2_lab: Labels for masked tokens in second positive pairs.
            x1_len: Unpadded length of sequences in first positive pair.
            x2_len: Unpadded length of sequences in second positive pair.

        Returns:
            Binary cross entropy for all sequences in batch.
        """
        bce_loss_1 = self.compute_bce_loss(x1_rep, x1_lab, x1_len)
        bce_loss_2 = self.compute_bce_loss(x2_rep, x2_lab, x2_len)

        return bce_loss_1 + bce_loss_2

    def setup(self, stage: str):
        if stage == "fit":
            if self.train_config.get("number_steps", None) is None:
                # Estimate the number of steps if not specified
                number_steps = self.train_config["number_epochs"] * len(
                    self.trainer.train_dataloader
                )
            else:
                number_steps = self.train_config["number_steps"]

            # Compute the annealing steps for MLM and CL loss weights
            self.mlm_weight_schedule = compute_loss_weights(
                number_steps,
                self.mlm_start_weight,
                self.mlm_end_weight,
                self.mlm_anneal_steps,
            )
            self.cl_weight_schedule = compute_loss_weights(
                number_steps,
                self.cl_start_weight,
                self.cl_end_weight,
                self.cl_anneal_steps,
            )

    def training_step(self, batch, batch_idx):
        if self.train_config["split_by_length"]:
            if self.trainer.local_rank % 4 == 0:
                batch = batch["long"]
            else:
                batch = batch["short"]

        transcripts, transcript_lengths, transcript_pairing_types = batch
        self.train_pairing_counter.update(transcript_pairing_types)

        x1, x2 = (
            transcripts[: transcripts.shape[0] // 2],
            transcripts[transcripts.shape[0] // 2 :],
        )
        t1_len, t2_len = (
            transcript_lengths[: transcript_lengths.shape[0] // 2],
            transcript_lengths[transcript_lengths.shape[0] // 2 :],
        )

        x1_aux = None
        x2_aux = None

        if self.predict_splice:
            x1_aux = x1[:, 4:, :]
            x2_aux = x2[:, 4:, :]
            x1 = x1[:, :4, :]
            x2 = x2[:, :4, :]

        if self.predict_masked:
            mask_x1, lab_x1 = self.mask_sequence(x1, t1_len)
            mask_x2, lab_x2 = self.mask_sequence(x2, t2_len)
        else:
            if self.model_config["n_tracks"] == 4:
                mask_x1 = x1[:, :4, :]
                mask_x2 = x2[:, :4, :]
            else:
                mask_x1 = x1
                mask_x2 = x2

        x1_rep = self(mask_x1, t1_len)
        x2_rep = self(mask_x2, t2_len)

        total_loss = 0
        metrics_to_log = {}

        if self.opt_config["loss_fn"] == "mlm":
            c_loss = 0
        else:
            current_cl_weight = self.cl_weight_schedule[self.trainer.global_step]
            self.log(
                "cl_weight",
                current_cl_weight,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=x1.shape[0],
            )

            if self.use_dual_heads:
                loss_ortho, metrics_ortho = self.ortho_loss_fn(
                    x1_rep["ortho_global"],
                    x2_rep["ortho_global"],
                    pairing_types=transcript_pairing_types,
                )
                loss_ss, metrics_ss = self.splice_self_loss_fn(
                    x1_rep["splice_self_global"],
                    x2_rep["splice_self_global"],
                    pairing_types=transcript_pairing_types,
                )
                c_loss = loss_ortho + loss_ss
                metrics_dict = {f"{k}_ortho": v for k, v in metrics_ortho.items()}
                metrics_dict.update(
                    {f"{k}_splice_self": v for k, v in metrics_ss.items()}
                )
                metrics_to_log["contrastive_loss_ortho"] = loss_ortho
                metrics_to_log["contrastive_loss_splice_self"] = loss_ss

            else:
                c_loss, metrics_dict = self.c_loss_fn(
                    x1_rep["global"],
                    x2_rep["global"],
                    pairing_types=transcript_pairing_types,
                )

            for metric_name, metric_value in metrics_dict.items():
                metrics_to_log[f"contrastive_{metric_name}"] = metric_value

            c_loss = current_cl_weight * c_loss
            metrics_to_log["contrastive_loss"] = c_loss

        total_loss += c_loss

        if self.predict_masked:
            masked_loss = self.compute_mask_loss(
                x1_rep["sequence"], x2_rep["sequence"], lab_x1, lab_x2
            )

            current_mlm_weight = self.mlm_weight_schedule[self.trainer.global_step]

            # Log the MLM weight to see how it changes over time
            self.log(
                "mlm_weight",
                current_mlm_weight,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=x1.shape[0],
            )

            masked_loss = current_mlm_weight * masked_loss

            # Finally, add the weighted MLM loss
            total_loss += masked_loss
            metrics_to_log["mask_loss"] = masked_loss

        if self.predict_splice:
            codon_loss = self.compute_single_track_bce_loss(
                x1_rep["codon"],
                x2_rep["codon"],
                torch.transpose(x1_aux[:, 0:1, :], 1, 2),
                torch.transpose(x2_aux[:, 0:1, :], 1, 2),
                t1_len,
                t2_len,
            )

            splice_loss = self.compute_single_track_bce_loss(
                x1_rep["splice"],
                x2_rep["splice"],
                torch.transpose(x1_aux[:, 1:2, :], 1, 2),
                torch.transpose(x2_aux[:, 1:2, :], 1, 2),
                t1_len,
                t2_len,
            )

            metrics_to_log["codon_loss"] = codon_loss
            metrics_to_log["splice_loss"] = splice_loss

            weight = min(1, max(0, ((self.trainer.global_step - 2000) / 4000)))
            total_loss += weight * (splice_loss + codon_loss)

        self._log_metrics("train", metrics_to_log, x1.shape[0])
        self.log(
            "global_step",
            self.trainer.global_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def on_train_epoch_end(self):
        # Log the pairing type counts for the training set
        self.log_dict(
            self.train_pairing_counter.compute(),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        # Manually reset the metric
        self.train_pairing_counter.reset()

    def on_train_epoch_start(self):
        # log learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        if self.opt_config["max_weight_decay"] == -1:
            return super().on_train_epoch_start()

        current_step = self.trainer.global_step

        if current_step <= self.opt_config["warmup_steps"]:
            return super().on_train_epoch_start()

        for param_group in self.trainer.optimizers[0].param_groups:
            # param_group name
            if param_group["name"] == "model":
                initial_weight_decay = self.opt_config["model_weight_decay"]
            else:
                initial_weight_decay = self.opt_config["projection_head_weight_decay"]

            final_weight_decay = self.opt_config["max_weight_decay"]
            # linearly interpolate weight decay
            increment = (
                (final_weight_decay - initial_weight_decay)
                * (current_step - self.opt_config["warmup_steps"])
                / (self.train_config["number_steps"] - self.opt_config["warmup_steps"])
            )
            param_group["weight_decay"] = initial_weight_decay + increment

            self.log(
                f"weight_decay_{param_group['name']}",
                param_group["weight_decay"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        return super().on_train_epoch_start()

    def validation_step(self, batch, batch_idx):
        transcripts, transcript_lengths, transcript_pairing_types = batch
        self.val_pairing_counter.update(transcript_pairing_types)

        x1, x2 = (
            transcripts[: transcripts.shape[0] // 2],
            transcripts[transcripts.shape[0] // 2 :],
        )
        t1_len, t2_len = (
            transcript_lengths[: transcript_lengths.shape[0] // 2],
            transcript_lengths[transcript_lengths.shape[0] // 2 :],
        )

        x1_aux = None
        x2_aux = None

        if self.predict_splice:
            x1_aux = x1[:, 4:, :]
            x2_aux = x2[:, 4:, :]
            x1 = x1[:, :4, :]
            x2 = x2[:, :4, :]

        mask_x1, lab_x1 = self.mask_sequence(x1, t1_len)
        mask_x2, lab_x2 = self.mask_sequence(x2, t2_len)

        x1_rep = self(mask_x1, t1_len)
        x2_rep = self(mask_x2, t2_len)

        total_loss = 0
        metrics_to_log = {}

        if self.opt_config["loss_fn"] == "mlm":
            c_loss = 0
        else:
            current_cl_weight = self.cl_weight_schedule[self.trainer.global_step]
            self.log(
                "cl_weight",
                current_cl_weight,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=x1.shape[0],
            )

            if self.use_dual_heads:
                loss_ortho, metrics_ortho = self.ortho_loss_fn(
                    x1_rep["ortho_global"],
                    x2_rep["ortho_global"],
                    pairing_types=transcript_pairing_types,
                )
                loss_ss, metrics_ss = self.splice_self_loss_fn(
                    x1_rep["splice_self_global"],
                    x2_rep["splice_self_global"],
                    pairing_types=transcript_pairing_types,
                )
                c_loss = loss_ortho + loss_ss
                metrics_dict = {f"{k}_ortho": v for k, v in metrics_ortho.items()}
                metrics_dict.update(
                    {f"{k}_splice_self": v for k, v in metrics_ss.items()}
                )
                metrics_to_log["contrastive_loss_ortho"] = loss_ortho
                metrics_to_log["contrastive_loss_splice_self"] = loss_ss
            else:
                c_loss, metrics_dict = self.c_loss_fn(
                    x1_rep["global"],
                    x2_rep["global"],
                    pairing_types=transcript_pairing_types,
                )

            metrics_to_log["contrastive_loss"] = c_loss
            for metric_name, metric_value in metrics_dict.items():
                metrics_to_log[f"contrastive_{metric_name}"] = metric_value

        total_loss += c_loss

        if self.predict_masked:
            masked_loss = self.compute_mask_loss(
                x1_rep["sequence"], x2_rep["sequence"], lab_x1, lab_x2
            )

            total_loss += masked_loss
            metrics_to_log["mask_loss"] = masked_loss

        if self.predict_splice:
            codon_loss = self.compute_single_track_bce_loss(
                x1_rep["codon"],
                x2_rep["codon"],
                torch.transpose(x1_aux[:, 0:1, :], 1, 2),
                torch.transpose(x2_aux[:, 0:1, :], 1, 2),
                t1_len,
                t2_len,
            )

            splice_loss = self.compute_single_track_bce_loss(
                x1_rep["splice"],
                x2_rep["splice"],
                torch.transpose(x1_aux[:, 1:2, :], 1, 2),
                torch.transpose(x2_aux[:, 1:2, :], 1, 2),
                t1_len,
                t2_len,
            )

            metrics_to_log["codon_loss"] = codon_loss
            metrics_to_log["splice_loss"] = splice_loss

            total_loss += splice_loss + codon_loss

        self._log_metrics("validation", metrics_to_log, x1.shape[0])
        return total_loss

    def on_validation_epoch_end(self):
        # Log the pairing type counts for the validation set
        self.log_dict(
            self.val_pairing_counter.compute(),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        # Manually reset the metric
        self.val_pairing_counter.reset()

    def configure_optimizers(self):
        """"""
        # Extracting optimizer configurations for readability
        model_lr = self.opt_config["model_lr"]
        projection_head_lr = self.opt_config["projection_head_lr"]
        model_wd = self.opt_config["model_weight_decay"]
        projection_head_wd = self.opt_config["projection_head_weight_decay"]

        # Separate parameter groups for model and projection head
        model_params = {
            "params": self.model.parameters(),
            "lr": model_lr,
            "weight_decay": model_wd,
            "name": "model",
        }

        all_params = [model_params]

        if self.opt_config["loss_fn"] != "mlm":
            if self.use_dual_heads:
                splice_self_head_params = {
                    "params": self.splice_self_head.parameters(),
                    "lr": projection_head_lr,
                    "weight_decay": projection_head_wd,
                    "name": "splice_self_head",
                }
                ortho_head_params = {
                    "params": self.ortho_head.parameters(),
                    "lr": projection_head_lr,
                    "weight_decay": projection_head_wd,
                    "name": "ortho_head",
                }
                all_params.append(splice_self_head_params)
                all_params.append(ortho_head_params)
            else:
                cl_projection_head_params = {
                    "params": self.contrastive_projection_head.parameters(),
                    "lr": projection_head_lr,
                    "weight_decay": projection_head_wd,
                    "name": "cl_projection_head",
                }
                all_params.append(cl_projection_head_params)

        if self.predict_masked:
            mask_projection_head_params = {
                "params": self.sequence_head.parameters(),
                "lr": projection_head_lr,
                "weight_decay": projection_head_wd,
                "name": "mask_projection_head",
            }
            all_params.append(mask_projection_head_params)

        optimizer = AdamW(all_params)

        total_steps = self.train_config["number_steps"]
        warmup_steps = self.opt_config["warmup_steps"]
        # add a cooldown to 0 as well
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
        }
        # Return optimizer and scheduler
        # Note: The 'scheduler' key in the return dictionary is a list
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
