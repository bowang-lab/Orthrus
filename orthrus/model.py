import pytorch_lightning as pl
from torch.optim import AdamW
from orthrus.layers import ProjectionHead
from orthrus.mamba import MixerModel
from orthrus.losses import DCLLoss, PairingTypeCounter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch import nn
import torch

from orthrus.dilated_resnet import dilated_small, dilated_medium
from orthrus.saluki import saluki_small, saluki_medium


class ContrastiveLearningModel(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        projector_config: dict,
        optimizer_config: dict,
        train_config: dict | None = None,
        data_config: dict | None = None,
    ):
        self.opt_config = optimizer_config
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config

        super(ContrastiveLearningModel, self).__init__()

        self.use_dual_heads = projector_config.get("use_dual_heads", False)

        self.model = self._create_model()

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

        if self.opt_config["loss_fn"] == "dcl":
            if self.use_dual_heads:
                self.ortho_loss_fn = DCLLoss(
                    self_weight=0.0,
                    splice_weight=0.0,
                    ortho_weight=optimizer_config.get("ortho_weight", 1.0),
                )
                self.splice_self_loss_fn = DCLLoss(
                    self_weight=optimizer_config.get("self_weight", 1.0),
                    splice_weight=1.0,
                    ortho_weight=0.0,
                )
            else:
                self.c_loss_fn = DCLLoss(
                    self_weight=optimizer_config.get("self_weight", 1.0),
                    splice_weight=1.0,
                    ortho_weight=optimizer_config.get("ortho_weight", 1.0),
                )
        else:
            raise NotImplementedError(
                f"Loss function {self.opt_config['loss_fn']} not implemented"
            )

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
        elif model_class == "resnet":
            resnet_type = self.model_config.get("resnet", "dilated_small")
            if resnet_type == "dilated_small":
                return dilated_small(
                    num_classes=self.model_config["num_classes"],
                    dropout_prob=self.model_config.get("dropout_prob", 0.1),
                    norm_type=self.model_config.get("norm_type", "batchnorm"),
                    add_shift=self.model_config.get("add_shift", True),
                    final_layer=False
                )
            elif resnet_type == "dilated_medium":
                return dilated_medium(
                    num_classes=self.model_config["num_classes"],
                    dropout_prob=self.model_config.get("dropout_prob", 0.1),
                    norm_type=self.model_config.get("norm_type", "batchnorm"),
                    add_shift=self.model_config.get("add_shift", True),
                    final_layer=False
                )
            else:
                raise ValueError(f"Unknown resnet type: {resnet_type}")
        elif model_class == "saluki":
            saluki_type = self.model_config.get("saluki", "saluki_small")
            if saluki_type == "saluki_small":
                return saluki_small(
                    num_classes=self.model_config["num_classes"],
                    seq_depth=self.model_config["n_tracks"],
                    final_layer=False
                )
            elif saluki_type == "saluki_medium":
                return saluki_medium(
                    num_classes=self.model_config["num_classes"],
                    seq_depth=self.model_config["n_tracks"],
                    final_layer=False
                )
            else:
                raise ValueError(f"Unknown saluki type: {saluki_type}")
        else:
            raise ValueError(f"Unknown model class: {model_class}")

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

    def forward(self, x, lengths=None):
        representation = self.model.representation(x, lengths)

        if self.use_dual_heads:
            splice_self_proj = self.splice_self_head(representation)
            ortho_proj = self.ortho_head(representation)
            return {"splice_self_global": splice_self_proj, "ortho_global": ortho_proj}
        else:
            projection = self.contrastive_projection_head(representation)
            return {"global": projection}

    def training_step(self, batch, batch_idx):
        if self.train_config["split_by_length"]:
            if self.trainer.local_rank == 0 or self.trainer.local_rank == -1:
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

        if self.model_config["n_tracks"] == 4:
            x1 = x1[:, :4, :]
            x2 = x2[:, :4, :]

        x1_projs = self(x1, t1_len)
        x2_projs = self(x2, t2_len)

        metrics_to_log = {}
        if self.use_dual_heads:
            # Calculate losses using the specialized heads and loss functions
            loss_ortho, metrics_ortho = self.ortho_loss_fn(
                x1_projs["ortho_global"],
                x2_projs["ortho_global"],
                pairing_types=transcript_pairing_types,
            )
            loss_ss, metrics_ss = self.splice_self_loss_fn(
                x1_projs["splice_self_global"],
                x2_projs["splice_self_global"],
                pairing_types=transcript_pairing_types,
            )

            total_loss = loss_ortho + loss_ss

            # Log metrics
            metrics_to_log["contrastive_loss_ortho"] = loss_ortho
            for k, v in metrics_ortho.items():
                metrics_to_log[f"contrastive_{k}_ortho"] = v

            metrics_to_log["contrastive_loss_splice_self"] = loss_ss
            for k, v in metrics_ss.items():
                metrics_to_log[f"contrastive_{k}_splice_self"] = v
        else:
            loss, metrics_dict = self.c_loss_fn(
                x1_projs["global"],
                x2_projs["global"],
                pairing_types=transcript_pairing_types,
            )
            total_loss = loss
            for metric_name, metric_value in metrics_dict.items():
                metrics_to_log[f"contrastive_{metric_name}"] = metric_value

        metrics_to_log["contrastive_loss"] = total_loss
        self._log_metrics("train", metrics_to_log, x1.shape[0])

        self.log(
            "global_step",
            self.trainer.global_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "contrastive_train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=x1.shape[0],
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

        if self.model_config["n_tracks"] == 4:
            x1 = x1[:, :4, :]
            x2 = x2[:, :4, :]

        x1_projs = self(x1, t1_len)
        x2_projs = self(x2, t2_len)

        metrics_to_log = {}
        if self.use_dual_heads:
            # Calculate losses using the specialized heads and loss functions
            loss_ortho, metrics_ortho = self.ortho_loss_fn(
                x1_projs["ortho_global"],
                x2_projs["ortho_global"],
                pairing_types=transcript_pairing_types,
            )
            loss_ss, metrics_ss = self.splice_self_loss_fn(
                x1_projs["splice_self_global"],
                x2_projs["splice_self_global"],
                pairing_types=transcript_pairing_types,
            )

            total_loss = loss_ortho + loss_ss
            metrics_dict = {f"{k}_ortho": v for k, v in metrics_ortho.items()}
            metrics_dict.update({f"{k}_splice_self": v for k, v in metrics_ss.items()})
            metrics_to_log["contrastive_loss_ortho"] = loss_ortho
            metrics_to_log["contrastive_loss_splice_self"] = loss_ss
        else:
            total_loss, metrics_dict = self.c_loss_fn(
                x1_projs["global"],
                x2_projs["global"],
                pairing_types=transcript_pairing_types,
            )

        metrics_to_log["contrastive_loss"] = total_loss
        for metric_name, metric_value in metrics_dict.items():
            metrics_to_log[f"contrastive_{metric_name}"] = metric_value

        self._log_metrics("validation", metrics_to_log, x1.shape[0])

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
        model_weight_decay = self.opt_config["model_weight_decay"]
        projection_head_weight_decay = self.opt_config["projection_head_weight_decay"]

        all_params = []

        # Separate parameter groups for model and projection head
        model_params = {
            "params": self.model.parameters(),
            "lr": model_lr,
            "weight_decay": model_weight_decay,
            "name": "model",
            # "betas": [0.9, 0.95]
        }

        all_params.append(model_params)

        if self.use_dual_heads:
            splice_self_head_params = {
                "params": self.splice_self_head.parameters(),
                "lr": projection_head_lr,
                "weight_decay": projection_head_weight_decay,
                "name": "splice_self_head",
            }
            all_params.append(splice_self_head_params)

            ortho_head_params = {
                "params": self.ortho_head.parameters(),
                "lr": projection_head_lr,
                "weight_decay": projection_head_weight_decay,
                "name": "ortho_head",
            }
            all_params.append(ortho_head_params)
        else:
            projection_head_params = {
                "params": self.contrastive_projection_head.parameters(),
                "lr": projection_head_lr,
                "weight_decay": projection_head_weight_decay,
                "name": "cl_projection_head",
            }
            all_params.append(projection_head_params)

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
        # Note: The 'scheduler' key in the return dictionary is expected to be a list
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
