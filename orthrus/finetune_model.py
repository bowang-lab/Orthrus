import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.optim import AdamW, Adam

from torchmetrics.classification import AUROC, Accuracy, AveragePrecision
from orthrus.eval_utils import load_model, PearsonR, SpearmanR
from orthrus.layers import ProjectionHead
from orthrus.saluki import saluki_small
from orthrus.dilated_resnet import DilatedResnet
from orthrus.mamba import MixerModel


class RNATaskModel(pl.LightningModule):

    def __init__(
        self,
        model_config: dict,
        projector_config: dict,
        optimizer_config: dict,
        train_config: dict,
        data_config: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config", "optimizer_config",
                                          "train_config", "data_config", "projector_config"])
        self.opt_config = optimizer_config
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.projector_config = projector_config

        self.explicit_l2_weight_decay = optimizer_config.get("explicit_l2_weight_decay", 0.0)
        self.task = data_config["task"]
        assert self.task in ["regression", "classification", "multilabel"], \
            f"Unsupported task: {self.task}. Must be 'regression', 'classification', or 'multilabel'."

        self.pooling_length = -1  # if you need specialized pooling
        self._init_base_model()
        self._init_projection_head()
        self._init_metrics()

    def _init_base_model(self):
        """Initialize the base model (Saluki, ResNet, or load from checkpoint)."""
        model_class = self.model_config["model_class"]

        # 1) Load from a pre-trained model if run_name + model_repository are specified
        if (self.model_config.get("run_path") is not None
                and self.model_config.get("checkpoint") is not None):

            run_path = self.model_config["run_path"]
            checkpoint = self.model_config["checkpoint"]

            self.model = load_model(
                run_path=run_path,
                checkpoint_name=checkpoint,
            )

            if self.model_config.get("freeze_layers", None) is not None:
                freeze_layers = self.model_config["freeze_layers"]

                for layer_num in freeze_layers:
                    for param in self.model.layers[layer_num].parameters():
                        param.requires_grad = False
                    print(f"Freezing layer {layer_num}")

            if model_class == "resnet":
                # set special pooling length if needed
                if (self.model_config["pooling_layer"] == "max_pool"
                    and self.model_config["global_pooling_layer"] == "dynamic_avgpool"):

                    self.pooling_length = 32

        # 2) Otherwise, create from scratch
        elif model_class == "resnet":

            self.model = DilatedResnet(
                num_classes=self.data_config["num_classes"],
                layer_params=self.model_config["layer_params"],
                dilation_params=self.model_config["dilation_params"],
                dropout_prob=self.model_config["dropout_prob"],
                pooling_layer=self.model_config["pooling_layer"],
                kernel_size=self.model_config["kernel_size"],
                filter_nums=self.model_config["filter_nums"],
                norm_type=self.model_config["norm_type"],
                increase_dilation=self.model_config["increase_dilation"],
                add_shift=self.model_config["add_shift"],
                n_tracks=self.data_config["n_tracks"],
                global_pooling_layer=self.model_config["global_pooling_layer"],
            )

            # set special pooling length if needed
            if (self.model_config["pooling_layer"] == "max_pool"
                and self.model_config["global_pooling_layer"] == "dynamic_avgpool"):

                self.pooling_length = 32

        elif model_class == "saluki":
            self.model = saluki_small(
                seq_depth=self.data_config["n_tracks"],
                add_shift=self.model_config["add_shift"],
                final_layer=False,
            )

        elif model_class == "ssm":

            self.model = MixerModel(
                d_model=self.model_config["ssm_model_dim"],
                n_layer=self.model_config["ssm_n_layers"],
                input_dim=self.data_config["n_tracks"],
                bidirectional=self.model_config.get("bidirectional", None),
                bidirectional_strategy=self.model_config.get("bidirectional_strategy", "add"),
                bidirectional_weight_tie=self.model_config.get("bidirectional_weight_tie", True),
                gradient_checkpointing=self.model_config.get("gradient_checkpointing", False),
            )

        else:
            raise ValueError(f"Unsupported model_class: {model_class}")


    def _init_projection_head(self):
        """Projection head for final output (classification or regression)."""
        self.projection_head = ProjectionHead(
            input_features=self.projector_config["representation_dim"],
            projection_body=self.projector_config["projection_body"],
            projection_head_size=self.data_config["num_classes"],
            norm_type=self.projector_config["projection_norm_type"],
            output_bias=self.projector_config["output_bias"],
            n_layers=self.projector_config["n_layers"],
            output_sigmoid=self.projector_config["output_sigmoid"],
        )

        linear_layers = [layer for layer in self.projection_head.layers if isinstance(layer, nn.Linear)]

        # check if we specified to zero-init layers
        if self.projector_config.get("zero_init_layers", None) is not None:
            for idx in self.projector_config["zero_init_layers"]:
                layer = linear_layers[idx]
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


        # check if we specified to random-init layers
        if self.projector_config.get("random_init_layers", None) is not None:
            for idx in self.projector_config["random_init_layers"]:
                layer = linear_layers[idx]
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _init_metrics(self):
        """Initialize metrics for classification or regression."""
        if self.task == "classification":

            if self.data_config["num_classes"] >= 2:
                self.loss_fn = nn.CrossEntropyLoss()
                self.accuracy = Accuracy(task="multiclass", num_classes=self.data_config["num_classes"])
                self.auroc = AUROC(task="multiclass", num_classes=self.data_config["num_classes"])
                self.auprc = AveragePrecision(task="multiclass", num_classes=self.data_config["num_classes"])

            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
                self.accuracy = Accuracy(task="binary")
                self.auroc = AUROC(task="binary")
                self.auprc = AveragePrecision(task="binary")

            self.metrics = {
                "auroc": self.auroc,
                "accuracy": self.accuracy,
                "auprc": self.auprc
            }
        elif self.task == "multilabel":
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.accuracy = Accuracy(task="multilabel", num_labels=self.data_config["num_classes"])

            self.auroc = AUROC(
                task="multilabel",
                num_labels=self.data_config["num_classes"],
                average="micro"
            )
            self.auprc = AveragePrecision(
                task="multilabel",
                num_labels=self.data_config["num_classes"],
                average="micro"
            )

            self.metrics = {
                "auroc": self.auroc,
                "accuracy": self.accuracy,
                "auprc": self.auprc
            }
        else:
            self.loss_fn = nn.MSELoss()
            self.pearsonr = PearsonR()
            self.spearmanr = SpearmanR()
            self.metrics = {
                "pearsonr": self.pearsonr,
                "spearmanr": self.spearmanr
            }

    def forward(self, x, lengths=None):
        """
        Forward pass:
         1) get representation from the base model
         2) project to final dimension with the projection head
        """
        representation = self.model.representation(x, lengths)
        x = self.projection_head(representation)
        return x

    def _compute_predictions(self, batch):
        """Handles specialized forward calls depending on model details."""
        X, y, lengths = batch

        # If we need special pooling length for e.g. ResNet with dynamic pooling
        if self.pooling_length != -1:
            pool_length = lengths // self.pooling_length + 1
            preds = self(X, pool_length)
        elif self.model_config["model_class"] == "ssm":
            preds = self(X, lengths)
        else:
            preds = self(X)

        return preds, y

    def training_step(self, batch, batch_idx):
        preds, y = self._compute_predictions(batch)

        for metric in self.metrics.values():
            if self.task == "classification" or self.task == "multilabel":
                metric.update(preds, y.to(torch.int64))
            else:
                metric.update(preds, y)

        loss = self.loss_fn(preds, y)

        self.log("training_loss", loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=batch[0].shape[0])

        # Log metrics
        for metric_name, metric in self.metrics.items():
            self.log(
                f"training_{metric_name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch[0].shape[0]
            )

        # optional explicit weight decay
        if self.explicit_l2_weight_decay > 0.0:
            with torch.no_grad():
                for param in self.parameters():
                    if param.requires_grad: # in the case where we have frozen layers
                        param.data = param.data - self.explicit_l2_weight_decay * param.data

        return loss

    def validation_step(self, batch, batch_idx):
        preds, y = self._compute_predictions(batch)

        for metric in self.metrics.values():
            if self.task == "classification" or self.task == "multilabel":
                metric.update(preds, y.to(torch.int64))
            else:
                metric.update(preds, y)

        loss = self.loss_fn(preds, y)

        self.log("validation_loss", loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=batch[0].shape[0])

        for metric_name, metric in self.metrics.items():
            self.log(
                f"validation_{metric_name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch[0].shape[0]
            )
        return loss


    def test_step(self, batch, batch_idx):
        preds, y = self._compute_predictions(batch)

        for metric in self.metrics.values():
            if self.task == "classification" or self.task == "multilabel":
                metric.update(preds, y.to(torch.int64))
            else:
                metric.update(preds, y)

        loss = self.loss_fn(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 logger=True, batch_size=batch[0].shape[0])

        for metric_name, metric in self.metrics.items():
            self.log(
                f"test_{metric_name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch[0].shape[0]
            )
        return loss

    def on_train_epoch_start(self):
        # Log learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        for metric in self.metrics.values():
            metric.reset()
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        for metric in self.metrics.values():
            metric.reset()
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        for metric in self.metrics.values():
            metric.reset()
        return super().on_test_epoch_start()

    def configure_optimizers(self):
        """
        Set up optimizer + (optional) LR scheduler.
        """
        optim_name = self.opt_config.get("optimizer", "adamw")
        model_lr = self.opt_config["model_lr"]
        proj_lr = self.opt_config["projection_head_lr"]
        model_wd = self.opt_config["model_weight_decay"]
        proj_wd = self.opt_config["projection_head_weight_decay"]

        model_params = {
            "params": self.model.parameters(),
            "lr": model_lr,
            "weight_decay": model_wd,
            "name": "model",
        }
        proj_params = {
            "params": self.projection_head.parameters(),
            "lr": proj_lr,
            "weight_decay": proj_wd,
            "name": "projection_head",
        }

        if optim_name == "adam":
            optimizer = Adam([model_params, proj_params])
        else:  # default to AdamW
            optimizer = AdamW([model_params, proj_params])

        total_steps = self.train_config["number_steps"]
        warmup_steps = self.opt_config.get("warmup_steps", 0)
        scheduler_name = self.opt_config.get("scheduler", "warmup_cosine")

        if scheduler_name == "warmup_cosine":
            if total_steps <= warmup_steps:
                print(f"Total steps ({total_steps}) <= warmup steps ({warmup_steps}). Using linear warmup for all steps.")
                lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=total_steps)
            else:
                lr_scheduler = SequentialLR(
                    optimizer,
                    schedulers=[
                        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
                    ],
                    milestones=[warmup_steps],
                )
        elif scheduler_name == "linear":
            lr_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=total_steps
            )
        elif scheduler_name == "warmup":
            lr_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
        elif scheduler_name == "exponential":
            gamma = self.opt_config.get("exponential_gamma", 0.95)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        else:
            lr_scheduler = None

        if lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    # REQUIRED: The scheduler instance
                    "scheduler": lr_scheduler, # expected to be a list
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "step",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}