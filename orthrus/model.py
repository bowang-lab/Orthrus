import pytorch_lightning as pl
from torch.optim import AdamW
from orthrus.layers import ProjectionHead
from orthrus.mamba import MixerModel
from orthrus.losses import DCLLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


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

        self.model = MixerModel(
            model_config["ssm_model_dim"],
            model_config["ssm_n_layers"],
            model_config["n_tracks"],
        )

        self.projection_head = ProjectionHead(
            input_features=projector_config["representation_dim"],
            projection_body=projector_config["projection_body"],
            projection_head_size=projector_config["projection_head_size"],
            norm_type=projector_config["projection_norm_type"],
        )

        self.loss_fn = DCLLoss()

    def forward(self, x, lengths=None):
        representation = self.model.representation(x, lengths)
        x = self.projection_head(representation)
        return x

    def training_step(self, batch, batch_idx):
        if self.train_config["split_by_length"]:
            if self.trainer.local_rank == 0:
                batch = batch["long"]
            else:
                batch = batch["short"]

        x1, x2, t1_len, t2_len = batch

        if self.model_config["n_tracks"] == 4:
            x1 = x1[:, :4, :]
            x2 = x2[:, :4, :]

        x1_rep = self(x1, t1_len)
        x2_rep = self(x2, t2_len)

        loss, metrics_dict = self.loss_fn(x1_rep, x2_rep)

        self.log(
            "global_step",
            self.trainer.global_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=x1.shape[0],
        )

        for metric_name, metric_value in metrics_dict.items():
            self.log(
                f"train_{metric_name}",
                metric_value,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                batch_size=x1.shape[0],
            )
        return loss

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
        x1, x2, t1_len, t2_len = batch

        if self.model_config["n_tracks"] == 4:
            x1 = x1[:, :4, :]
            x2 = x2[:, :4, :]

        x1_rep = self(x1, t1_len)
        x2_rep = self(x2, t2_len)

        loss, metrics_dict = self.loss_fn(x1_rep, x2_rep)

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=x1.shape[0],
        )

        for metric_name, metric_value in metrics_dict.items():
            self.log(
                f"validation_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=x1.shape[0],
            )

    def configure_optimizers(self):
        """"""
        # Extracting optimizer configurations for readability
        model_lr = self.opt_config["model_lr"]
        projection_head_lr = self.opt_config["projection_head_lr"]
        model_weight_decay = self.opt_config["model_weight_decay"]
        projection_head_weight_decay = self.opt_config["projection_head_weight_decay"]

        # Separate parameter groups for model and projection head
        model_params = {
            "params": self.model.parameters(),
            "lr": model_lr,
            "weight_decay": model_weight_decay,
            "name": "model",
            # "betas": [0.9, 0.95]
        }
        projection_head_params = {
            "params": self.projection_head.parameters(),
            "lr": projection_head_lr,
            "weight_decay": projection_head_weight_decay,
            "name": "projection_head",
            # "betas": [0.9, 0.95]
        }
        optimizer = AdamW([model_params, projection_head_params])
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
