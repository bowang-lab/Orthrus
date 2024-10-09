import os

from absl import app
from absl import flags
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import MulticlassAccuracy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torch.nn as nn
import torch
import wandb

from orthrus.dilated_resnet import dilated_small, not_dilated_small
from orthrus.mamba import MixerModel
from orthrus.config_utils import Config, save_config
from orthrus.rna_task_dataset import create_dataloaders
from orthrus.layers import ProjectionHead
from orthrus.util import PearsonR, SpearmanR
from orthrus.linear_probe_eval import load_model
from orthrus.saluki import SalukiModel


class RNATaskModel(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        train_config: dict,
        data_config: dict,
        projector_config: dict,
    ):
        self.opt_config = optimizer_config
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.projector_config = projector_config
        self.explicit_weight_decay = optimizer_config["explicit_weight_decay"]
        self.task = data_config["task"]
        assert self.task in ["regression", "classification"]

        super(RNATaskModel, self).__init__()

        if (
            model_config["model_class"] == "resnet"
            and model_config["resnet"] in ["dilated_small", "not_dilated_small"]
            and model_config["pooling_layer"] == "max_pool"
            and model_config["global_pooling_layer"] == "dynamic_avgpool"
        ):
            self.pooling_length = 32
        else:
            self.pooling_length = -1

        cond1 = model_config["run_name"] is not None
        cond2 = model_config["model_repository"] is not None

        # Load from a pre-trained model
        if cond1 and cond2:
            model_path = os.path.join(
                model_config["model_repository"],
                model_config["run_name"],
            )
            assert os.path.exists(model_path), f"{model_path} does not exist"
            model_name = model_config["model_name"]

            self.model = load_model(
                repository=model_config["model_repository"],
                run_name=model_config["run_name"],
                model_name=model_name,
                n_tracks=data_config["n_tracks"],
            )

        # Train a new model convnet
        elif model_config["model_class"] == "resnet":
            if model_config["resnet"] == "dilated_small":
                resnet = dilated_small
            elif model_config["resnet"] == "not_dilated_small":
                resnet = not_dilated_small
            else:
                raise ValueError
            self.model = resnet(
                num_classes=model_config["num_classes"],
                dropout_prob=model_config["dropout_prob"],
                norm_type=model_config["norm_type"],
                kernel_size=model_config["kernel_size"],
                pooling_layer=model_config["pooling_layer"],
                add_shift=model_config["add_shift"],
                increase_dilation=model_config["increase_dilation"],
                global_pooling_layer=model_config["global_pooling_layer"],
            )

        # Train a new model with Mamba
        elif model_config["model_class"] == "ssm":
            self.model = MixerModel(
                model_config["ssm_model_dim"],
                model_config["ssm_n_layers"],
                data_config["n_tracks"],
            )

        elif model_config["model_class"] == "saluki":
            self.model = SalukiModel(
                seq_depth=data_config["n_tracks"],
                add_shift=model_config["add_shift"],
            )
        else:
            raise ValueError

        self.projection_head = ProjectionHead(
            input_features=projector_config["representation_dim"],
            projection_body=projector_config["projection_body"],
            projection_head_size=projector_config["projection_head_size"],
            norm_type=projector_config["projection_norm_type"],
            output_bias=projector_config["output_bias"],
            n_layers=projector_config["n_layers"],
            output_sigmoid=projector_config["output_sigmoid"],
        )

        if self.task == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
            self.auroc = BinaryAUROC()
            self.accuracy = MulticlassAccuracy(
                num_classes=projector_config["projection_head_size"]
            )

            self.metrics = {"auroc": self.auroc, "accuracy": self.accuracy}
        else:
            self.loss_fn = nn.MSELoss()
            self.pearsonr = PearsonR()
            self.spearmanr = SpearmanR()
            self.metrics = {"pearsonr": self.pearsonr, "spearmanr": self.spearmanr}

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
        if self.opt_config["optimizer"] == "adam":
            optimizer = Adam([model_params, projection_head_params])
        elif self.opt_config["optimizer"] == "adamw":
            optimizer = AdamW([model_params, projection_head_params])

        total_steps = self.train_config["number_steps"]
        warmup_steps = self.opt_config["warmup_steps"]

        if self.opt_config["scheduler"] == "warmup_cosine":
            # add a cooldown to 0 as well
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                    CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
                ],
                milestones=[warmup_steps],
            )
        elif self.opt_config["scheduler"] == "linear":
            lr_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=total_steps
            )
        elif self.opt_config["scheduler"] == "warmup":
            lr_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
        else:
            lr_scheduler = None

        if lr_scheduler is not None:
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
        else:
            return {"optimizer": optimizer}

    def on_train_epoch_start(self):
        # log learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        for metric in self.metrics.values():
            metric.reset()
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        for metric in self.metrics.values():
            metric.reset()
        return super().on_validation_epoch_start()

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        if y.ndim == 1:
            y = y.unsqueeze(1)

        if self.pooling_length != -1:
            pool_length = lengths // self.pooling_length + 1
            y_hat = self(X, pool_length)

        elif self.model_config["model_class"] == "ssm":
            y_hat = self(X, lengths)

        else:
            y_hat = self(X)

        for metric in self.metrics.values():
            if self.task == "classification":
                metric.update(y_hat, y.to(torch.int64))
            else:
                metric.update(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=X.shape[0],
        )
        for metric_name, metric in self.metrics.items():
            self.log(
                f"training_{metric_name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                batch_size=X.shape[0],
            )

        if self.explicit_weight_decay:
            ewd = self.explicit_weight_decay
            with torch.no_grad():
                for param in self.parameters():
                    param.data = param.data - ewd * param.data
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, lengths = batch
        if y.ndim == 1:
            y = y.unsqueeze(1)

        if self.pooling_length != -1:
            pool_length = lengths // self.pooling_length + 1
            y_hat = self(X, pool_length)
        elif self.model_config["model_class"] == "ssm":
            y_hat = self(X, lengths)
        else:
            y_hat = self(X)

        for metric in self.metrics.values():
            if self.task == "classification":
                metric.update(y_hat, y.to(torch.int64))
            else:
                metric.update(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=X.shape[0],
        )
        for metric_name, metric in self.metrics.items():
            self.log(
                f"validation_{metric_name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                batch_size=X.shape[0],
            )
        return loss

    def forward(self, x, lengths=None):
        representation = self.model.representation(x, lengths)
        x = self.projection_head(representation)
        return x


def create_run_name(config):
    run_name = f"{config.data['dataset_name']}_"

    if config.model["model_class"] == "ssm":
        run_name += (
            "ssm_"
            f"{config.model['ssm_n_layers']}_"
            f"{config.model['ssm_model_dim']}_"
            f"lr{config.optimizer['model_lr']}_"
            f"wd{config.optimizer['model_weight_decay']}_"
            f"seed{config.train['rand_seed']}"
        )
    elif config.model["model_class"] == "saluki":
        run_name += (
            "saluki_"
            f"lr{config.optimizer['model_lr']}_"
            f"wd{config.optimizer['model_weight_decay']}_"
            f"seed{config.train['rand_seed']}"
        )

    else:
        run_name += (
            f"{config.model['resnet']}_"
            f"k{config.model['kernel_size']}_"
            f"lr{config.optimizer['model_lr']}_"
            f"wd{config.optimizer['model_weight_decay']}_"
            f"seed{config.train['rand_seed']}"
        )

    if "note" in config.train and config.train["note"] != "":
        run_name += f"_{config.train['note']}"

    print(f"Run name: {run_name}")
    return run_name


def update_wandb_logger(wandb_logger, config):
    wandb_logger.experiment.config.update(config.model)
    wandb_logger.experiment.config.update(config.optimizer)
    wandb_logger.experiment.config.update(config.data)
    wandb_logger.experiment.config.update(config.projector)
    wandb_logger.experiment.config.update(config.train)

    # update wandb_logger with the following FLAGS variables
    wandb_logger.experiment.config.update({"model_config": FLAGS.model_config})
    wandb_logger.experiment.config.update({"train_config": FLAGS.train_config})
    wandb_logger.experiment.config.update({"projector_config": FLAGS.projector_config})
    wandb_logger.experiment.config.update({"data_config": FLAGS.data_config})
    wandb_logger.experiment.config.update({"optimizer_config": FLAGS.optimizer_config})


def main(argv):
    config = Config("./rna_task_config", FLAGS)
    torch.set_float32_matmul_precision("medium")

    if FLAGS.seed_override is not None:
        config.train["rand_seed"] = FLAGS.seed_override

    pl.seed_everything(config.train["rand_seed"])

    train, val, test = create_dataloaders(
        split_type=config.data["split_type"],
        dataset_name=config.data["dataset_name"],
        train_batch_size=config.train["gpu_batch_sizes"][0],
        val_batch_size=config.train["gpu_batch_sizes"][1],
        test_batch_size=config.train["gpu_batch_sizes"][1],
        species=config.data["species"],
        n_tracks=config.data["n_tracks"],
        seed=config.train["rand_seed"],
        data_dir=config.data["data_dir"],
        verbose=config.train["verbose"],
        subset_fraction=config.data["subset_fraction"],
        subset_n_samples=config.data["subset_n_samples"],
    )
    run_name = create_run_name(config)
    run_path = os.path.join(config.train["wandb_run_dir"], run_name)
    checkpoint_path = save_config(
        run_path,
        config.model,
        config.optimizer,
        config.data,
        config.projector,
        config.train,
    )
    wandb_logger = WandbLogger(
        name=run_name,
        save_dir=config.train["wandb_run_dir"],
        project="rna_task2",
    )
    # update wandb config
    update_wandb_logger(wandb_logger, config)

    model = RNATaskModel(
        model_config=config.model,
        optimizer_config=config.optimizer,
        train_config=config.train,
        data_config=config.data,
        projector_config=config.projector,
    )

    # save the model according to val loss
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}",
        monitor="validation_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        dirpath=checkpoint_path,
        verbose=True,
    )
    if config.model["model_class"] == "resnet":
        train_strat = DDPStrategy(find_unused_parameters=True)
    else:
        train_strat = "ddp"

    trainer_params = {
        "accelerator": "gpu",
        "strategy": train_strat,
        "devices": 1,
        "num_nodes": 1,
        "logger": wandb_logger,
        "precision": "16-mixed" if config.train["mixed_precision"] else 32,
        "callbacks": [model_checkpoint],
        "gradient_clip_val": config.optimizer["gradient_clip_val"],
        "gradient_clip_algorithm": config.optimizer["gradient_clip_algorithm"],
        "max_steps": config.train["number_steps"],
        # "val_check_interval": 80,
    }
    trainer = pl.Trainer(**trainer_params)

    _ = trainer.validate(model, val)
    trainer.fit(
        model, train_dataloaders=train, val_dataloaders=val, ckpt_path=checkpoint_path
    )

    # load model from model_checkpoint.best_model_path
    top_model = RNATaskModel.load_from_checkpoint(
        model_checkpoint.best_model_path,
        model_config=config.model,
        optimizer_config=config.optimizer,
        train_config=config.train,
        data_config=config.data,
        projector_config=config.projector,
    )

    validation_results = trainer.validate(top_model, val)
    for metric, value in validation_results[0].items():
        wandb.log(
            {
                f"best_val_results/{metric}": value,
                "trainer/global_step": trainer.global_step,
            }
        )

    test_results = trainer.validate(top_model, test)
    for metric, value in test_results[0].items():
        # log the results
        wandb.log(
            {
                f"best_test_results/{metric}": value,
                "trainer/global_step": trainer.global_step,
            }
        )


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # Misc flags
    flags.DEFINE_string("note", "", "Note for WandB")
    flags.DEFINE_bool("mini", False, "Load a miniature version of the dataset")

    # Specify configs to use
    flags.DEFINE_string("data_config", "rna_hl", "Data config.")
    flags.DEFINE_string("model_config", "resnet_base", "Model config.")
    flags.DEFINE_string("projector_config", "default_256", "Projector config.")
    flags.DEFINE_string("optimizer_config", "default", "Optimizer config.")
    flags.DEFINE_string("train_config", "bs_64", "Train config.")
    flags.DEFINE_integer("seed_override", None, "Training seed to override config.")

    app.run(main)
