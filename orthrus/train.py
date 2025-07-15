import json
import re
import os
# import sys
import torch

from absl import app
from absl import flags

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.strategies import DDPStrategy

from orthrus.config_utils import Config, save_config
from orthrus.data_loader import init_data_loader


def generate_run_name(config, flags):
    model_config = config.model
    optimizer_config = config.optimizer
    train_config = config.train
    data_config_name = flags.data_config

    model_class = model_config.get("model_class", "ssm")

    # Common parameters
    lr = optimizer_config['model_lr']
    wd = optimizer_config['model_weight_decay']
    mask_p = max(train_config.get("mask_prop", 0), config.data.get("proportion_to_mask", 0))
    n_tracks = model_config.get('n_tracks', 6)

    # Model-specific parameters
    if model_class == "ssm":
        base_name = (
            "ssm_"
            + f"{n_tracks}t_"
            + f"{model_config['ssm_n_layers']}_"
            + f"{model_config['ssm_model_dim']}_"
            + f"lr{lr}_"
            + f"wd{wd}_"
            + f"mask{mask_p}_"
            + f"{data_config_name}"
        )
        if model_config.get('bidirectional'):
            base_name += f"_bidirectional_{model_config['bidirectional']}"
    elif model_class == "resnet":
        resnet_type = model_config.get("resnet", "dilated_small")
        base_name = (
            f"resnet_{resnet_type}_"
            + f"{n_tracks}t_"
            + f"lr{lr}_"
            + f"wd{wd}_"
            + f"mask{mask_p}_"
            + f"{data_config_name}"
        )
    elif model_class == "saluki":
        saluki_type = model_config.get("saluki", "saluki_small")
        base_name = (
            f"saluki_{saluki_type}_"
            + f"{n_tracks}t_"
            + f"lr{lr}_"
            + f"wd{wd}_"
            + f"mask{mask_p}_"
            + f"{data_config_name}"
        )
    else:
        raise ValueError(f"Unknown model class for run name generation: {model_class}")

    # Append suffixes
    if model_config.get('predict_masked') and optimizer_config.get('loss_fn') != 'mlm':
        run_name = base_name + "_mlm"
    elif optimizer_config.get('loss_fn') == 'mlm':
        run_name = base_name + "_mlm_only"
    else:
        run_name = base_name
    
    if flags.optimizer_config == "anneal":
        run_name += "_wd-anneal"

    if "note" in train_config and train_config["note"] != "":
        run_name += f"_{train_config['note']}"
        
    return run_name


def main(argv):
    config = Config("./config", FLAGS)

    pl.seed_everything(config.train["rand_seed"])

    warmup_steps = min([10_000, 0.1 * config.train["number_steps"]])
    config.optimizer["warmup_steps"] = warmup_steps

    # Initialize wandb logger
    run_name = generate_run_name(config, FLAGS)

    if FLAGS.optimizer_config == "anneal":
        run_name += "_wd-anneal"

    if "note" in config.train and config.train["note"] != "":
        run_name += f"_{config.train['note']}"

    run_path = os.path.join(config.train["wandb_run_dir"], run_name)

    # Check if checkpoint at run_path exists else start from scratch
    # iterate over checkpoints in "{epoch}-{step}.ckpt" and find latest
    checkpoint_path = save_config(
        run_path,
        config.model,
        config.optimizer,
        config.data,
        config.projector,
        config.train
    )

    if config.train['logging']:
        # Tells WandB that run resumes previous run from checkpoint
        logged_config = config.wandb_repr()
        wandb_kwargs = {"config": logged_config}
        if checkpoint_path:
            wandb_kwargs["resume"] = True

        # Initialize wandb logger
        wandb_logger = WandbLogger(
            name=run_name,
            save_dir=run_path,
            project="Orthrus Finetuning",
            tags = ["pretraining"],
            **wandb_kwargs
        )

    print(f"Logging to: {run_path}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    loss_to_monitor = "validation/contrastive_loss"

    if config.train["mask_prop"] > 0:
        from orthrus.mlm_model import ContrastiveLearningModel as Model

        if config.optimizer["loss_fn"] == "mlm":
            print("Using MLM only model")
            loss_to_monitor = "validation/mask_loss"
        else:
            print("Using Contrastive + MLM model")
    else:
        from orthrus.model import ContrastiveLearningModel as Model
        print("Using Contrastive only model")

    short_loader, long_loader, val_data_loader = init_data_loader(
        data_config=config.data,
        train_config=config.train,
    )

    train_data_loader = CombinedLoader(
        {"short": short_loader, "long": long_loader},
        mode="max_size_cycle"
    )

    print(f"Total number of batches per GPU: {max(len(short_loader), len(long_loader))/(4 * FLAGS.nodes)}")

    model = Model(
        model_config=config.model,
        projector_config=config.projector,
        optimizer_config=config.optimizer,
        train_config=config.train,
        data_config=config.data
    )

    if config.train['logging']:
        wandb_logger.watch(model, log_freq=500)

    strategy = "ddp" # DDPStrategy(static_graph=True) #FSDPStrategy(auto_wrap_policy=auto_wrap_policy, sharding_strategy = "SHARD_GRAD_OP", activation_checkpointing_policy=auto_wrap_policy) # FSDPStrategy() # "ddp"

    trainer_params = {
        "accelerator": "gpu",
        "strategy": strategy,
        "devices": -1, # use all GPUs
        "num_nodes": FLAGS.nodes,
        "precision": "bf16-mixed" if config.train["mixed_precision"] else 32,
        "callbacks": [
            ModelCheckpoint(
                filename="{epoch}-{step}",
                every_n_epochs=1,
                dirpath=run_path,
                save_top_k=-1,
            ),
            ModelCheckpoint(
                filename="{epoch}-{step}-2k",
                every_n_train_steps=2000,
                save_top_k=-1,
                dirpath=run_path
            ),
            ModelCheckpoint(
                filename="best-{epoch}-{step}",
                monitor=loss_to_monitor,
                mode="min",
                save_top_k=1,
                dirpath=run_path
            )
        ],
        "sync_batchnorm": config.train.get("sync_batchnorm", False),
        # "max_steps": config.train["number_steps"]
    }

    if config.train['logging']:
        trainer_params["logger"] = wandb_logger

    # if number_epochs is None, then max_steps is used else number_epochs
    if config.train.get("number_epochs", None) is not None:
        trainer_params["max_epochs"] = config.train["number_epochs"]
    else:
        trainer_params["max_steps"] = config.train["number_steps"]


    trainer = pl.Trainer(**trainer_params)

    trainer.fit(
        model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
        ckpt_path=checkpoint_path
    )


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # Misc flags
    flags.DEFINE_string("note", "", "Note for WandB")

    # Specify configs to use
    flags.DEFINE_string("data_config", "splice_all_basic_eutheria", "Data config.")
    flags.DEFINE_string("model_config", "mamba_deep", "Model config.")
    flags.DEFINE_string("projector_config", "default_512", "Projector config.")
    flags.DEFINE_string("optimizer_config", "anneal", "Optimizer config.")
    flags.DEFINE_string("train_config", "a100_ssm_512_6", "Train config.")
    flags.DEFINE_integer("nodes", 1, "Number of nodes")
    app.run(main)
