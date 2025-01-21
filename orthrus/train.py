import json
import re
import os

from absl import app
from absl import flags

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from orthrus.config_utils import Config
from orthrus.data_loader import init_data_loader
from orthrus.model import ContrastiveLearningModel


def save_config(
    run_path: str,
    model_config: dict,
    optimizer_config: dict,
    data_config: dict,
    projector_config: dict,
    train_config: dict,
) -> str | None:
    """Persist run config or load run from checkpoint if pre-existing.

    Args:
        run_path: Directory where config is stored.
        model_config: Model configs to be stored.
        optimizer_config: Optimizer configs to be stored.
        data_config: Data configs to be stored.
        projector_config: Projection head config to be stored.
        train_config: Training config to be stored.

    Returns:
        Path to model checkpoint if existing, or else None.
    """
    # Search for existing model checkpoints
    if os.path.exists(run_path):
        checkpoints = [x for x in os.listdir(run_path) if x.endswith(".ckpt")]
        # sort over epoch and step
        if checkpoints:
            sorted_checkpoint_list = sorted(
                checkpoints,
                key=lambda x: tuple(map(int, re.findall(r"(\d+)", x)))
            )
            last_checkpoint = sorted_checkpoint_list[-1]

            print(f"Resuming from checkpoint: {last_checkpoint}")
            checkpoint_path = os.path.join(run_path, last_checkpoint)
        else:
            checkpoint_path = None
    else:
        os.makedirs(run_path, exist_ok=True)
        checkpoint_path = None

    # Save all config to run_path if not pre-existing
    if not checkpoint_path:
        print("Starting model training from scratch")
        with open(os.path.join(run_path, "model_config.json"), "w") as f:
            json.dump(model_config, f)
        with open(os.path.join(run_path, "optimizer_config.json"), "w") as f:
            json.dump(optimizer_config, f)
        with open(os.path.join(run_path, "data_config.json"), "w") as f:
            json.dump(data_config, f)
        with open(os.path.join(run_path, "projector_config.json"), "w") as f:
            json.dump(projector_config, f)
        with open(os.path.join(run_path, "train_config.json"), "w") as f:
            json.dump(train_config, f)

    return checkpoint_path


def main(argv):
    config = Config("./config", FLAGS)

    pl.seed_everything(config.train["rand_seed"])

    warmup_steps = min([10_000, 0.1 * config.train["number_steps"]])
    config.optimizer["warmup_steps"] = warmup_steps

    mask_p = max(config.train["mask_prop"], config.data["proportion_to_mask"])

    # Initialize wandb logger
    run_name = (
        "ssm_"
        f"{config.model['n_tracks']}t_"
        f"{config.model['ssm_n_layers']}_"
        f"{config.model['ssm_model_dim']}_"
        f"lr{config.optimizer['model_lr']}_"
        f"wd{config.optimizer['model_weight_decay']}_"
        f"mask{mask_p}_"
        f"{FLAGS.data_config}"
    )

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
        config.train,
    )

    # Tells WandB that run resumes previous run from checkpoint
    logged_config = config.wandb_repr()
    wandb_kwargs = {"config": logged_config}
    if checkpoint_path:
        wandb_kwargs["resume"] = True

    wandb_logger = WandbLogger(
        name=run_name,
        save_dir=run_path,
        project="rna_rep_grid",
        **wandb_kwargs
    )

    # Allows PyTorch to resize memory allocated to batches. This allows
    # efficient handling of variable length batches.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    short_loader, long_loader, val_data_loader = init_data_loader(
        data_config=config.data,
        train_config=config.train,
    )

    train_data_loader = CombinedLoader(
        {"short": short_loader, "long": long_loader},
        mode="max_size_cycle"
    )

    # Model
    model = ContrastiveLearningModel(
        config.model,
        config.projector,
        config.optimizer,
        config.train,
        config.data
    )

    wandb_logger.watch(model, log_freq=500)

    trainer_params = {
        "accelerator": "gpu",
        "strategy": "ddp",
        "devices": 4,
        "num_nodes": 1,
        "logger": wandb_logger,
        "precision": "bf16" if config.train["mixed_precision"] else 32,
        "callbacks": [
            ModelCheckpoint(
                filename="{epoch}-{step}",
                every_n_train_steps=2000,
                dirpath=run_path,
                save_top_k=-1,
            )
        ],
        "gradient_clip_val": config.optimizer["clipnorm"],
        "max_steps": config.train["number_steps"],
    }

    trainer = pl.Trainer(**trainer_params)

    trainer.fit(
        model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # Misc flags
    flags.DEFINE_string("note", "", "Note for WandB")

    # Specify configs to use
    flags.DEFINE_string("data_config", "splice_all_basic_homo", "Data config.")
    flags.DEFINE_string("model_config", "mamba_large", "Model config.")
    flags.DEFINE_string("projector_config", "default_512", "Projector config.")
    flags.DEFINE_string("optimizer_config", "anneal", "Optimizer config.")
    flags.DEFINE_string("train_config", "a100_ssm_512_4", "Train config.")
    app.run(main)
