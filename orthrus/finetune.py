import os
import re
import gc
import torch
import wandb
import pandas as pd
import pytorch_lightning as pl

from absl import app
from absl import flags
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from orthrus.config_utils import Config, save_configs, find_latest_checkpoint
from orthrus.finetune_dataloader import get_split_dataframes, get_data_loaders_for_target
from mrna_bench.datasets import DATASET_INFO
from orthrus.finetune_model import RNATaskModel

def create_run_name(config) -> str:
    """Generate a run name based on config."""
    run_name = f"{config.data['dataset_name']}"

    if FLAGS.model_config in ["saluki", "resnet", "mamba_base", "contrastive_only_small"]:
        run_name += (
            f"_{FLAGS.model_config}_"
            f"lr{config.optimizer['model_lr']}_"
            f"wd{config.optimizer['model_weight_decay']}_"
            f"projlr{config.optimizer['projection_head_lr']}_"
            f"projwd{config.optimizer['projection_head_weight_decay']}_"
            f"seed{config.train['rand_seed']}"
        )
    else:
        mc = config.model["model_class"]
        run_name += f"_{mc}_lr{config.optimizer['model_lr']}_seed{config.train['rand_seed']}"

    # If you have an extra note
    if FLAGS.note:
        run_name += f"_{FLAGS.note}"

    # print(f"Run name: {run_name}")
    return run_name

def main(argv):
    # Load config
    config = Config("./finetune_config", FLAGS)

    if FLAGS.seed_override is not None:
        config.train["rand_seed"] = FLAGS.seed_override

    if FLAGS.subset_n_samples is not None:
        config.data["subset_n_samples"] = FLAGS.subset_n_samples
        config.data["subset_fraction"] = 1.0  # Ensure we don't fractionally subset

    torch.set_float32_matmul_precision("medium")

    if config.data.get("multi_seed", []): # if multi_seed is True, then avoid doing pytorch lightning seed_everything until after data loader is created

        # Create data loaders
        split_data = get_split_dataframes(config.data, config.train)
        target_cols = DATASET_INFO[config.data["dataset_name"]]["target_col"]
        if isinstance(target_cols, str):
            target_cols = [target_cols,]
        task_types = DATASET_INFO[config.data["dataset_name"]]["task"]
        if isinstance(task_types, str):
            task_types = [task_types,]

        pl.seed_everything(config.train["rand_seed"])
    else:
        pl.seed_everything(config.train["rand_seed"])

        split_data = get_split_dataframes(config.data, config.train)
        target_cols = DATASET_INFO[config.data["dataset_name"]]["target_col"]
        if isinstance(target_cols, str):
            target_cols = [target_cols,]
        task_types = DATASET_INFO[config.data["dataset_name"]]["task"]
        if isinstance(task_types, str):
            task_types = [task_types,]

    split_type = config.data["split_type"]
    run_name = create_run_name(config)
    run_path = os.path.join(config.train["wandb_run_dir"], run_name)

    os.makedirs(run_path, exist_ok=True)

    save_configs(
        run_path,
        config.model,
        config.optimizer,
        config.data,
        config.projector,
        config.train
    )

    results_file = os.path.join(run_path, "results.tsv")

    for i, target_col in enumerate(target_cols):
        # Get the task type for the target column
        task = task_types[i]

        print (f"Training on {target_col}")

        target_ckpt_path = os.path.join(run_path, target_col)
        os.makedirs(target_ckpt_path, exist_ok=True)

        for seed, train_df, val_df, test_df in split_data:

            train_data_loader, val_data_loader, test_data_loader = get_data_loaders_for_target(
                target_col=target_col,
                task=task,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                data_config=config.data,
                train_config=config.train,
                seed=seed,
            )

            print(f"Split type {split_type} - Task {task} - Seed {seed}")

            # add task to config.data["task"]
            config.data["task"] = task if task in ["classification", "multilabel"] else "regression"

            seed_suffix = f"{task}_" + \
                f"tcol-{target_col}_" + \
                f"split-{split_type}_" + \
                f"rs-{seed}"

            seed_run_path = os.path.join(target_ckpt_path, seed_suffix)

            os.makedirs(seed_run_path, exist_ok=True)

            # Check if checkpoint at run_path exists else start from scratch
            # iterate over checkpoints in "{epoch}-{step}.ckpt" and find latest
            checkpoint_path = find_latest_checkpoint(seed_run_path)

            if config.train['wandb_logging']:
                # Tells WandB that run resumes previous run from checkpoint
                logged_config = config.wandb_repr()
                wandb_kwargs = {"config": logged_config}

                # Initialize WandB
                wandb_logger = WandbLogger(
                    name=f"{run_name}_{seed_suffix}",
                    save_dir=seed_run_path,
                    project="Orthrus - DataSubSample2",
                    tags = ["finetuning"],
                    group=config.data["dataset_name"],
                    job_type=f"{FLAGS.model_config}_{target_col}",
                    reinit=True,
                    **wandb_kwargs
                )

            print(f"Logging to: {seed_run_path}")

            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            model = RNATaskModel(
                model_config=config.model,
                projector_config=config.projector,
                optimizer_config=config.optimizer,
                train_config=config.train,
                data_config=config.data
            )

            # if config.train['wandb_logging']:
            #     wandb_logger.watch(model, log_freq=500)

            # Choose a strategy
            if config.model["model_class"] == "resnet":
                strategy = DDPStrategy(find_unused_parameters=True)
            else:
                strategy = "ddp"

            tracking_metric = "validation_pearsonr"

            if task == "classification" or task == "multilabel":
                tracking_metric = "validation_auprc"

            # Checkpoint callback
            best_ckpt_callback = ModelCheckpoint(
                filename="max-metric-{epoch}-{step}",
                monitor=tracking_metric,
                mode="max",
                save_top_k=1,
                dirpath=seed_run_path,
            )
            callbacks = [
                best_ckpt_callback,
                # min_loss_ckpt_callback,
            ]

            trainer_params = {
                "accelerator": "gpu",
                "strategy": strategy,
                "devices": -1, # use all GPUs
                "num_nodes": FLAGS.nodes,
                "precision": "bf16-mixed" if config.train["mixed_precision"] else 32,
                "callbacks": callbacks,
                "gradient_clip_val": config.optimizer["gradient_clip_val"],
                "gradient_clip_algorithm": config.optimizer["gradient_clip_algorithm"],
                "log_every_n_steps": 25,
            }

            # if number_epochs is None, then max_steps is used else number_epochs
            if config.train.get("number_epochs", None) is not None:
                trainer_params["max_epochs"] = config.train["number_epochs"]
            else:
                trainer_params["max_steps"] = config.train["number_steps"]

            if config.train['wandb_logging']:
                trainer_params["logger"] = wandb_logger

            trainer = pl.Trainer(**trainer_params)

            trainer.fit(
                    model,
                    train_dataloaders=train_data_loader,
                    val_dataloaders=val_data_loader,
                    ckpt_path=checkpoint_path
                )


            # Load best model and evaluate
            top_model = RNATaskModel.load_from_checkpoint(
                checkpoint_path=best_ckpt_callback.best_model_path,
                model_config=config.model,
                optimizer_config=config.optimizer,
                train_config=config.train,
                data_config=config.data,
                projector_config=config.projector,
            )

            validation_results = trainer.validate(
                top_model,
                val_data_loader,
            )

            validation_r = float("nan")
            validation_auprc = float("nan")

            for metric, value in validation_results[0].items():

                if metric == 'validation_pearsonr':
                    validation_r = value
                if metric == 'validation_auprc':
                    validation_auprc = value

                if config.train['wandb_logging']:
                    wandb_logger.experiment.log({
                        f"best_val_results/{metric}": value,
                    })
                else:
                    print(f"Validation {metric}: {value}")

            test_results = trainer.test(
                top_model,
                test_data_loader,
            )

            test_r = float("nan")
            test_auprc = float("nan")

            for metric, value in test_results[0].items():

                if metric == 'test_pearsonr':
                    test_r = value
                if metric == 'test_auprc':
                    test_auprc = value

                if config.train['wandb_logging']:
                    wandb_logger.experiment.log({
                        f"best_test_results/{metric}": value,
                    })
                else:
                    print(f"Test {metric}: {value}")

            row = {
                "model": config.model["model_class"],
                "dataset": config.data["dataset_name"],
                "task": task,
                "target_col": target_col,
                "split_type": split_type,
                "seed": seed,
                "test_auprc": test_auprc,
                "test_r": test_r,
                "metric": test_auprc if task == "classification" or task == "multilabel" else test_r,
                "validation_auprc": validation_auprc,
                "validation_r": validation_r
            }

            df = pd.DataFrame([row])

            if trainer.global_rank == 0:
                if not os.path.exists(results_file):
                    df.to_csv(results_file, sep="\t", index=False)
                else:
                    df.to_csv(results_file, sep="\t", index=False, mode="a", header=False)

                if config.train['wandb_logging']:
                    wandb.finish()

            del model, trainer, top_model, train_data_loader, val_data_loader, test_data_loader
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # Misc flags
    flags.DEFINE_string("note", "", "Note for WandB")

    # Define absl flags
    flags.DEFINE_string("data_config", "go_mf", "Data config.")
    flags.DEFINE_string("model_config", "saluki", "Model config.")
    flags.DEFINE_string("projector_config", "saluki_projector", "Projector config.")
    flags.DEFINE_string("optimizer_config", "saluki", "Optimizer config.")
    flags.DEFINE_string("train_config", "bs_32", "Train config.")
    flags.DEFINE_integer("seed_override", None, "Training seed to override config.")
    flags.DEFINE_integer("subset_n_samples", None, "Number of training samples to use.")
    flags.DEFINE_integer("nodes", 1, "Number of nodes")
    app.run(main)