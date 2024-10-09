import json
import yaml
import os
import re


class Config:
    config_types = ["data", "model", "projector", "optimizer", "train"]

    data_attrs = [
        "base_genome_set_path",
        "base_genome_annot_path",
        "base_genome_fasta_path",
        "transcript_save_dir",
        "transcript_map_save_dir"
        "n_tracks",
        "zero_pad",
        "zero_mean",
        "pad_length_to",
        "always_different_transcripts",
        "transcript_transcript",
        "proportion_to_mask",
        "add_paralogs",
        "ortholog_path",
        "mini"
    ]

    model_attrs = [
        "model_class",
        "num_classes",
        "pooling_layer",
        "add_shift",
        "increase_dilation",
        "resnet",
        "norm_type",
        "kernel_size",
        "global_pooling_layer",
        "ssm_model_dim",
        "ssm_n_layers",
    ]

    projector_attrs = [
        "representation_dim",
        "projection_head_size",
        "projection_body",
        "projection_norm_type",
    ]

    train_attrs = [
        "rand_seed",
        "number_steps",
        "mixed_precision",
        "wandb_run_dir",
        "split_by_length",
        "transcript_split_len"
        "gpu_batch_sizes",
        "note",
    ]

    optimizer_attrs = [
        "model_lr",
        "model_weight_decay",
        "projection_head_lr",
        "projection_head_weight_decay",
        "clipnorm"
    ]

    def __init__(self, config_dir: str, flags):
        self.flags = flags
        self.config_dir = config_dir

        self.attr_to_subconfig_map = self.get_attr_to_config_map()

        self.load_config_yaml()
        self.update_config_flags()

    def get_attr_to_config_map(self) -> dict:
        """Generate map from attribute to containing subconfig."""
        attr_to_subconfig_map = {}
        for config_type in self.config_types:
            for attr in getattr(self, config_type + "_attrs"):
                if attr in attr_to_subconfig_map:
                    raise ValueError("Duplicate config key.")
                else:
                    attr_to_subconfig_map[attr] = config_type

        return attr_to_subconfig_map

    def load_config_yaml(self):
        """Load base config from YAML files in specified folder."""
        used_subconfigs = {}

        for config_type in self.config_types:
            config_path = self.config_dir + "/" + config_type + ".yaml"
            with open(config_path, "r") as f:
                all_subconfig = yaml.safe_load(f)

                subconfig_key = getattr(self.flags, config_type + "_config")
                subconfig = all_subconfig[subconfig_key]

                used_subconfigs[config_type] = subconfig_key

            setattr(self, config_type, subconfig)

        self.used_subconfigs = used_subconfigs

    def update_attr(self, attr: str, value):
        """Update attribute that exists in subconfig."""
        getattr(self, self.attr_to_subconfig_map[attr])[attr] = value

    def update_config_flags(self):
        """Update config with values defined in flags."""
        all_attrs = [k for k in self.attr_to_subconfig_map.keys()]

        for a in all_attrs:
            if hasattr(self.flags, a) and getattr(self.flags, a) is not None:
                self.update_attr(a, getattr(self.flags, a))

    def __repr__(self) -> str:
        """Return string representation of config."""
        dict_repr = self.wandb_repr()
        return json.dumps(dict_repr, indent=4)

    def wandb_repr(self) -> dict[str, dict]:
        """Return a stripped config suitable for WandB logging."""
        dict_repr = {"used_subconfigs": self.used_subconfigs}
        for config_type in self.config_types:
            dict_repr[config_type] = getattr(self, config_type)

        return dict_repr


def save_config(
    run_path: str,
    model_config: dict,
    optimizer_config: dict,
    data_config: dict,
    projector_config: dict,
    train_config: dict
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
                key=lambda x: tuple(map(int, re.findall(r'(\d+)', x)))
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
