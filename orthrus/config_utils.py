import json
import yaml
import os
import re
from typing import Optional

class Config:
    config_types = ["data", "model", "projector", "optimizer", "train"]

    data_attrs = []
    model_attrs = []
    projector_attrs = []
    train_attrs = []
    optimizer_attrs = []

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

def find_latest_checkpoint(run_path: str) -> Optional[str]:
    if not os.path.exists(run_path):
        return None
    ckpts = [f for f in os.listdir(run_path) if f.endswith(".ckpt")]
    if not ckpts:
        return None
    sorted_ckpts = sorted(
        ckpts,
        key=lambda x: tuple(map(int, re.findall(r"(\d+)", x)))
    )
    return os.path.join(run_path, sorted_ckpts[-1])


def save_configs(
    run_path: str,
    model_config: dict,
    optimizer_config: dict,
    data_config: dict,
    projector_config: dict,
    train_config: dict
) -> None:
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


def save_config(
    run_path: str,
    model_config: dict,
    optimizer_config: dict,
    data_config: dict,
    projector_config: dict,
    train_config: dict
) -> Optional[str]:
    ckpt = find_latest_checkpoint(run_path)
    if ckpt:
        print(f"Resuming from checkpoint: {os.path.basename(ckpt)}")
        return ckpt
    os.makedirs(run_path, exist_ok=True)
    print("Starting model training from scratch")
    save_configs(
        run_path,
        model_config,
        optimizer_config,
        data_config,
        projector_config,
        train_config
    )
    return None