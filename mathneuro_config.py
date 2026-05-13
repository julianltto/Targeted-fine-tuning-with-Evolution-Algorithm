import sys
from types import SimpleNamespace

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to read config files. Install it with `pip install PyYAML`.") from exc


DEFAULTS = {
    "text_file": "results.txt",
    "num_repeats": 5,
    "pre_train_eval": False,
    "random_state": 42,
    "scalar": 0,
    "eval_dataset_size": None,
    "eval_dataset_subset": 100,
    "num_samples": 500,
    "train_lm_eval_task": None,
    "proportion": None,
}

VALIDATION = {
    "required_keys": [
        "model",
        "eval_datasets",
        "train_dataset",
        "calibration_datasets",
        "save_path",
        "calibration_dataset_names",
    ],
    "list_keys": [
        "eval_datasets",
        "calibration_datasets",
        "calibration_dataset_names",
    ],
    "equal_length_groups": [
        [
            "calibration_datasets",
            "calibration_dataset_names",
        ],
    ],
}


def get_config_path(argv):
    if len(argv) == 2 and not argv[1].startswith("-"):
        return argv[1]
    if len(argv) == 3 and argv[1] in {"--config", "-c"}:
        return argv[2]
    raise SystemExit("Usage: python run.py path/to/config.yaml\n   or: python run.py --config path/to/config.yaml")


def read_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"The YAML config at `{config_path}` must contain a mapping of config keys to values.")
    return config


def load_config(argv=None):
    argv = sys.argv if argv is None else argv
    config = read_yaml_config(get_config_path(argv))

    args_dict = {**DEFAULTS, **config}

    missing_keys = [key for key in VALIDATION["required_keys"] if args_dict.get(key) is None]
    if missing_keys:
        raise ValueError(f"Missing required config key(s): {', '.join(missing_keys)}")

    for key in VALIDATION["list_keys"]:
        if not isinstance(args_dict[key], list):
            raise TypeError(f"`{key}` must be a YAML list.")

    for key_group in VALIDATION["equal_length_groups"]:
        lengths = {key: len(args_dict[key]) for key in key_group}
        if len(set(lengths.values())) != 1:
            keys = ", ".join(key_group)
            raise ValueError(f"`{keys}` must have the same length.")

    return SimpleNamespace(**args_dict)
