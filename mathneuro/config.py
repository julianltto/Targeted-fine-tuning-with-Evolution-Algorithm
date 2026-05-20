from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any


@dataclass
class MathNeuroConfig:
    model: str
    save_path: str
    train_dataset: str
    eval_datasets: list[str]
    calibration_datasets: list[str]
    calibration_dataset_names: list[str]

    text_file: str = "results.txt"
    num_repeats: int = 5
    pre_train_eval: bool = False
    random_state: int = 42
    scalar: float = 0.0
    eval_dataset_size: int | None = None
    eval_dataset_subset: int = 100
    num_samples: int = 500
    train_lm_eval_task: str | None = None
    proportion: float | list[float] | None = None

    with_ea: bool = False
    ea_mode: str = 'scale'       
    ea_max_scale: float = 0.1
    ea_max_prune: float = 0.1
    ea_pop_size: int = 20
    ea_n_gen: int = 15
    ea_eval_samples: int = 30
    ea_fitness_version: str = 'gsm8k_cot_flex_v2'
    ea_general_task: str = 'mmlu_high_school_world_history'

    batch_size: int | str = 1

    def __post_init__(self) -> None:
        if len(self.calibration_datasets) != len(self.calibration_dataset_names):
            raise ValueError(
                "`calibration_datasets` and `calibration_dataset_names` must have the same length "
                f"(got {len(self.calibration_datasets)} and {len(self.calibration_dataset_names)})."
            )

    def replace(self, **overrides: Any) -> "MathNeuroConfig":
        return replace(self, **overrides)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "MathNeuroConfig":
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required to read YAML configs. Install with `pip install PyYAML`."
            ) from exc

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"YAML config at `{path}` must be a mapping.")
        return cls(**data)


CONFIG: MathNeuroConfig = MathNeuroConfig(
    model="meta-llama/Llama-3.2-1B-Instruct",
    save_path="./results",
    train_dataset="data/gsm8k.csv",
    eval_datasets=["race", "mmlu"],
    calibration_datasets=["data/race.csv", "data/mmlu.csv"],
    calibration_dataset_names=["Race", "MMLU"],
    text_file="results.txt",
    num_repeats=1,
    proportion=0.15,
    num_samples=16,
    eval_dataset_subset=8,
    train_lm_eval_task="gsm8k_cot",
    with_ea=True,
    ea_max_scale=0.1,
    ea_max_prune=0.1,
    ea_fitness_version='gsm8k_cot_flex_v2',
)


def load_config(argv: list[str] | None = None) -> MathNeuroConfig:
    argv = sys.argv if argv is None else argv

    if len(argv) == 1:
        return CONFIG
    if len(argv) == 2 and not argv[1].startswith("-"):
        return MathNeuroConfig.from_yaml(argv[1])
    if len(argv) == 3 and argv[1] in {"--config", "-c"}:
        return MathNeuroConfig.from_yaml(argv[2])

    raise SystemExit(
        f"Usage: python {argv[0]} [path/to/config.yaml]\n"
        f"   or: python {argv[0]} --config path/to/config.yaml"
    )
