from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    datasets_root: Path = Path("datasets")
    checkpoints_root: Path = Path("checkpoints")

    def for_family(self, family: str, purpose: str) -> Path:
        return self.checkpoints_root / f"{family}_{purpose}"


@dataclass
class ModelConfig:
    pass


@dataclass
class PPOConfig:
    pass


@dataclass
class PretrainConfig:
    pass


@dataclass
class EnvConfig:
    pass


@dataclass
class DataGenConfig:
    pass
