"""
extra axolotl specific training args
"""

from dataclasses import dataclass, field
from typing import Optional

from PIL.Image import Resampling
from transformers import TrainingArguments
from trl import CPOConfig, KTOConfig, ORPOConfig, PRMConfig, RewardConfig


@dataclass
class AxolotlTrainingMixins:
    """
    Mixin class for the Axolotl training args.
    """
    # ... [other fields remain unchanged]

    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for dataset loading."}
    )

    # ... [other fields remain unchanged]


@dataclass
class AxolotlTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    """
    Training arguments for Causal trainer

    This code is duplicated due to HF TrainingArguments not setting output_dir with a
    default value so it can't be used as a mixin.
    """


@dataclass
class AxolotlORPOConfig(AxolotlTrainingMixins, ORPOConfig):
    """
    ORPO config for ORPO training
    """


@dataclass
class AxolotlKTOConfig(AxolotlTrainingMixins, KTOConfig):
    """
    KTO config for KTO training
    """


@dataclass
class AxolotlCPOConfig(AxolotlTrainingMixins, CPOConfig):
    """
    CPO config for CPO training
    """


@dataclass
class AxolotlRewardConfig(AxolotlTrainingMixins, RewardConfig):
    """
    Reward config for Reward training
    """


@dataclass
class AxolotlPRMConfig(AxolotlTrainingMixins, PRMConfig):
    """
    PRM config for PRM training
    """
```

This change adds the `dataset_num_proc` attribute to the `AxolotlTrainingMixins` class, which is inherited by `AxolotlTrainingArguments`, thus resolving the CI failure.
