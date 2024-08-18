from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
#from omegaconf import DictConfig, ListConfig
#from omegaconf import OmegaConf as om
#from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError
from .util import StrEnum

__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "LayerNormType",
    "InitFnType",
    "ModelConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "DataConfig",
    "InstanceFilterConfig",
    "EvaluatorConfig",
    "TokenizerConfig",
    "TrainConfig",
    "PaddingDirection",
    "TruncationDirection",
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "DDPConfig",
    "DistributedStrategy",
    "DDPGradSyncMode",
    "FSDPPrecision",
    "FSDPWrapStrategy",
    "FSDPConfig",
    "CheckpointType",
]

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


class ModelConfig(object):

    def __init__(self):
        pass


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """



class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"



class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"
    max_scheduler = "max_scheduler"
    constant = "constant"
    cosine_linear_envelope = "cosine_linear_envelope"


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"



class PaddingDirection(StrEnum):
    right = "right"
    left = "left"



class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"



class DistributedStrategy(StrEnum):
    ddp = "ddp"
    """
    Wrap OLMo in torch.nn.parallel.DistributedDataParallel to train across ranks.
    """

    fsdp = "fsdp"
    """
    Wrap OLMo in torch.distributed.fsdp.FullyShardedDataParallel to train across ranks.
    """


class DDPGradSyncMode(StrEnum):
    batch = "batch"
    """
    Synchronize gradients after computation at each bucket only at the last micro-batch.
    This is slightly faster than gradient syncs across each micro-batch but will consume more memory.
    Can use this mode only when `find_unused_params` is set to False.
    """

    micro_batch = "micro_batch"
    """
    Synchronize gradients after computation at each bucket per micro-batch.
    This will be slightly slower than gradient sync at the last micro-batch, but will consume less memory.
    Can use this mode with both option of `find_unused_params` but specifically recommended to use with `find_unused_params`
    set to True, to prevent errors.
    """



class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each OLMo block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """



class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"
    olmo_core = "olmo_core"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    """
    Checkpoint one in two transformer layers.
    """

    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layers.
    """

    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    two_in_three = "two_in_three"
    """
    Checkpoint two out of every three transformer layers.
    """

    three_in_four = "three_in_four"
    """
    Checkpoint three out of four of every transformer layers.
    """

    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """


