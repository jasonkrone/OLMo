"""
OLMo configuration
"""
from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from copy import deepcopy

logger = logging.get_logger(__name__)


DEFAULT_CONFIG_DICT = {
    'd_model': 768,
    'n_heads': 12,
    'n_kv_heads': None,
    'clip_qkv': None,
    'n_layers': 12,
    'mlp_ratio': 4,
    'mlp_hidden_size': None,
    'activation_type': 'swiglu',
    'block_type': 'sequential',
    'block_group_size': 1,
    'alibi': False,
    'alibi_bias_max': 8.0,
    'rope': False,
    'rope_full_precision': True,
    'rope_theta': 10000,
    'flash_attention': False,
    'attention_dropout': 0.1,
    'multi_query_attention': None,
    'attention_layer_norm': False,
    'residual_dropout': 0.1,
    'embedding_dropout': 0.1,
    'embedding_layer_norm': False,
    'layer_norm_type': 'default',
    'layer_norm_with_affine': True,
    'layer_norm_eps': 1e-05,
    'attention_layer_norm_with_affine': True,
    'max_sequence_length': 1024,
    'include_bias': True,
    'bias_for_layer_norm': None,
    'scale_logits': False,
    'vocab_size': 50257,
    'embedding_size': 50304,
    'weight_tying': True,
    'eos_token_id': 50256,
    'pad_token_id': 50256,
    'init_device': None,
    'init_fn': 'normal',
    'init_std': 0.02,
    'init_cutoff_factor': None,
    'precision': None,
    'scale_emb_init': False,
    'emb_init_std': None,
    'norm_after': False
}


class OLMoConfig(PretrainedConfig):
    model_type = "hf_olmo"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, use_cache: bool = False, **kwargs):
        all_kwargs = deepcopy(DEFAULT_CONFIG_DICT)
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {"architectures": all_kwargs.get("architectures", ["OLMoForCausalLM"]) or ["OLMoForCausalLM"]}
        )
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model


# Register the config class so that it is available for transformer pipelines, auto-loading etc.
# OLMo is integrated directly in transformers from v4.40.0 onwards, but the version in transformers
# may not support the newest architectures we create.
AutoConfig.register("hf_olmo", OLMoConfig)
