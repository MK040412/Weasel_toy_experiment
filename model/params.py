"""Weight loading from safetensors, adapted from jax-ml/bonsai for JAX 0.6.2."""

import gc
import re
from enum import Enum
from pathlib import Path

import jax
import safetensors
from flax import nnx

from model import modeling as model_lib


class Transform(Enum):
    DEFAULT = (None, None, False)
    BIAS = (None, None, False)
    LINEAR = ((1, 0), None, False)
    CONV3D = ((2, 3, 4, 1, 0), None, False)
    EMBED = (None, None, False)


def _get_vision_key_mapping():
    return {
        r"^model\.visual\.patch_embed\.proj\.weight$": ("model.visual.patch_embed.proj.kernel", Transform.CONV3D),
        r"^model\.visual\.patch_embed\.proj\.bias$": ("model.visual.patch_embed.proj.bias", Transform.BIAS),
        r"^model\.visual\.pos_embed\.weight$": ("model.visual.pos_embed.embedding", Transform.EMBED),
        r"^model\.visual\.blocks\.(\d+)\.norm1\.weight$": (r"model.visual.blocks.\1.norm1.scale", Transform.DEFAULT),
        r"^model\.visual\.blocks\.(\d+)\.norm1\.bias$": (r"model.visual.blocks.\1.norm1.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.norm2\.weight$": (r"model.visual.blocks.\1.norm2.scale", Transform.DEFAULT),
        r"^model\.visual\.blocks\.(\d+)\.norm2\.bias$": (r"model.visual.blocks.\1.norm2.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.attn\.qkv\.weight$": (r"model.visual.blocks.\1.attn.qkv.kernel", Transform.LINEAR),
        r"^model\.visual\.blocks\.(\d+)\.attn\.qkv\.bias$": (r"model.visual.blocks.\1.attn.qkv.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.attn\.proj\.weight$": (r"model.visual.blocks.\1.attn.proj.kernel", Transform.LINEAR),
        r"^model\.visual\.blocks\.(\d+)\.attn\.proj\.bias$": (r"model.visual.blocks.\1.attn.proj.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.weight$": (r"model.visual.blocks.\1.mlp.linear_fc1.kernel", Transform.LINEAR),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.bias$": (r"model.visual.blocks.\1.mlp.linear_fc1.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.weight$": (r"model.visual.blocks.\1.mlp.linear_fc2.kernel", Transform.LINEAR),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.bias$": (r"model.visual.blocks.\1.mlp.linear_fc2.bias", Transform.BIAS),
        r"^model\.visual\.merger\.norm\.weight$": ("model.visual.merger.norm.scale", Transform.DEFAULT),
        r"^model\.visual\.merger\.norm\.bias$": ("model.visual.merger.norm.bias", Transform.BIAS),
        r"^model\.visual\.merger\.linear_fc1\.weight$": ("model.visual.merger.linear_fc1.kernel", Transform.LINEAR),
        r"^model\.visual\.merger\.linear_fc1\.bias$": ("model.visual.merger.linear_fc1.bias", Transform.BIAS),
        r"^model\.visual\.merger\.linear_fc2\.weight$": ("model.visual.merger.linear_fc2.kernel", Transform.LINEAR),
        r"^model\.visual\.merger\.linear_fc2\.bias$": ("model.visual.merger.linear_fc2.bias", Transform.BIAS),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.norm\.weight$": (r"model.visual.deepstack_merger_list.\1.norm.scale", Transform.DEFAULT),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.norm\.bias$": (r"model.visual.deepstack_merger_list.\1.norm.bias", Transform.BIAS),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc1\.weight$": (r"model.visual.deepstack_merger_list.\1.linear_fc1.kernel", Transform.LINEAR),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc1\.bias$": (r"model.visual.deepstack_merger_list.\1.linear_fc1.bias", Transform.BIAS),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc2\.weight$": (r"model.visual.deepstack_merger_list.\1.linear_fc2.kernel", Transform.LINEAR),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc2\.bias$": (r"model.visual.deepstack_merger_list.\1.linear_fc2.bias", Transform.BIAS),
    }


def _get_text_key_mapping(tie_word_embeddings: bool = True):
    mapping = {
        r"^model\.language_model\.embed_tokens\.weight$": ("model.language_model.embed_tokens.embedding", Transform.EMBED),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (r"model.language_model.layers.\1.self_attn.q_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (r"model.language_model.layers.\1.self_attn.k_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (r"model.language_model.layers.\1.self_attn.v_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": (r"model.language_model.layers.\1.self_attn.o_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": (r"model.language_model.layers.\1.self_attn.q_norm.weight", Transform.DEFAULT),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": (r"model.language_model.layers.\1.self_attn.k_norm.weight", Transform.DEFAULT),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": (r"model.language_model.layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight$": (r"model.language_model.layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$": (r"model.language_model.layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$": (r"model.language_model.layers.\1.input_layernorm.weight", Transform.DEFAULT),
        r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight$": (r"model.language_model.layers.\1.post_attention_layernorm.weight", Transform.DEFAULT),
        r"^model\.language_model\.norm\.weight$": ("model.language_model.norm.weight", Transform.DEFAULT),
    }
    if tie_word_embeddings:
        mapping[r"^lm_head\.weight$"] = ("model.language_model.embed_tokens.embedding", Transform.EMBED)
    else:
        mapping[r"^lm_head\.weight$"] = ("lm_head.kernel", Transform.LINEAR)
    return mapping


def _get_key_and_transform_mapping(tie_word_embeddings: bool = True):
    mapping = {}
    mapping.update(_get_vision_key_mapping())
    mapping.update(_get_text_key_mapping(tie_word_embeddings))
    return mapping


def _torch_key_to_jax_key(mapping, source_key):
    matches = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if not matches:
        return None, None
    if len(matches) > 1:
        raise ValueError(f"Multiple mappings for {source_key}: {[m[0] for m in matches]}")
    return matches[0]


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def _apply_transform(tensor, transform):
    if transform is None or transform.value is None:
        return tensor
    permute, reshape, reshape_first = transform.value
    if reshape_first and reshape is not None:
        tensor = tensor.reshape(reshape)
    if permute is not None:
        tensor = tensor.transpose(permute)
    if not reshape_first and reshape is not None:
        tensor = tensor.reshape(reshape)
    return tensor


def _assign_weights(keys, tensor, state_dict, torch_key, transform):
    key, *rest = keys
    if not rest:
        tensor = _apply_transform(tensor, transform)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {torch_key}: got {tensor.shape}, expected {state_dict[key].shape}")
        state_dict[key] = jax.device_put(tensor)
    else:
        _assign_weights(rest, tensor, state_dict[key], torch_key, transform)


def create_model_from_safe_tensors(file_dir: str, config: model_lib.ModelConfig,
                                    model_filename: str | None = None) -> model_lib.Qwen3VLForConditionalGeneration:
    path = Path(file_dir).expanduser()
    if model_filename:
        files = [path / model_filename]
        if not files[0].exists():
            raise ValueError(f"File not found: {files[0]}")
    else:
        files = list(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files in {file_dir}")

    model = nnx.eval_shape(lambda: model_lib.Qwen3VLForConditionalGeneration(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)
    key_mapping = _get_key_and_transform_mapping(config.text_config.tie_word_embeddings)
    errors = []

    for f in files:
        print(f"  Loading {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)
                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform)
                except Exception as e:
                    errors.append(f"  {torch_key} -> {jax_key}: {e}")
        gc.collect()

    if errors:
        raise RuntimeError(f"{len(errors)} weight errors:\n" + "\n".join(errors))

    gc.collect()
    return nnx.merge(graph_def, state_dict)
