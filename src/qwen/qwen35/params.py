"""Weight loading for Qwen3.5-0.8B from safetensors."""

import gc
import re
from enum import Enum
from pathlib import Path

import jax
import safetensors
from flax import nnx

from qwen.qwen35 import modeling as model_lib


class Transform(Enum):
    DEFAULT = (None, None, False)
    BIAS = (None, None, False)
    LINEAR = ((1, 0), None, False)  # (out, in) -> (in, out)
    CONV3D = ((2, 3, 4, 1, 0), None, False)  # (out, in, T, H, W) -> (T, H, W, in, out)
    EMBED = (None, None, False)
    CONV1D_DEPTHWISE = None  # Special handling


def _get_vision_key_mapping():
    """Vision encoder weight mapping."""
    return {
        r"^model\.visual\.patch_embed\.proj\.weight$": ("visual.patch_embed.proj.kernel", Transform.CONV3D),
        r"^model\.visual\.patch_embed\.proj\.bias$": ("visual.patch_embed.proj.bias", Transform.BIAS),
        r"^model\.visual\.pos_embed\.weight$": ("visual.pos_embed.embedding", Transform.EMBED),
        # Vision blocks
        r"^model\.visual\.blocks\.(\d+)\.norm1\.weight$": (r"visual.blocks.\1.norm1.scale", Transform.DEFAULT),
        r"^model\.visual\.blocks\.(\d+)\.norm1\.bias$": (r"visual.blocks.\1.norm1.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.norm2\.weight$": (r"visual.blocks.\1.norm2.scale", Transform.DEFAULT),
        r"^model\.visual\.blocks\.(\d+)\.norm2\.bias$": (r"visual.blocks.\1.norm2.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.attn\.qkv\.weight$": (r"visual.blocks.\1.attn.qkv.kernel", Transform.LINEAR),
        r"^model\.visual\.blocks\.(\d+)\.attn\.qkv\.bias$": (r"visual.blocks.\1.attn.qkv.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.attn\.proj\.weight$": (r"visual.blocks.\1.attn.proj.kernel", Transform.LINEAR),
        r"^model\.visual\.blocks\.(\d+)\.attn\.proj\.bias$": (r"visual.blocks.\1.attn.proj.bias", Transform.BIAS),
        # Vision MLP: linear_fc1/linear_fc2 in safetensors -> fc1/fc2 in our model
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.weight$": (
            r"visual.blocks.\1.mlp.fc1.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.bias$": (r"visual.blocks.\1.mlp.fc1.bias", Transform.BIAS),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.weight$": (
            r"visual.blocks.\1.mlp.fc2.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.bias$": (r"visual.blocks.\1.mlp.fc2.bias", Transform.BIAS),
        # Merger
        r"^model\.visual\.merger\.norm\.weight$": ("visual.merger.norm.scale", Transform.DEFAULT),
        r"^model\.visual\.merger\.norm\.bias$": ("visual.merger.norm.bias", Transform.BIAS),
        r"^model\.visual\.merger\.linear_fc1\.weight$": ("visual.merger.fc1.kernel", Transform.LINEAR),
        r"^model\.visual\.merger\.linear_fc1\.bias$": ("visual.merger.fc1.bias", Transform.BIAS),
        r"^model\.visual\.merger\.linear_fc2\.weight$": ("visual.merger.fc2.kernel", Transform.LINEAR),
        r"^model\.visual\.merger\.linear_fc2\.bias$": ("visual.merger.fc2.bias", Transform.BIAS),
    }


def _get_text_key_mapping(tie_word_embeddings: bool = True):
    """Text decoder weight mapping."""
    mapping = {
        # Embeddings
        r"^model\.language_model\.embed_tokens\.weight$": ("language_model.embed_tokens.embedding", Transform.EMBED),
        r"^model\.language_model\.norm\.weight$": ("language_model.norm.weight", Transform.DEFAULT),
        # Full attention layers (self_attn)
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"language_model.layers.\1.self_attn.q_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"language_model.layers.\1.self_attn.k_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"language_model.layers.\1.self_attn.v_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": (
            r"language_model.layers.\1.self_attn.o_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": (
            r"language_model.layers.\1.self_attn.q_norm.weight",
            Transform.DEFAULT,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": (
            r"language_model.layers.\1.self_attn.k_norm.weight",
            Transform.DEFAULT,
        ),
        # GDN (linear_attn) layers
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$": (
            r"language_model.layers.\1.gdn.in_proj_qkv.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$": (
            r"language_model.layers.\1.gdn.in_proj_z.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight$": (
            r"language_model.layers.\1.gdn.in_proj_a.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight$": (
            r"language_model.layers.\1.gdn.in_proj_b.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$": (
            r"language_model.layers.\1.gdn.out_proj.kernel",
            Transform.LINEAR,
        ),
        # GDN conv1d: (1, conv_dim, kernel_size) in PyTorch -> (conv_dim, kernel_size) for us
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$": (
            r"language_model.layers.\1.gdn.conv1d_weight",
            Transform.CONV1D_DEPTHWISE,
        ),
        # conv1d has no bias in Qwen3.5
        # GDN scalar params
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.A_log$": (
            r"language_model.layers.\1.gdn.A_log",
            Transform.DEFAULT,
        ),
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.dt_bias$": (
            r"language_model.layers.\1.gdn.dt_bias",
            Transform.DEFAULT,
        ),
        # GDN norm
        r"^model\.language_model\.layers\.(\d+)\.linear_attn\.norm\.weight$": (
            r"language_model.layers.\1.gdn.norm.weight",
            Transform.DEFAULT,
        ),
        # MLP (all layers)
        r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": (
            r"language_model.layers.\1.mlp.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight$": (
            r"language_model.layers.\1.mlp.up_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$": (
            r"language_model.layers.\1.mlp.down_proj.kernel",
            Transform.LINEAR,
        ),
        # Layer norms
        r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$": (
            r"language_model.layers.\1.input_layernorm.weight",
            Transform.DEFAULT,
        ),
        r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight$": (
            r"language_model.layers.\1.post_attention_layernorm.weight",
            Transform.DEFAULT,
        ),
    }

    if tie_word_embeddings:
        mapping[r"^lm_head\.weight$"] = ("language_model.embed_tokens.embedding", Transform.EMBED)
    else:
        mapping[r"^lm_head\.weight$"] = ("lm_head.kernel", Transform.LINEAR)

    return mapping


def _get_all_mappings(tie_word_embeddings: bool = True):
    m = {}
    m.update(_get_vision_key_mapping())
    m.update(_get_text_key_mapping(tie_word_embeddings))
    return m


def _torch_key_to_jax(mapping, source_key):
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
    if transform is None or transform is Transform.CONV1D_DEPTHWISE:
        if transform is Transform.CONV1D_DEPTHWISE:
            # PyTorch conv1d grouped: (1, conv_dim, kernel_size) or (conv_dim, 1, kernel_size)
            # We store as (conv_dim, kernel_size)
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    tensor = tensor[0]  # (1, conv_dim, kernel_size) -> (conv_dim, kernel_size)
                elif tensor.shape[1] == 1:
                    tensor = tensor[:, 0, :]  # (conv_dim, 1, kernel_size) -> (conv_dim, kernel_size)
            return tensor
        return tensor
    if transform.value is None:
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


def create_model_from_safe_tensors(file_dir: str, config: model_lib.ModelConfig, model_filename: str | None = None):
    path = Path(file_dir).expanduser()
    if model_filename:
        files = [path / model_filename]
    else:
        files = list(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files in {file_dir}")

    model = nnx.eval_shape(lambda: model_lib.Qwen35ForConditionalGeneration(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)
    key_mapping = _get_all_mappings(config.text_config.tie_word_embeddings)
    errors = []
    mapped = 0
    skipped = []

    for f in files:
        print(f"  Loading {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)
                jax_key, transform = _torch_key_to_jax(key_mapping, torch_key)
                if jax_key is None:
                    skipped.append(torch_key)
                    continue
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform)
                    mapped += 1
                except Exception as e:
                    errors.append(f"  {torch_key} -> {jax_key}: {e}")
        gc.collect()

    print(f"  Mapped {mapped} weights, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped keys (first 10): {skipped[:10]}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")
        raise RuntimeError(f"{len(errors)} weight conversion errors")

    gc.collect()
    return nnx.merge(graph_def, state_dict)
