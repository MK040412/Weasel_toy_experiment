"""Unified inference for Qwen3-VL 2B and Qwen3.5-0.8B on TPU v4-8."""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_ROOT, "..", "..", "..", "models")

EOS_TOKEN_ID = 151643


# ---------------------------------------------------------------------------
# Qwen3-VL 2B
# ---------------------------------------------------------------------------


def run_qwen3vl(model_path: str):
    from transformers import AutoProcessor

    from qwen.qwen3vl import modeling

    hf_id = "Qwen/Qwen3-VL-2B-Instruct"

    print("=" * 60)
    print("Qwen3-VL 2B — JAX/TPU v4-8 Inference")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")

    print("\n1. Loading model...")
    t0 = time.time()
    config = modeling.ModelConfig.qwen3vl_2b()
    model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(model_path, config=config)
    print(f"   Loaded in {time.time() - t0:.2f}s")

    print("\n2. Loading processor...")
    processor = AutoProcessor.from_pretrained(hf_id)

    print("\n3. Preparing image input...")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg",
                },
                {"type": "text", "text": "What is in this image? Answer briefly."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )

    input_ids = jnp.array(inputs["input_ids"].numpy())
    seq_len = input_ids.shape[1]
    print(f"   Input tokens: {seq_len}")

    if "pixel_values" not in inputs:
        print("   ERROR: No pixel_values")
        return

    pixel_values = jnp.array(inputs["pixel_values"].numpy())
    image_grid_thw = jnp.array(inputs["image_grid_thw"].numpy())
    token_type_ids = (input_ids == config.image_token_id).astype(jnp.int32)
    print(f"   Pixel values: {pixel_values.shape}, Image tokens: {int(token_type_ids.sum())}")

    cache = modeling.init_cache(config, batch_size=1, token_len=seq_len, generate_steps=200)

    print("\n4. Generating...")
    t0 = time.time()
    generated = []
    logits, cache = modeling.forward_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids)
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated.append(next_token)

    for _ in range(99):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated.append(next_token)
        if int(np.array(next_token)[0, 0]) == EOS_TOKEN_ID:
            break

    elapsed = time.time() - t0
    num_new = len(generated)
    print(f"   {num_new} tokens in {elapsed:.2f}s ({num_new / elapsed:.1f} tok/s)")

    gen_ids = np.concatenate([np.array(t) for t in generated], axis=1)
    text_out = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("\n   Q: What is in this image?")
    print(f"   A: {text_out[0]}")


# ---------------------------------------------------------------------------
# Qwen3.5-0.8B
# ---------------------------------------------------------------------------


def run_qwen35(model_path: str):
    from transformers import AutoTokenizer

    from qwen.qwen35 import modeling

    hf_id = "Qwen/Qwen3.5-0.8B"

    print("=" * 60)
    print("Qwen3.5-0.8B — JAX/TPU v4-8 Inference")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")

    print("\n1. Loading model...")
    t0 = time.time()
    config = modeling.ModelConfig.qwen35_0_8b()
    model = modeling.Qwen35ForConditionalGeneration.from_pretrained(model_path, config=config)
    print(f"   Loaded in {time.time() - t0:.2f}s")

    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    prompt = "Explain what a neural network is in 2 sentences."
    print(f"\n3. Prompt: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = jnp.array(tokenizer(text, return_tensors="np")["input_ids"])
    seq_len = input_ids.shape[1]
    print(f"   Input: {seq_len} tokens")

    cache = modeling.init_cache(config, batch_size=1, token_len=seq_len, gen_steps=200)

    print("\n4. Generating...")
    t0 = time.time()
    generated = []
    logits, cache = modeling.forward(model, cache, input_ids)
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated.append(next_token)

    for _ in range(99):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated.append(next_token)
        if int(np.array(next_token).flat[0]) == tokenizer.eos_token_id:
            break

    elapsed = time.time() - t0
    all_toks = np.concatenate([np.array(t) for t in generated], axis=1)
    print(f"   {all_toks.shape[1]} tokens in {elapsed:.2f}s ({all_toks.shape[1] / elapsed:.1f} tok/s)")
    print(f"\n   A: {tokenizer.decode(all_toks[0], skip_special_tokens=True)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen inference on TPU v4-8")
    parser.add_argument("--model", choices=["qwen3vl", "qwen35"], default="qwen3vl")
    parser.add_argument("--model-path", default=None, help="Override model weights path")
    args = parser.parse_args()

    if args.model == "qwen3vl":
        path = args.model_path or os.environ.get("QWEN3VL_MODEL_PATH", os.path.join(_MODELS_DIR, "qwen3-vl-2b"))
        run_qwen3vl(path)
    else:
        path = args.model_path or os.environ.get("QWEN35_MODEL_PATH", os.path.join(_MODELS_DIR, "qwen35-0.8b"))
        run_qwen35(path)


if __name__ == "__main__":
    main()
