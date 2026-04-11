"""Qwen3-VL 2B batch serving benchmark on TPU v4-8."""

import os
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor

from model import modeling

MODEL_PATH = os.environ.get("QWEN3VL_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen3-vl-2b"))
HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
EOS_TOKEN_ID = 151643


def generate_text_only(model, cache, input_ids, max_new_tokens=50):
    """Text-only generation after vision prefill."""
    generated_tokens = []
    logits, cache = modeling.forward(model, cache, input_ids)
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated_tokens.append(next_token)

    for _ in range(max_new_tokens - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated_tokens.append(next_token)
        tok_val = int(np.array(next_token).flat[0])
        if tok_val == EOS_TOKEN_ID:
            break

    return generated_tokens


def benchmark_text_generation(model, config, processor, batch_sizes=[1, 2, 4]):
    """Benchmark text-only generation at various batch sizes."""
    print("\n" + "=" * 60)
    print("TEXT-ONLY GENERATION BENCHMARK")
    print("=" * 60)

    prompt = "Explain quantum computing in simple terms."
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor.tokenizer(text, return_tensors="np")
    input_ids_np = inputs["input_ids"]
    seq_len = input_ids_np.shape[1]

    max_new = 100

    for bs in batch_sizes:
        print(f"\n--- Batch size: {bs} ---")

        # Replicate input for batch
        batched_ids = jnp.array(np.tile(input_ids_np, (bs, 1)))
        cache = modeling.init_cache(config, batch_size=bs, token_len=seq_len, generate_steps=max_new)

        # Warmup (JIT compile)
        print("  Warming up JIT...")
        warmup_start = time.time()
        _ = generate_text_only(model, cache, batched_ids, max_new_tokens=5)
        warmup_time = time.time() - warmup_start
        print(f"  JIT warmup: {warmup_time:.2f}s")

        # Reset cache for actual benchmark
        cache = modeling.init_cache(config, batch_size=bs, token_len=seq_len, generate_steps=max_new)

        # Benchmark
        print("  Running benchmark...")
        start = time.time()
        tokens = generate_text_only(model, cache, batched_ids, max_new_tokens=max_new)
        # Force sync
        jax.block_until_ready(tokens[-1])
        elapsed = time.time() - start

        num_generated = len(tokens)
        total_tokens = num_generated * bs
        tok_per_sec = total_tokens / elapsed

        print(f"  Generated: {num_generated} tokens/sample × {bs} batch = {total_tokens} total")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {tok_per_sec:.1f} tok/s (total), {num_generated/elapsed:.1f} tok/s (per sample)")

        # Decode first sample
        all_toks = np.concatenate([np.array(t) for t in tokens], axis=1)
        text_out = processor.tokenizer.decode(all_toks[0], skip_special_tokens=True)
        print(f"  Sample output: {text_out[:150]}...")


def benchmark_vision_generation(model, config, processor):
    """Benchmark vision+text generation."""
    print("\n" + "=" * 60)
    print("VISION + TEXT GENERATION BENCHMARK")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"},
                {"type": "text", "text": "What is in this image? Answer briefly."},
            ],
        }
    ]

    print("  Processing image...")
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )

    input_ids = jnp.array(inputs["input_ids"].numpy())
    pixel_values = jnp.array(inputs["pixel_values"].numpy())
    image_grid_thw = jnp.array(inputs["image_grid_thw"].numpy())
    token_type_ids = (input_ids == config.image_token_id).astype(jnp.int32)
    seq_len = input_ids.shape[1]

    print(f"  Input: {seq_len} tokens, {int(token_type_ids.sum())} image tokens")
    print(f"  Pixel values: {pixel_values.shape}")

    max_new = 100
    cache = modeling.init_cache(config, batch_size=1, token_len=seq_len, generate_steps=max_new)

    # Vision prefill
    print("  Running vision prefill...")
    start = time.time()
    logits, cache = modeling.forward_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids)
    prefill_time = time.time() - start
    print(f"  Vision prefill: {prefill_time:.2f}s")

    # Text decode
    print("  Running text decode...")
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated = [next_token]

    start = time.time()
    for _ in range(max_new - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated.append(next_token)
        if int(np.array(next_token)[0, 0]) == EOS_TOKEN_ID:
            break
    jax.block_until_ready(generated[-1])
    decode_time = time.time() - start

    num_gen = len(generated)
    print(f"  Text decode: {num_gen} tokens in {decode_time:.2f}s ({num_gen/decode_time:.1f} tok/s)")
    print(f"  Total (prefill+decode): {prefill_time + decode_time:.2f}s")

    all_toks = np.concatenate([np.array(t) for t in generated], axis=1)
    text_out = processor.tokenizer.decode(all_toks[0], skip_special_tokens=True)
    print(f"\n  Q: What is in this image?")
    print(f"  A: {text_out}")


def main():
    print("=" * 60)
    print("Qwen3-VL 2B — TPU v4-8 Batch Serving Benchmark")
    print("=" * 60)
    print(f"JAX: {jax.__version__}")
    print(f"Devices: {jax.device_count()} × TPU v4")

    # Load model
    print("\nLoading model...")
    config = modeling.ModelConfig.qwen3vl_2b()
    model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)

    # Load processor
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

    # Run benchmarks
    benchmark_text_generation(model, config, processor, batch_sizes=[1, 2, 4])
    benchmark_vision_generation(model, config, processor)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
