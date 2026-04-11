"""Qwen3.5-0.8B text inference test on TPU v4-8."""

import os
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from model35 import modeling

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen35-0.8b"))
HF_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def generate(model, cache, input_ids, tokenizer, max_new_tokens=100):
    generated = []
    logits, cache = modeling.forward(model, cache, input_ids)
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated.append(next_token)

    for _ in range(max_new_tokens - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated.append(next_token)
        tok_id = int(np.array(next_token).flat[0])
        if tok_id == tokenizer.eos_token_id:
            break

    all_toks = np.concatenate([np.array(t) for t in generated], axis=1)
    return all_toks


def main():
    print("=" * 60)
    print("Qwen3.5-0.8B — JAX/TPU v4-8 Text Inference")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")

    # Load
    print("\n1. Loading model...")
    start = time.time()
    config = modeling.ModelConfig.qwen35_0_8b()
    model = modeling.Qwen35ForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
    print(f"   Loaded in {time.time()-start:.2f}s")

    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    # Test text generation
    prompt = "Explain what a neural network is in 2 sentences."
    print(f"\n3. Prompt: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = jnp.array(tokenizer(text, return_tensors="np")["input_ids"])
    seq_len = input_ids.shape[1]
    print(f"   Input: {seq_len} tokens")

    cache = modeling.init_cache(config, batch_size=1, token_len=seq_len, gen_steps=200)

    print("\n4. Generating (includes JIT compile)...")
    start = time.time()
    output_ids = generate(model, cache, input_ids, tokenizer, max_new_tokens=100)
    elapsed = time.time() - start
    num_gen = output_ids.shape[1]
    print(f"   Generated {num_gen} tokens in {elapsed:.2f}s ({num_gen/elapsed:.1f} tok/s)")

    text_out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n--- Output ---")
    print(text_out)
    print("-" * 40)

    # Second run (JIT cached)
    print("\n5. Second generation (JIT cached)...")
    prompt2 = "What is 2+2? Answer with just the number."
    messages2 = [{"role": "user", "content": prompt2}]
    text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    input_ids2 = jnp.array(tokenizer(text2, return_tensors="np")["input_ids"])

    # Only works if same seq_len (JIT recompile otherwise)
    cache2 = modeling.init_cache(config, batch_size=1, token_len=input_ids2.shape[1], gen_steps=200)
    start = time.time()
    output_ids2 = generate(model, cache2, input_ids2, tokenizer, max_new_tokens=30)
    elapsed2 = time.time() - start
    print(f"   {output_ids2.shape[1]} tokens in {elapsed2:.2f}s ({output_ids2.shape[1]/elapsed2:.1f} tok/s)")
    print(f"   A: {tokenizer.decode(output_ids2[0], skip_special_tokens=True)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
