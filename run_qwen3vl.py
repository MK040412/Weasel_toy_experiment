"""Qwen3-VL 2B inference on TPU v4-8."""

import os
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor

from model import modeling

MODEL_ID = os.environ.get("QWEN3VL_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen3-vl-2b"))
HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"  # For processor
EOS_TOKEN_ID = 151643


def generate_with_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids,
                         max_new_tokens: int = 50):
    generated_tokens = []
    logits, cache = modeling.forward_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids)
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated_tokens.append(next_token)

    for _ in range(max_new_tokens - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated_tokens.append(next_token)
        if int(np.array(next_token)[0, 0]) == EOS_TOKEN_ID:
            break

    all_tokens = [np.array(input_ids)] + [np.array(t) for t in generated_tokens]
    return jnp.array(np.concatenate(all_tokens, axis=1))


def main():
    print("=" * 60)
    print("Qwen3-VL 2B - JAX/TPU v4-8 Inference")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")

    # Load model
    print("\n1. Loading pretrained model...")
    start = time.time()
    config = modeling.ModelConfig.qwen3vl_2b()
    model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, config=config)
    load_time = time.time() - start
    print(f"   Model loaded in {load_time:.2f}s")

    # Load processor
    print("\n2. Loading processor...")
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

    # Prepare image input
    print("\n3. Preparing image input...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"},
                {"type": "text", "text": "What is in this image? Answer briefly."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )

    input_ids = jnp.array(inputs["input_ids"].numpy())
    seq_len = input_ids.shape[1]
    print(f"   Input tokens: {seq_len}")

    if "pixel_values" in inputs:
        pixel_values = jnp.array(inputs["pixel_values"].numpy())
        image_grid_thw = jnp.array(inputs["image_grid_thw"].numpy())
        token_type_ids = (input_ids == config.image_token_id).astype(jnp.int32)
        print(f"   Pixel values shape: {pixel_values.shape}")
        print(f"   Image grid THW: {image_grid_thw}")
        print(f"   Image tokens: {int(token_type_ids.sum())}")

        # Init cache
        cache = modeling.init_cache(config, batch_size=1, token_len=seq_len, generate_steps=200)

        # Generate
        print("\n4. Generating...")
        start = time.time()
        generated_ids = generate_with_vision(model, cache, input_ids, pixel_values,
                                             image_grid_thw, token_type_ids, max_new_tokens=100)
        gen_time = time.time() - start
        num_new = generated_ids.shape[1] - seq_len
        print(f"   Generated {num_new} tokens in {gen_time:.2f}s ({num_new/gen_time:.1f} tok/s)")

        # Decode
        generated_trimmed = generated_ids[:, seq_len:]
        output_text = processor.batch_decode(
            np.asarray(generated_trimmed), skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

        print("\n" + "-" * 40)
        print(f"Q: What is in this image?")
        print(f"A: {output_text[0]}")
        print("-" * 40)
    else:
        print("   ERROR: No pixel_values in processor output")

    print("\nDone!")


if __name__ == "__main__":
    main()
