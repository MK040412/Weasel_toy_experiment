# vla/models/ — 모델 컴포넌트

## 파일

### `vla.py` — VLAPolicy

VLM encoder + action expert를 연결하는 policy wrapper.

```python
class VLAPolicy(nnx.Module):
    vlm: Qwen3VLForConditionalGeneration | None  # frozen, 캐시 후 None 가능
    obs_proj: Linear(2048 → 1536)                 # VLM hidden → action expert dim
    action_expert: GemmaActionExpert              # ~311M params
```

- `encode_observations()`: VLM forward → obs_proj → (B, seq, 1536)
- `predict_actions()`: denoise → (actions, gripper_probs)

### `action_expert.py` — GemmaActionExpert

pi0-style action denoising transformer.

```
Input:
  obs_embed  (B, n_obs, 1536)    — VLM에서 온 관측 임베딩
  noisy_acts (B, 50, 6)          — 노이즈 섞인 continuous actions
  timestep   (B, 50, 1)          — per-token flow matching timestep

Layers:
  action_in_proj(6 → 1536) + timestep_mlp(1 → 1536)
  12 × GemmaDecoderLayer (GQA 12/4, SwiGLU)
  RMSNorm → action_out_proj(1536 → 6)   [velocity prediction]
           → gripper_head(1536 → 1)      [BCE logits]
```

주요 메서드:
- `forward_joint(obs, noisy, t)` — 학습용, prefix-LM mask 사용
- `build_prefix_kv_cache(obs)` — 추론용, obs KV 캐시 생성
- `forward_cached(noisy, t, kv_cache)` — 추론용, 캐시된 obs로 빠른 forward
- `denoise(obs, chunk_size, n_steps, rng)` — Euler denoising (t=1→0)

### `layers.py` — Transformer 빌딩 블록

- `RMSNorm` — learnable scale, pre-norm
- `GQAttention` — Grouped Query Attention (12 heads, 4 kv-heads), optional prefix_kv cache
- `GatedMLP` — SwiGLU: `down(SiLU(gate(x)) * up(x))`
- `GemmaDecoderLayer` — pre-norm residual: RMSNorm→GQA→residual→RMSNorm→MLP→residual

## Attention Mask (Prefix-LM)

```
         [obs_tokens | action_tokens]
obs      [    T     |      F       ]   ← obs는 action을 못 봄
action   [    T     |      T       ]   ← action은 전부 봄
```
