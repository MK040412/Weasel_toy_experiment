# Paper bib links — for the W0 paper-writing Claude

This file bridges the **code** in this repo to the **W0 paper** (`/home/perelman/W0Tech/paper/main.tex`,
bib `w0_references.bib`). It exists because `kd_fewstep` (the step-axis self-distillation term
added in `scripts/train_fastdvlm_tpu.py`) and the `degree2` block-size curriculum were implemented
**after** the current paper draft — the paper does not yet describe `kd_fewstep`. Use the keys below
to cite; add the "to-add" entries if you write up `kd_fewstep`.

## What is already in the paper (correct, reuse as scaffolding)

- **degree-2 Gaussian-in-log-b curriculum** `P(b) ∝ exp(-λ1·ln b - λ2·(ln b)²)` — paper Prop. ~L1920-1941.
  Now implemented: `degree2_bd_probs()` + `--bd-curriculum degree2`. λ2=0 ⇒ Boltzmann power law `b^{-λ1}` (paper ~L1832-1861).
- **`b*` critical block size** — paper finding ~L975-994: strict-JSON validity 1.000@b1, 0.952@b4,
  0.945@b16, **0.569@b32**; "b=16 holds, b=32 breaks", usable frontier ≈16.
- **b32 latency headline** — 1290ms→461ms (2.8×), paper abstract/Fig.1 ~L123-125.
- **BARD-style stage-wise distillation from a small-block anchor (b≤4)** — paper Remark ~L1963-1967.

## Existing bib keys (in `w0_references.bib`) relevant to this code

**Block / masked diffusion (the dLLM action head):**
- `arriola2025bd3lm` — Block Diffusion: Interpolating Between AR and Diffusion LMs — https://arxiv.org/abs/2503.09573
- `sahoo2024mdlm` — Simple and Effective Masked Diffusion Language Models — https://arxiv.org/abs/2406.07524
- `lou2024sedd` — Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution — https://arxiv.org/abs/2310.16834
- `austin2021d3pm` — Structured Denoising Diffusion Models in Discrete State-Spaces — https://arxiv.org/abs/2107.03006
- `wu2026fastdvlm` — Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from AR VLM — https://arxiv.org/abs/2604.06832
- `wu2025fastdllm` — Fast-dLLM: Training-free Acceleration of Diffusion LLM (KV cache + parallel decode) — https://openreview.net/forum (ICLR 2026)
- `wu2025fastdllmv2` — Fast-dLLM v2: Efficient Block-Diffusion LLM — https://openreview.net/forum?id=1NZ3DHF9nT
- `liang2025discretevla` — Discrete Diffusion VLA: discrete diffusion for action decoding — https://arxiv.org/abs/2508.20072

**Distillation / curriculum (what `kd_fewstep` sits next to):**
- `bard2026` — BARD: Bridging AR and Diffusion VLMs via Progressive Block Merging + **Stage-Wise Distillation** — https://arxiv.org/abs/2604.16514
- `salimans2022progressive` — Progressive Distillation for Fast Sampling of Diffusion Models — https://arxiv.org/abs/2202.00512

**Backbone / GUI agents:**
- `bai2025qwen3vl` — Qwen3-VL Technical Report — https://arxiv.org/abs/2511.21631
- `xu2026mobileagent35` — Mobile-Agent-v3.5 (GUI-Owl-1.5 family) — https://arxiv.org/abs/2602.16855
- `ye2025mobileagentv3` — Mobile-Agent-v3 — https://arxiv.org/abs/2508.15144
- `black2024pi0` — π₀: VLA flow model (action-head lineage) — RSS 2025

## To ADD if writing up `kd_fewstep` (NOT yet in `w0_references.bib`)

`kd_fewstep` is a **step-axis** self-distillation term. Its lineage (cite when describing it):
- **SDTT** — Deschenaux & Gulcehre, "Beyond Autoregression: Fast LLMs via Self-Distillation Through Time", ICLR 2025 — https://arxiv.org/abs/2410.21035
- **Diffusion Duality (Duo)** — Sahoo et al., "The Diffusion Duality", ICML 2025 — https://arxiv.org/abs/2506.10892
- **Consistency Models** — Song et al., ICML 2023 — https://arxiv.org/abs/2303.01469
- **Knowledge Distillation** — Hinton et al., "Distilling the Knowledge in a Neural Network", 2015 — https://arxiv.org/abs/1503.02531

## `kd_fewstep` in paper notation (for the methods section)

`kd_fewstep` is novel vs `bard2026`: BARD distills from a fixed small-block anchor across stages;
ours distills the model's **own clean/AR branch (stop-grad)** into the **pair-0 large-block,
few-effective-step diffusion student**, bd-weighted and **b16-targeted**. Suggested notation:

```latex
% step-axis self-distillation; clean/AR branch is the teacher (= b<=4 anchor by our own evidence)
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{noisy}}
            + \tfrac{3}{4}\,\mathcal{L}_{\mathrm{CE}}^{\mathrm{clean}}
            + \tfrac{1}{4}\,\mathcal{L}_{\mathrm{KD}}^{\mathrm{noisy}}
            + \lambda_{\mathrm{fs}}(b)\;\mathcal{L}_{\mathrm{KD\text{-}step}}
\qquad
\lambda_{\mathrm{fs}}(b) = \lambda_0 \cdot \min\!\big(b/b_{\mathrm{ref}},\, c\big),\;
b_{\mathrm{ref}}=4,\; c=4\ (\text{b16-cap})
```
where `L_{KD-step} = KL(p^{clean}_{\mathrm{AR}} \,\|\, p^{noisy}_{\text{pair-0}})` at temperature `kd_temp`,
`stop_gradient` on the teacher. Implemented in `dual_stream_loss_jax` (`scripts/train_fastdvlm_tpu.py`),
weight scheduled host-side in the dispatch loop. Evidence target: recover strict-JSON@b16/b32 (paper ~L975-994).
