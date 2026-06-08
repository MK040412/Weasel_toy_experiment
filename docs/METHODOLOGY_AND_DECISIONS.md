# Methodology & Decisions — Fast-dVLM (GUI-Owl-2B AR → block-diffusion dVLM)

> **Read this first.** This is the centerpiece narrative for the whole project: *what* we are
> building and *why* every knob is set the way it is. It is written for a fresh Claude/engineer
> who has just pulled the repo and needs to (1) run the **reasoning-SFT** phase and then (2) the
> **AndroidWorld** eval, with full understanding of the design rationale.
>
> **Companion docs (do not duplicate — cross-linked here):**
> - `commands/REASONING_SFT_RECIPE.md` — the exact, copy-paste reasoning-SFT plan + verified HPs.
> - `commands/launch_fastdvlm_reasoning.sh` — the launch wrapper for the reasoning run.
> - `commands/CHECKPOINT_DECOUPLED.md` — how saving works under multihost + ZeRO-1 (why naive save fails).
> - `commands/tpu_v6e16_fastdvlm_zero1_recipe.md` / `V6E16_MULTIHOST_PLAYBOOK.md` — the action-SFT
>   run + multihost launch mechanics (prereqs shared with the reasoning run).
> - `docs/COORDINATE_CONVENTION.md` — the `[x,y]` 0–1000 mobile_use coordinate contract (eval-time repair classes).
>
> The loss/curriculum code referenced below lives in `scripts/train_fastdvlm_tpu.py`
> (loss: `_loss_fn` / `train_step` ~L1060–1293; host-side `lambda_fs` scaling ~L2580–2602;
> the think-then-act loader patch: `_build_episode_messages` ~L519).

---

## 1. Problem & premise

**Goal:** convert **GUI-Owl-1.5-2B** (a Qwen3-VL-2B-arch GUI agent) from a standard
**autoregressive (AR)** decoder into a **block-diffusion discrete-diffusion VLM (dVLM)**, by
**continued SFT** from `~/models/boltzmann-final` (a GUI-Owl checkpoint that has already had its
ViT surgered/aligned for this project). The **vision tower is frozen and its features are
precomputed** — we never backprop into the ViT; training only touches the language/decoder stack.
The single optimization target is **AndroidWorld** (an on-device GUI-agent benchmark): maximize
task success as a phone-controlling agent that emits native `mobile_use` tool calls.

**Why block-diffusion at all (the whole point):** an AR decoder emits one token per forward pass,
so an N-token action/CoT costs N sequential steps. A **block-diffusion** decoder denoises a *block*
of masked positions **in parallel**, committing many tokens per forward pass, so a full response
can be produced in a *few* denoising steps instead of N. For a GUI agent — which must perceive a
screenshot and act under latency pressure, step after step, for a whole episode — **few-step
parallel decode is the latency win that motivates the entire conversion.** We are not chasing a new
capability the AR model lacks; we are trying to **preserve** GUI-Owl's well-SFT'd agent behavior
while making decode cheap, and to do so at a **usable block size (bd16)** where quality is still
near-lossless (see the curriculum section: strict-JSON 0.945 @ bd16 vs 0.569 @ bd32 — bd16 is the
frontier, bd32 is past the cliff).

**Block-diffusion mechanics in one paragraph.** During training we corrupt the *response* tokens
(never the prompt/image) by masking, in contiguous blocks of size `bd ∈ {1,2,4,8,16,32}`. `bd=1`
is the AR-like easy regime (one token masked/predicted at a time, full causal context); larger
`bd` masks a whole block at once, which the student must denoise jointly — harder, but that is
exactly the few-step parallel-decode regime we want at inference. The student sees, per sample,
**two noised "pairs"** (two independent maskings) so the objective averages over corruption draws;
pair-0 is the **heavily-masked large-block** view used by the few-step KD term (see §3).

---

## 2. The 4-term loss (each term, and why its weight)

The training loss is a fixed weighted sum of four terms (flags in
`scripts/train_fastdvlm_tpu.py`; defaults shown):

```
loss = 1.0 * ce_noisy  +  0.75 * ce_clean  +  0.25 * kd_noisy  +  lambda_fs * kd_fewstep
        (--ce-noisy-weight)  (--ce-clean-weight)  (--kd-noisy-weight)  (host-scaled; see §3)
```

| term | what it is | weight | why this weight |
|---|---|---|---|
| **ce_noisy** | Cross-entropy on the **masked/noisy** block-diffusion response tokens. This is the **core diffusion objective** — the only term that directly trains the parallel-denoise capability we are converting *to*. | **1.0** | It is the primary objective, so it carries weight 1.0 by definition; every other term is auxiliary and scaled relative to it. |
| **ce_clean** | Cross-entropy on the **clean/AR** full-context response (no masking — standard next-token CE under the causal prefix). | **0.75** | Keeps the model's **AR ability intact** during the conversion. The dVLM still has a usable clean/AR branch (useful as a fallback and as the KD teacher, §3). Sub-1.0 so it supports rather than dominates the diffusion objective. |
| **kd_noisy** | Forward-KL self-distillation: `KL(clean/AR teacher ‖ noisy student)`, teacher = the **same model's clean branch with `stop_gradient`**, temperature 2.0, **averaged over BOTH mask pairs**, flat **0.25** weight for **all block sizes**. | **0.25** | A light **alignment-preservation anchor**: it tethers the noisy/diffusion branch to the well-SFT'd clean/AR behavior across *all* (mostly small) blocks. Flat and light → an anchor, not a driver. |
| **kd_fewstep** | The **same** forward-KL `KL(clean/AR teacher ‖ student)`, but on **pair-0 ONLY** (the heavily-masked, large-block "few-step" student), with a **host-side bd-scaled weight `lambda_fs`** (see §3). | **`lambda_fs`** (per-bd, ramped) | A **capability-attempt** term aimed specifically at the hard large-block regime, where the plain diffusion CE struggles most. bd-scaled so it concentrates on the large blocks (see §3 and the BARD caveat in §5). |

**Reasoning-SFT turns the two KD terms OFF** (`--kd-noisy-weight 0 --kd-fewstep-weight 0`) — see §6.
Setting `--kd-fewstep-weight 0` makes the loss **byte-identical to the 3-term form**.

---

## 3. KD self-distillation: teacher, the two KD terms, and the bd-scaling

**Teacher = the model's own clean/AR branch, `stop_gradient`'d.** There is **no external/frozen
foreign teacher network.** On each step the student runs a clean (unmasked, causal) forward and a
noisy (masked, block-diffusion) forward; the clean branch's logits — detached — are the teacher,
and the noisy branch is the student. This is **self-distillation**: teacher and student are the
**same co-evolving weights**, so the teacher improves with the student rather than being a fixed
target. KL is **forward-KL** `KL(teacher ‖ student)` at **temperature 2.0** (mass-covering: the
student must put mass everywhere the teacher does, which is the safe direction for preserving
behavior rather than mode-collapsing onto it).

**kd_noisy vs kd_fewstep — same KL, different scope:**
- **kd_noisy** is averaged over **both mask pairs** (so it sees mostly small/medium blocks too), at
  a **flat 0.25** for all block sizes → a broad, gentle alignment anchor.
- **kd_fewstep** is the *same* KL but on **pair-0 only** — the most heavily-masked, largest-block
  view (the genuine "few-step" student) — and is **bd-scaled** so it bites hardest exactly at the
  large blocks where the diffusion CE is weakest.

**The `lambda_fs` bd-scaling formula** (host-side, `scripts/train_fastdvlm_tpu.py` ~L2580–2602):

```
lambda_fs(bd, step) = lambda0 * warmup_ramp(step) * min(bd / bd_ref, bd_cap)
                    = 0.25   * min((step+1)/500, 1) * min(bd / 4, 4)
```

with `lambda0 = --kd-fewstep-weight` (action run: 0.25), `--kd-fewstep-warmup-steps 500`,
`--kd-fewstep-bd-ref 4.0`, `--kd-fewstep-bd-cap 4.0`. After warmup the per-bd weights are:

| bd | `min(bd/4, 4)` | `lambda_fs` (post-warmup) |
|---|---|---|
| 4  | 1.0  | **0.25** |
| 8  | 2.0  | **0.50** |
| 16 | 4.0  | **1.00** |
| 32 | 4.0 (capped) | **1.00** (capped) |

**Why this shape:**
- **bd-scaled (∝ bd)** → it **concentrates on the large-block tail**, which is precisely the
  few-step regime that the plain diffusion CE finds hardest and that motivates the conversion.
- **`bd_ref=4`** treats **bd4 as the lossless anchor** (weight 1.0× → 0.25 total): small blocks
  barely use it because they are already easy.
- **`bd_cap=4` (saturates at bd16)** → it is **bd16-conservative**: we do not let the term blow up
  at bd32 (which is past the quality cliff). bd16 is the eval frontier, so the term is tuned to peak
  there, not beyond.
- **500-step warmup** lets the **clean/AR teacher settle first** before the noisy student is asked
  to match it — early on the teacher is itself unstable.
- `bd` is only known **host-side** (it is the masking draw for the step), so the bd-scaling +
  warmup are folded into a single scalar `lambda_fs` computed on the host and passed into the
  jitted `train_step`; this guarantees the **same scalar on every host** so multihost stays in
  lockstep.

The `--start-step` / resume offset feeds the warmup and `--max-steps` budget so a spot-preemption
resume **continues** the ramp instead of restarting it.

---

## 4. The degree-2 Gaussian curriculum (block-size schedule)

We do **not** sample block sizes uniformly. We sample `bd ∈ {1,2,4,8,16,32}` from a **degree-2
Gaussian in log-block-size** (`--bd-curriculum degree2`):

```
P(b) ∝ exp( -lambda1 * ln b  -  lambda2 * (ln b)^2 )
```

`lambda2 = 1.04` is **fixed** (sets the width of the bell in log-b). `lambda1` **cosine-anneals
from 0 → -5.77** over `--bd-anneal-steps`. The **mode** of this distribution is

```
b* = exp( -lambda1 / (2 * lambda2) )
```

so:
- `lambda1 = 0` → `b* = exp(0) = 1` → **mode bd1** (AR-like, easy start).
- `lambda1 = -5.77` → `b* = exp(5.77 / 2.08) = exp(2.77) ≈ 16` → **mode bd16** (the EVAL block size).

So as `lambda1` cosine-anneals over training, the sampling distribution **slides AR(bd1) → bd16**:
start where the model already is (AR), and gradually shift mass to the hard parallel-decode regime
we ship at.

**Why anneal over 2 of 3 epochs, then hold bd16 (eval-centered).** For the **action run**,
`--max-steps 10122` ≈ 3 epochs (~3345 steps/epoch) and `--bd-anneal-steps 6749` ≈ 2 epochs.
So the distribution slides AR→bd16 over **epochs 1–2**, then `lambda1` is held at its end value for
**epoch 3**, which **holds the mode at bd16** and lets the model **consolidate the exact regime it
is evaluated in.** Everything is **eval-centered**: bd16 is the usable frontier (e.g. PAPER_BIB
strict-JSON **0.945 @ bd16 vs 0.569 @ bd32** — quality falls off a cliff past bd16), so the
curriculum lands and dwells precisely there rather than overreaching to bd32.

For the **reasoning run** the same shape is used scaled to a 2-epoch budget:
`--bd-anneal-steps 1500` ≈ 1 epoch of slide, then ~1 epoch holding bd16 (`--max-steps 3000`,
~1497 steps/epoch). Same eval-centering, smaller dataset.

---

## 5. The BARD comparison (arXiv 2604.16514) — a fair critique of kd_fewstep

**BARD** (concurrent Fudan work, same premise: AR → block-diffusion-of-Qwen3-VL) makes a claim we
must take seriously: **"direct autoregressive-to-diffusion distillation is poorly aligned."** The AR
teacher predicts under a **clean causal prefix**, while the diffusion student denoises under
**corrupted/masked states** — so their logits are **not directly comparable**, and matching them can
*hurt*. BARD's Table 4 (B=32) shows **AR-KD = −7.6 MMStar (HURTS)**, while a **fixed small-block
("anchor") diffusion-KD = +7.7 MMStar (HELPS)**: the fix is a teacher that is itself a diffusion
model at a *small, lossless* block size, not the AR model.

**Where this lands on us — be precise, not alarmist:**
- **kd_fewstep is structurally BARD's "poorly-aligned AR→diffusion KD" variant**, and we apply it
  **heaviest at the largest blocks** (bd16, where logit mismatch is worst) — exactly the regime BARD
  flags. This is a real, honest risk.
- **Mitigants that make our case weaker than BARD's worst-case:**
  1. It is a **0.25 auxiliary** — **CE dominates** the loss; KD only nudges.
  2. It is **self-distillation** (same co-evolving weights), **not a frozen foreign AR net** — the
     teacher tracks the student rather than pulling toward a stale, mismatched target.
  3. It is **bd16-capped** (`bd_cap=4`) — we never push it into the bd32 danger zone.
  4. It is **forward-KL (mass-covering)** — the safe, non-collapsing KL direction.

**The key distinction BARD sharpens for us — alignment-preservation vs large-block-capability:**
- The clean-branch teacher is actually a **good alignment-preservation anchor**: it keeps the dVLM
  **tethered to the well-SFT'd AR behavior**. That role fits **kd_noisy** (flat, small blocks) well
  — a light leash, BARD-safe.
- **kd_fewstep** (bd-scaled toward large blocks) is the **BARD-risky capability-attempt** form: it
  tries to *teach a new large-block skill* from a teacher that does not have it (the AR branch can't
  denoise a corrupted bd16 block). That is the structurally-misaligned move.

**Recommended next experiments (in priority order):**
1. **kd-on vs kd-off ablation at bd16.** The action run's kd_fewstep benefit is **UNVERIFIED**. Run
   the action recipe once with `--kd-fewstep-weight 0.25` and once with `0` (3-term, byte-identical
   loss otherwise) and compare strict-JSON / AndroidWorld **at bd16**. This is the single most
   important missing data point — it tells us whether kd_fewstep helps or is BARD-style harm.
2. **Frozen bd4 diffusion-anchor teacher** (BARD's *proven* fix): replace the AR teacher for the
   large-block capability term with a **frozen diffusion model at bd4** (the lossless anchor). This
   is the principled upgrade for large-block capability; keep the light clean-teacher kd_noisy as
   the alignment anchor.
3. **Token-revision decode.** BARD allows **overwriting low-confidence committed tokens** during
   denoising; **our decode is monotonic / commit-only** (once a token is committed it is never
   revised). Adding revision lets the model fix early mistakes.
4. **Mixed-noise scheduler.** Corrupt some positions with **uniform-vocab noise** (not just `[MASK]`)
   and **supervise those corrupted positions** — this trains the "fix a wrong token" skill.
   **(3) and (4) are a PAIR:** mixed-noise *trains* the fix-wrong-token skill that token-revision
   *exploits* at decode. Adopt them together.

**Bottom line:** keep the **clean-teacher kd_noisy as a light alignment anchor** (BARD-compatible
role), treat **kd_fewstep as unverified and BARD-risky** (ablate it), and for genuine large-block
*capability* prefer BARD's frozen-bd4-anchor + token-revision + mixed-noise upgrades. **Reasoning-SFT
is run kd-OFF, which is BARD-safe by construction** (§6).

---

## 6. Reasoning-SFT decisions (the NEXT phase you are running)

> Exact HPs and the copy-paste launch live in `commands/REASONING_SFT_RECIPE.md` and
> `commands/launch_fastdvlm_reasoning.sh`. This section explains the *why*.

**Data.** `KMK040412/gui-libra-reasoning-phone` (HF, 8.93GB, 11 nested parquet:
`aitw/(2) amex/(5) gui_odyssey/(4)`, glob `*/*.parquet`), **23,948 episodes / 41,876 steps**,
phone/portrait only, coords 0–1000. The `reasoning` CoT column is **100% non-empty** (~847 chars
median). Screenshots are **raw JPEG** (verified NOT base64). Crucially, it is **already distilled
from a strong reasoner (GUI-Libra ASFT)** — the CoT is teacher-quality text we learn by plain CE.

**Why distillation is OFF (`--kd-noisy-weight 0 --kd-fewstep-weight 0`).** The KD teacher is the
**model's own clean/AR branch**, and the base 2B **has no reasoning ability**. So the AR teacher
**cannot teach CoT**, and worse, its KL would **fight the CoT cross-entropy** (it would pull the
student back toward its no-reasoning clean distribution). Reasoning must be learned **only from the
CoT column via CE** (ce_noisy + ce_clean). This is also exactly what GUI-Libra Stage-1 ASFT is
(plain CE on CoT), so it is fully compatible. It is additionally **BARD-safe by construction**:
kd-off means there is no misaligned AR→diffusion KD at all. (We also reject an *external* 8B-Think
teacher: offline distillation from it = SFT on its CoT = what this dataset already is; online
co-hosting a ~9B teacher won't fit the busy 16 chips and is 2–4× slower — marginal gain, huge cost.)

**Why the think-then-act loader patch.** The episode loader previously put **only the action** in
the assistant turn, so the `reasoning` column got **zero loss**. The patch (staged on this branch,
`_build_episode_messages` ~L519) injects the CoT into the supervised assistant turn:

```python
reasoning = str(row_value(step, "reasoning", "") or "").strip()
action = native_action(step)
content_str = f"<think>\n{reasoning}\n</think>\n\n{action}" if reasoning else action
messages.append({"role": "assistant", "content": content_str})
```

`assistant_labels` masks the **whole assistant span**, so `<think>…</think>` is now supervised by
ce_noisy + ce_clean automatically. It is **backward-compatible**: a dataset with no `reasoning`
column falls back to action-only. We use the **native Qwen3-VL** tokens `<think>`=151667
`</think>`=151668 `<tool_call>`=151657 `</tool_call>`=151658 (all single tokens). **Never** use
`<answer>` — it splits into `[27,9217,29]`. *Verify after one step: decode the `labels != -100`
span and confirm it contains the `<think>` CoT.*

**Why 2 epochs is a CEILING, not a proven optimum.** The reasoning set is **small and
already-distilled**, so it is easy to **overfit the CoT templates past ~1–1.5 epochs**, while <1
epoch under-trains. So **2 epochs is a hard ceiling**, not a target. **Recommendation: checkpoint
every ~0.5 epoch (`--hf-upload-every-steps 750`) and EVAL-SELECT** the best checkpoint on
AndroidWorld — **1 epoch may already be best.** Do not assume 2.

**Mixed-noise option for the reasoning run.** Because the dataset is small, **mixed-noise** (§5,
item 4) is a reasonable upgrade here: it gives **denser supervision** (supervise corrupted positions,
not just masked ones), which helps a small dataset. **Caveat:** its full benefit needs **token-
revision at decode** (§5, item 3) — they are a pair. Without revision you get the denser training
signal but not the inference-time exploitation.

**Memory reality (do not try to "fix" by raising batch).** The memory cost is the **dual-stream
attention buffer** (`total_dual_len²`), **not** the KD terms. So turning KD off **frees no memory**:
batch stays 16 (1/chip), pair-batch 1, ctx 4096, noisy-pad 1536 (CoT lengthens the noisy branch;
loader-smoke: 12.5% of episodes exceed 1536 → handled by drop-oldest-turn, **0% skipped, 0% lose
CoT**; recipe pre-authorizes noisy-pad 2048 if there is headroom). Context length is the **known
future bottleneck** — train longer context as a separate later stage.

**Model-dir gotcha (read before launching).** `--model-dir` in the launch script is a
**PLACEHOLDER** for the best action-SFT checkpoint (from `KMK040412/fastdvlm-aw-guiowlvit`;
documented fallback `~/models/boltzmann-final`). **Shipped action checkpoints may LACK
image-processor files** (`preprocessor_config.json`, `special_tokens_map.json`,
`video_preprocessor_config.json`). Before using a checkpoint as `--model-dir` for any run that
**processes images**, **complete the processor** by copying those files from the base
`Qwen/Qwen3-VL-2B-Instruct` (or from `boltzmann-final`), or `AutoProcessor` will crash.

---

## 7. Engineering decisions

**ZeRO-1 AdamW (`--optim adamw_bf16 --shard-opt-state`).** The run is multihost data-parallel on a
**v6e-16 spot TPU** (4 hosts × 4 chips). AdamW keeps **two extra optimizer states (mu, nu)** per
param — in bf16 that is still 2× the param memory on top of the params and gradients, which **does
not fit** on the vision-constrained chips. **ZeRO-1 (`--shard-opt-state`) shards the optax mu/nu
across the dp axis** (params stay **replicated**; only the optimizer moments are sharded), which is
enough to fit. `--skip-nonfinite` guards the **rare bf16 NaN** so a single bad step cannot kill a
multi-hour spot run.

**Decoupled checkpointing (full detail in `commands/CHECKPOINT_DECOUPLED.md` — read it before
touching save code).** ZeRO-1 + multihost makes the **vocab embedding param dp-SHARDED** across all
16 chips, so a naive in-process `device_get`/allgather **fails or deadlocks** (the primary host
holds only 4/16 of the embedding; a collective on the primary alone hangs the pod). The fix
**decouples** the save: **Part 1 (in-process, deliberately DUMB)** — each host dumps only its own
local addressable shards to `/dev/shm` (pure local reads, no collective → cannot deadlock; wrapped
in try/except → a save bug can never kill training). **Part 2 (external, FIXABLE without restarting)**
— `scripts/stitch_and_ship_checkpoint.py` reassembles the shards into HF safetensors and ships them.
**If the save/ship logic has a bug, fix it and re-run on the already-dumped shards — training keeps
running, no ~14-min recompile.** (Also: save to `/dev/shm` because the worker root disk is ~97%
full; the primary is `jax.process_index()==0` = gcloud **worker 1**, not worker 0.)

**Eval-centered everything.** The through-line of every choice is **AndroidWorld at bd16**: the
curriculum modes to bd16 and dwells there (§4); kd_fewstep is bd16-capped (§3); the reasoning CoT
format uses native eval-compatible tokens (§6); and checkpoint selection is **eval-on-AndroidWorld**,
not loss-on-train. We tune for the regime we are scored in, not for training-set loss.

---

## 8. Decision → Reason (compact summary)

| Decision / knob | Setting | Reason |
|---|---|---|
| Architecture target | AR → **block-diffusion dVLM** | **Few-step parallel decode** = the latency win for a per-step GUI agent. |
| Vision tower | **Frozen + precomputed** | Only convert the decoder; keep GUI-Owl perception fixed, save memory/compute. |
| Start checkpoint | `boltzmann-final` (action) / best action-SFT ckpt (reasoning) | Continue from a well-SFT'd GUI agent; don't relearn agent behavior. |
| `ce_noisy` weight | **1.0** | Core diffusion objective — the capability we convert *to*. |
| `ce_clean` weight | **0.75** | Preserve AR ability + provide the KD teacher; support, not dominate. |
| `kd_noisy` weight | **0.25**, flat, both pairs, fwd-KL T=2 | Light **alignment-preservation anchor** to the clean/AR behavior (BARD-safe role). |
| `kd_fewstep` weight | `lambda_fs = 0.25·warmup·min(bd/4,4)`, pair-0 only | **Large-block capability attempt**; bd-scaled to concentrate on the few-step tail. |
| `kd_fewstep` bd_cap | **4 (saturate @ bd16)** | bd16-conservative; never push into the bd32 cliff. |
| KD warmup | **500 steps** | Let the clean/AR teacher settle before the student matches it. |
| Curriculum | **degree-2 Gaussian**, λ2=1.04, λ1 cosine 0→−5.77 | Mode `b*=exp(−λ1/2λ2)` slides **bd1→bd16** (easy AR start → eval regime). |
| Anneal length | **~2 of 3 epochs**, hold bd16 epoch 3 | Slide to the eval block size, then **consolidate bd16** (eval-centered). |
| Eval block size | **bd16** | Usable frontier: strict-JSON **0.945@bd16 vs 0.569@bd32** (bd32 past the cliff). |
| Reasoning KD | **OFF** (`--kd-*-weight 0`) | AR teacher has no CoT → can't teach it & KL fights CoT CE; learn CoT from data via CE. BARD-safe. |
| CoT format | `<think>…</think>` + `<tool_call>…</tool_call>` | Native single Qwen3-VL tokens; **never `<answer>`** (splits to 3). |
| Loader patch | inject CoT into supervised assistant turn | Otherwise the `reasoning` column gets **zero loss**; masked span supervises it via CE. |
| Reasoning epochs | **2 = CEILING**, eval-select | Small + already-distilled → overfit risk >1–1.5 ep; checkpoint every 0.5 ep, pick best on AW. |
| Reasoning ctx / noisy-pad | **4096 / 1536** (2048 if headroom) | CoT lengthens the noisy branch; 1536 → 0% lose CoT (drop-oldest-turn). |
| batch / pair-batch | **16 / 1** | `total_dual_len²` attention buffer caps memory; KD-off frees **no** memory. |
| Optimizer | **adamw_bf16 + `--shard-opt-state`** (ZeRO-1) | Shard optax mu/nu over dp so AdamW fits the vision-constrained chips. |
| Stability | **`--skip-nonfinite`** | Guard rare bf16 NaN on a long spot run. |
| Checkpointing | **decoupled** (dumb in-proc dump + external stitch) | dp-sharded embedding breaks naive save; external part is fixable without recompile. |
| Save target | **`/dev/shm`** | Worker root disk ~97% full; tmpfs avoids ENOSPC. |
| Checkpoint upload cadence | **every 750 steps** (`--hf-upload-every-steps`) | ~0.5-epoch granularity for **eval-select** on AndroidWorld. |
| Recommended next exp. | **kd-on vs kd-off @ bd16**; frozen-bd4 anchor; token-revision; mixed-noise | kd_fewstep benefit is **unverified** + BARD-risky; these are the principled upgrades. |
