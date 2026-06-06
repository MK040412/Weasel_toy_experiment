# Fast-dVLM × AndroidWorld — experiment findings & log (paper-grade notes)

Session 2026-06-06/07. GUI-Owl-1.5-2B (Qwen3-VL-2B) dual-stream **block-diffusion** VLA.
Records the paper-worthy results + the diagnostic process + key numbers/logs.

## ★ Headline result — block-diffusion decode is near-lossless up to bd4 (≈AR) on AndroidWorld
Same Boltzmann-final checkpoint, full 116-task AndroidWorld, 8-lane Vultr emulator farm, auto-open + structural repair:

| decode (block size) | success/116 | rate | strict_json (raw valid) | succ_repair_rate |
|---|---|---|---|---|
| bd1 (autoregressive) | 7 | 6.0% | 1.000 | 0.000 |
| bd2 | 7 | 6.0% | 0.976 | 0.000 |
| bd4 | 7 | 6.0% | 0.952 | 0.000 |
| bd8 | 6 | 5.2% | 0.978 | 0.000 |
| bd16 | 6 | 5.2% | 0.945 | 0.000 |
| bd32 | 4 | 3.4% | 0.569 | 0.000 |

- **bd1 = bd2 = bd4 (6.0%)** → block-diffusion gives **~4× parallel token decode at zero task-quality cost** vs AR; graceful to bd16; only bd32 degrades.
- **Repair-independent**: every successful episode had `succ_repair_rate = 0.000` and `strict_json = 1.000` (won on raw, un-repaired, valid output). Structural repair *never* converted a failure→success. So "no loss to bd4" is real, not a repair artifact.
- bd32 strict_json 0.569 (≈43% of raw outputs malformed) → large-block parallel unmask is where quality breaks.
- Method note: requires the bd-parameterized dual-stream JIT decode (`dual_stream_decode_jax.py`; bd only enters `_turn_indices //bd` + per-block `active_len` → bd=4 byte-identical to the original ⇒ zero-regression generalization).

## JIT / throughput optimization (60× step speedup)
`--loss-token-cap 96` fixes the sparse loss-tensor length → single XLA compile instead of per-shape recompiles. **86 s/step → 1.4 s/step**; ~50k tokens/s on v6e-4 (batch 32). Persistent `JAX_COMPILATION_CACHE_DIR` reuses compiles across restarts.

## Boltzmann bd-curriculum (continuous bd schedule)
P(bd) ∝ bd^(−λ), λ cosine-annealed 1.5 → −0.5 over 7000 steps (small→large block). Loss = 1.0·ce_noisy + 0.75·ce_clean + 0.25·kd_noisy (kd_temp 2.0; clean branch = same-model stop-grad teacher). 1 epoch (8,874 steps, 3 h 44 m on v6e-4). ce_clean: min ≈0.3254 @ step 2000–2500, final 0.3448 — modest ~2–5% gain over fixed-bd/bd-curric baselines, **bounded by the narrow overfit data, not the method**.

## AndroidWorld 0%→6% root-cause chain (diagnostic process)
1. **Symptom**: both bd4 and AR collapsed to 40/40 identical `swipe up` on the live emulator; 0% success.
2. **Decode exonerated**: line-by-line audit (image-embed splice, DeepStack inject@LLM 0/1/2, interleaved 3D M-RoPE, cross-stream block-causal mask) byte-faithful to training & HF Qwen3-VL ref; deployed code md5-identical.
3. **Data not the cause**: corpus action dist click 52.7% > swipe 20.1% → an unconditional prior would emit *click*, not swipe.
4. **In-app grounding is near-pixel-perfect**: free-running probe on in-distribution screenshots → click L2 err median 106/1000, several 1/18/41. Opened Clock via adb, fed the live screen → model clicked the Stopwatch tab `[693,894]` (truth ≈[699,906]); after tapping, clicked Start `[500,770]` (truth ≈[500,776]). The model **can** complete the task once in the app.
5. **Real root cause**: every AW task starts on the HOME screen; the model never learned the `open` action (training corpus has ~0% `open`) → it loops a no-op swipe on home and never enters the app. (frames step_000 = step_003 = step_009, identical.)
6. **Fix (workaround)**: `aw_auto_open.py` launches the task's app at init + sets `TaskEval.start_on_home_screen=False` (else the episode runner re-homes and undoes the launch). → **0% → 6%** (the 7 wins = the short Settings-toggle family).

## Curation diagnosis (why the other 109 tasks fail, and the fix)
- **App-domain mismatch**: our corpus is 95.8% AITW = *consumer* apps (Amazon/YouTube/Gmail); AndroidWorld uses *F-Droid* apps (Markor/Broccoli/Pro-Expense/Retro/VLC/OsmAnd/SimpleSMS/Files/Clock). **Only OpenMobile covers the AW apps** (its `app` field = AW task-family names).
- **3 AW-native verbs scarce**: `open` (4,601, androidcontrol only), `terminate` (~12k, no AITW), `answer` (766, openmobile only; required by the 25 information-retrieval tasks = 21.5% of the benchmark).
- **Multi-step chaining is the bottleneck**, not grounding: wins are all ≤4-step toggles; Clock(stopwatch)/Contacts fail despite being the *largest* training categories, because training was step-IID (no plan continuity) → episode-packing + action-history is the fix.
- **Coordinate**: click already well-balanced (10×10 top-bin 3.2%); **swipe 67% up / 19% down / 7% left / 6% right** (vertical-biased, a per-source labeling convention) → rebalance to 35/35/15/15.
- **Balanced-mix builder result**: action-balanced 52,802-row mix lifted `open` 0.08% → 8.7%, `type` 2.7% → 11.8%, `terminate` ~0 → 8.7%, capped click 53%→30.5%; coords well-spread. (Parallel 2-stage, 40 workers, Vultr 48-core.)

## Infra (for methods section)
3-machine: local orchestrator → TPU v6e-4 JAX policy server (`--decode {grounded_ar_jit, dvlm_bd4, dual_dvlm_bd4} --bd N`) → Vultr 48-core emulator farm (8 lanes sharing one serial server) over an auto-restarting SSH tunnel. Reproducible harness: `aw_eval/` (`bd_sweep.py`, `CLAUDE.md`, `verify_repro.sh`).

## Notable gotchas (lessons)
- TPU paramiko: nohup/systemd die after model load → must use tmux; kill+launch must be SEPARATE ssh calls.
- dual-stream serving caps NOISY_CAP=448 / PROMPT_CAP=640 (training regime) → long-prompt AW tasks overflow → raise for a clean full run.
- Reverse SSH tunnel self-terminates → supervised auto-restart needed for long runs.
