# AndroidWorld base-capability CURATION PLAN (canonical)

_Generated 2026-06-07 from aw-curation-analysis workflow (AW 116-task analysis + corpus app/episode-length/coordinate measurement). See project_aw_bd_sweep_pipeline + project_aw_eval_domain_collapse memories._

This is a pure analysis/strategy task — I have all the data I need in the prompt. No code to write, so no delegation needed. Let me produce the canonical curation plan directly, tying every number to the measured data.

# Canonical Curation Plan — GUI-Owl-1.5-2B Block-Diffusion VLA for AndroidWorld (116 tasks)

## Root-cause framing (drives every number below)
The proven bottleneck is **multi-step chaining**, not grounding: the model grounds single steps but cannot chain, so it only passes short toggles. AndroidWorld tasks **all start on home** and need `open` (open-app) at the start and `terminate`/`answer` at the end. Our corpus is **95.8% AITW** (single-app, mean 8 steps, no `open`/`terminate` discipline), and the two AW-native action verbs are nearly absent: `open` lives only in androidcontrol_par (~4.6k), `terminate` only in amex/odyssey/openmobile (~12k total), `answer`/`long_press` tiny. So the dominant failure is structural (episode shape), and the plan is built to fix episode shape first, app/coordinate coverage second.

---

## (1) COVERAGE GAPS vs AndroidWorld

### App-category gaps (AW need → corpus signal)
| AW category (count) | AW need | Corpus coverage | Gap severity |
|---|---|---|---|
| productivity = 60 (markor 14–18, broccoli 13, pro expense 9, calendar 9, +25 info-retrieval reads) | note CRUD, recipe/expense form-fill, calendar create, **final `answer` for 25 IR tasks** | openmobile has the AW task families directly (Recipe/Expense/Notes/Tasks/Calendar templates); AITW/odyssey have generic Calendar/Docs but **none of the F-Droid third-party apps** | **CRITICAL.** markor/broccoli/pro-expense are non-standard F-Droid apps absent from AITW/odyssey/androidcontrol. Only openmobile touches them. |
| settings-toggle = 17 (all complexity 1.0) | wifi/bt/brightness toggles + 2 composite | AITW Settings (25.6k+3.8k) + odyssey "Setting" pairs heavily | **COVERED / over-covered** — these are exactly the short tasks we already pass. Do NOT up-weight. |
| media = 13 (retro music 4, audio recorder 2, camera 2, vlc 2, draw 1, gallery 1) | playlist building (multi-step), record, capture | openmobile has Retro/Vlc/AudioRecorder/Draw/SaveReceipt templates; AITW has YouTube/Vimeo (wrong apps) | **HIGH.** Retro/VLC playlist building (long-horizon) only in openmobile. |
| messaging = 6–7 (Simple SMS Messenger) | reply/send/resend, clipboard→SMS | openmobile SMS templates (SimpleSmsReply=570, SendReceivedAddress=454); AITW has Gmail not SMS | **MEDIUM.** Covered by openmobile only. |
| browser = 3 (Chrome JS mini-games: Draw/Maze/Multiply) | in-page canvas/JS interaction | AITW Chrome is huge (33.8k+16.5k) but it's URL/search navigation, **not in-page game mechanics** | **MEDIUM.** Volume present, task-type mismatched. |
| misc = 9 (osmand 3, OpenApp, clipboard, cross-app) | maps marker/track, **OpenAppTaskEval = pure open-app**, clipboard | osmand only in openmobile; clipboard tiny; open-app only androidcontrol ~4.6k | **HIGH** for OsmAndTrack (complexity 12) + clipboard. |
| contacts = 2, clock = 3, files = 2 | add contact, stopwatch/timer, delete/move file | weak/absent in named sources | **LOW count but ZERO corpus signal** — file manager + clock not in any top-app list. |

### Action-type gaps (the structural killers)
- **`open` (open-app):** AW needs it at the **start of all 116 tasks** (all begin on home). Corpus has it in androidcontrol_par only (~4.6k steps). **Severe under-representation** relative to "every episode should start with open."
- **`terminate`:** AW needs it to **end every action task**; only ~12k in amex/odyssey/openmobile. **Severe.**
- **`answer`:** AW needs it as the **final action for all 25 information-retrieval tasks** (Calendar 9, Tasks 6, SportsTracker 6, Notes 4). Corpus `answer` is "tiny." **Severe — 21.5% of the benchmark cannot be completed without it.**
- **`type`:** heavy text entry for markor/expense/recipe/calendar titles+descriptions. Present but **not over-sampled**; AITW is click-dominated. **Under-weighted vs need.**
- **`scroll`/swipe:** needed for long lists (calendar/tasks/recipe/expense). Present (68k swipes) but **direction-biased by source convention** (see §4).
- **`long_press`:** only 1,242 samples and **22.8% concentrated in one bin** [9,8]. Needed for some delete/context-menu flows. **Genuine imbalance, low volume.**

### Horizon gap
AW has **~13 long-horizon tasks** (complexity 4–12: OsmAndTrack 12, MarkorMergeNotes 7.8, all Expense/Recipe-AddMultiple = 6, RetroSavePlaylist 5, VlcCreateTwoPlaylists 4.8). These require open→search→scroll→multi-form-fill→save, often cross-app (gallery→expense, markor→recipe). Corpus mean episode lengths (androidcontrol 5.5, AITW 8) are **too short**; only gui_odyssey (14.7), amex (12.86), openmobile (9.88) reach this regime. **Long-multi-step + cross-app trajectories are under-represented relative to the difficulty mass.**

---

## (2) SOURCES TO DRAW FROM + relative weight + where NEW data is required

**Relative pull weight (per-step sampling multiplier vs natural corpus share). Natural share is 95.8% AITW; we invert this.**

| Source | Natural role | Pull weight | Why (tied to data) |
|---|---|---|---|
| **openmobile** | **Anchor — only AW-native source** | **HIGHEST (≈8–10×)** | n_distinct=116 = exact AW task templates; the ONLY source with markor/broccoli/pro-expense/retro/vlc/osmand/SMS/Notes/Tasks. p50=9, p90=18 multi-step. This is the keystone for both app-coverage and chaining. |
| **gui_odyssey** | **Cross-app + long-horizon + horizontal-swipe donor** | **HIGH (≈3–4×)** | mean 14.7 steps (richest), p90=23; the only source with balanced swipe directions (43% up/36% left/16% right) and explicit cross-app pairs. Matches the 13 long/cross-app AW tasks. |
| **amex** | **Long-horizon + `terminate` donor** | **MEDIUM (≈2–3×)** | mean 12.86, p90=23, has `terminate`. Caveat: `app` column empty → use only for episode-shape/length and terminate discipline, not app coverage. |
| **androidcontrol_par** | **`open`-app donor + consumer-app breadth** | **MEDIUM (≈2×) but targeted** | Sole source of `open` (~4.6k) — critical for "start on home." Short (p50=5) so weight its *open-containing* episodes, not bulk. |
| **aitw** | **Grounding ballast — DOWN-weight hard** | **LOW (≈0.1–0.15×)** | 95.8% of corpus, single-app, click-heavy, wrong apps (Chrome/Maps/Gmail). Keep enough for click-grounding generalization but **cap to ~30–35% of final mix** (down from 95.8%). |

### Where we need NEW data (beyond the 5 sources)
1. **Home-screen `open`-app launches** for the specific AW F-Droid apps (markor, broccoli, pro expense, simple calendar pro, retro music, vlc, simple sms, osmand, files, clock, contacts). Synthesize/collect short "from home → open <app>" episodes — directly attacks the "all tasks start on home" + open-action gap.
2. **`answer`-terminated information-retrieval episodes** over Calendar/Tasks/SportsTracker/Joplin seeded data (the 25 IR tasks = 21.5% of bench). openmobile has the *templates* but corpus `answer` is tiny — **we need answer-ending trajectories explicitly.**
3. **Long-horizon cross-app form-fill** for the 13 hard tasks (gallery→expense, markor→recipe, merge-notes, osmand-track). Even openmobile p90=18 < required depth for complexity-12 tasks; collect/bootstrap targeted long episodes (preferably via successful rollouts on the actual apps).
4. **In-page Chrome JS mini-game** trajectories (Draw/Maze/Multiply) — AITW Chrome volume is navigation, not canvas mechanics.
5. **File-manager + Clock** episodes — zero corpus signal for Files (delete/move) and Clock (stopwatch/timer).

---

## (3) EPISODE-LEVEL BALANCING TARGETS (concrete floors/caps)

**Assume target corpus = N final training steps (sizing in §5). Targets expressed as shares of final steps unless noted.**

### A. Action floors / caps (fix the structural bottleneck)
| Action | Current | Target floor/cap | Rule |
|---|---|---|---|
| `open` (open-app) | ~4.6k, 1 source | **FLOOR ≥ 1 per episode where AW-style; ≥ 6% of all steps** | Every packed AW-style episode must begin with `open`. Up-sample androidcontrol open-episodes + synthesize. |
| `terminate` | ~12k | **FLOOR: ≥ 1 per action-episode; ≥ 5% of steps** | Every non-IR episode ends in `terminate`. |
| `answer` | tiny | **FLOOR ≥ 4–5% of steps; ≥ 1 per IR episode** | Mandatory final action for IR-style episodes; target ≥ 25/116 proportion of AW-aligned episodes end in `answer`. |
| `type` | present, low | **FLOOR ≥ 15% of steps** | Heavy text entry need (markor/expense/recipe/calendar). Up-sample type-bearing steps. |
| `swipe`/scroll | 68k | **FLOOR ≥ 12% of steps**, with 4-dir balance (§4) | Long-list scrolling. |
| `long_press` | 1,242 | **FLOOR ≥ 1% of steps; CAP single coord-bin ≤ 15%** | Raise volume, break the [9,8] 22.8% spike. |
| `click` | 237k (dominant) | **CAP ≤ 45% of steps** | Prevent AITW click-bias from dominating; currently the easy path. |

### B. App-category floors (re-weight toward AW difficulty mass, NOT raw AW counts)
Target the **difficulty-weighted** mix, since settings already passes. Floors as share of AW-aligned episodes:
| AW category | AW raw share | Target episode floor | Rationale |
|---|---|---|---|
| productivity (markor/broccoli/expense/calendar + IR) | 52% | **≥ 45%** | Largest + hardest; the apps only openmobile covers. |
| media (retro/vlc/audio/camera/draw/gallery) | 11% | **≥ 12%** | Multi-step playlist building under-covered. |
| messaging (SMS) | 6% | **≥ 7%** | SMS apps only in openmobile. |
| misc (osmand/open-app/clipboard) | 8% | **≥ 8%** | Contains OpenAppTaskEval + osmand-track (complexity 12). |
| browser / files / clock / contacts | 8% | **≥ 8% combined, each ≥ 1.5%** | Zero-signal apps need a guaranteed floor. |
| settings-toggle | 15% | **CAP ≤ 8%** | Already solved; do not waste budget. |

### C. Episode-length up-weighting (fix chaining)
Current length means: gui_odyssey 14.7 > amex 12.86 > openmobile 9.88 > aitw 8.0 > androidcontrol 5.51.
- **Up-weight episodes by length tier.** Sampling multiplier by trajectory length L:
  - L ≤ 4 (short toggles): **×0.5** (we already pass these; AITW/androidcontrol short tail).
  - 5 ≤ L ≤ 9: **×1.0**.
  - 10 ≤ L ≤ 18: **×2.0** (the openmobile/odyssey core).
  - L ≥ 19 (≥ p90 of odyssey/amex): **×3.0** — directly feeds the complexity-4–12 AW tasks.
- **Long-episode share target:** **≥ 30% of training steps must come from episodes with L ≥ 10**, and **≥ 12% from L ≥ 19**. (Currently long episodes are a small minority because AITW@8 dominates 95.8%.)
- **Tail hygiene:** spot-check and drop/clip androidcontrol max=91 and AITW max=112 outliers before packing (flagged as likely concatenated/noisy).

---

## (4) COORDINATE DE-BIASING RULE

**Click — already well-balanced; light cap only.**
- Measured: pooled top_bin_share 3.19%, per-source 3.5–5.4%, IQR spans most of screen. The 3 hot bins ([0,0]=3.2%, [9,0]=2.7%, [1,1]=2.6%) are legitimate UI affordances (back/nav/action).
- **Rule:** per-action **coord-bin cap on a 10×10 grid ≤ 6%** of that action's samples per cell (just above the natural 5.4% per-source max, so it trims only pathological packing, not real affordances). No reweighting needed otherwise.

**Long_press — fix the genuine spike.**
- Measured: 22.8% in bin [9,8]; only 1,242 samples.
- **Rule:** hard **cap any single long_press coord-bin ≤ 15%**; require long_press samples spread over **≥ 6 distinct bins**; raise total long_press volume toward the ≥1% step floor (§3A).

**Swipe — enforce 4-direction balance (the big fix).**
- Measured pooled: up 67.1% / down 19.5% / left 7.2% / right 6.3% (vertical 86.5%). This is a **per-source labeling-convention artifact** (androidcontrol 78.6% down; amex 78.2% up; openmobile 87.8% up; gui_odyssey balanced 43/36/16/5).
- **Rule (target distribution): up 35% / down 35% / left 15% / right 15%** (±5% tolerance), enforced **globally after resampling**, with **vertical:horizontal = 70:30** as the hard band.
  - Achieve by **down-weighting the single dominant vertical direction within each non-odyssey source** (cap each source's dominant swipe direction so it contributes ≤ 45% of that source's swipes) and **up-weighting gui_odyssey swipes** (the only horizontal-diverse source) to ≈ 2× to supply left/right mass.
  - Do **not** flip/augment labels blindly (convention differs by source); rebalance by sampling, preserving each episode's internal label semantics.

---

## (5) RECOMMENDED TOTAL SIZE + COMPOSITION TABLE

**Total target: ~1.0M training steps** (down-sample from the 5.93M raw; quality/shape over volume — we are fixing episode shape, not data starvation). Compose by *role*, not natural share.

| Source | Role | Final step share | Approx steps | Effective multiplier vs natural | Notes |
|---|---|---|---|---|---|
| **openmobile** | AW-native anchor (apps + chaining + templates for answer/SMS/recipe/expense) | **30%** | ~300k | ≈ ×8–10 (natural ~0.46%) | Up-sample L≥10 episodes; primary source of app-category + answer floors. |
| **gui_odyssey** | Cross-app, long-horizon, horizontal-swipe donor | **18%** | ~180k | ≈ ×3–4 | Supplies L≥19 mass + left/right swipes. |
| **amex** | Long-horizon shape + `terminate` discipline | **10%** | ~100k | ≈ ×2–3 | Episode-shape only (app col empty). |
| **androidcontrol_par** | `open`-app donor + consumer-app breadth | **9%** | ~90k | ≈ ×2 (targeted to open-episodes) | Weight open-containing episodes, not bulk short. |
| **aitw** | Click-grounding ballast (DOWN-weighted) | **18%** | ~180k | ≈ ×0.03 (natural 95.8%) | Hard cap; keep for grounding generalization only. |
| **NEW: home→open launches (AW F-Droid apps)** | Fix "start on home" + open floor | **6%** | ~60k | new | markor/broccoli/expense/calendar/retro/vlc/sms/osmand/files/clock/contacts. |
| **NEW: answer-terminated IR episodes** | Fix the 25 IR tasks (21.5% of bench) | **5%** | ~50k | new | Calendar/Tasks/SportsTracker/Joplin reads → final `answer`. |
| **NEW: long cross-app form-fill + Chrome-game + file/clock** | Fix the 13 hard tasks + zero-signal apps | **4%** | ~40k | new | gallery→expense, markor→recipe, merge-notes, osmand-track, Browser Draw/Maze/Multiply, Files, Clock. |
| **Total** | | **100%** | **~1.0M** | | |

AW-native + AW-targeted (openmobile + 3 NEW buckets) = **45% of steps**; generic grounding/shape donors (aitw + amex + androidcontrol + odyssey) = **55%**, but reshaped to satisfy the action/length floors above.

---

## (6) VERIFICATION CHECKLIST — what `balance_report` MUST confirm

**Action distribution**
- [ ] `open` ≥ 6% of steps AND ≥ 1 per AW-style packed episode (start-on-home discipline).
- [ ] `terminate` ≥ 5% of steps AND ≥ 1 per action-episode.
- [ ] `answer` ≥ 4% of steps AND ≥ 1 per IR-style episode; ≥ 25/116-proportion of AW-aligned episodes end in `answer`.
- [ ] `type` ≥ 15% of steps.
- [ ] `swipe` ≥ 12%, `long_press` ≥ 1%, `click` ≤ 45%.

**App-category floors**
- [ ] productivity ≥ 45%, media ≥ 12%, messaging ≥ 7%, misc ≥ 8%, (browser+files+clock+contacts) ≥ 8% with each ≥ 1.5%, settings-toggle ≤ 8%.
- [ ] All AW third-party apps present with non-zero episodes: markor, broccoli, pro expense, simple calendar pro, simple sms messenger, retro music, vlc, audio recorder, camera, simple draw, simple gallery, osmand, files, clock, contacts, chrome, joplin, tasks, opentracks.

**Episode length**
- [ ] ≥ 30% of steps from episodes with L ≥ 10; ≥ 12% from L ≥ 19.
- [ ] Length-tier multipliers applied (≤4 ×0.5, 10–18 ×2.0, ≥19 ×3.0).
- [ ] No episode with L > 60 admitted without spot-check (androidcontrol max=91, aitw max=112 quarantined).

**Coordinate de-biasing**
- [ ] Per-action click coord-bin (10×10) max cell ≤ 6%.
- [ ] long_press: no single bin > 15%, spread over ≥ 6 bins.
- [ ] swipe direction: up 30–40% / down 30–40% / left 12.5–17.5% / right 12.5–17.5%; vertical:horizontal within 70:30 ±. Each non-odyssey source's dominant swipe direction ≤ 45% of its own swipes.

**Source mix**
- [ ] Final shares within ±2pp of §5 table; aitw ≤ 35% (sanity: down from 95.8%).
- [ ] openmobile ≥ 28%; 3 NEW buckets present and summing ≥ 14%.

**Sanity / leakage**
- [ ] No openmobile episodes whose seeded params overlap eval instances (template-name field is AW family — confirm train/eval split by instance, not just family).
- [ ] coord_mode = guiowl_norm1000_xy uniform across all packed sources (already verified, 0 parse failures).

---
**One-line thesis:** Invert the corpus (AITW 95.8%→18%), make **openmobile the 30% anchor**, **inject `open`/`terminate`/`answer` floors** so episodes have AW's home→act→terminate/answer shape, **up-weight L≥10 episodes to ≥30% of steps**, and **rebalance swipes to 35/35/15/15** — because the measured bottleneck is multi-step chaining and the measured corpus is single-app click-ballast that lacks the three AW-native verbs and the F-Droid apps that 60+ of the 116 tasks require.