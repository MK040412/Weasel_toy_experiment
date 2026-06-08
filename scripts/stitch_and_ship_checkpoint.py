#!/usr/bin/env python3
"""DECOUPLED checkpoint — part 2 (EXTERNAL, fixable WITHOUT restarting training).

WHY THIS EXISTS (read first):
  On a multihost TPU run with ZeRO-1, the model's vocab embedding param ends up dp-SHARDED across all
  16 chips. A naive in-process save (jax.device_get / addressable_shards[0]) on the primary host alone
  CANNOT reconstruct it ("spans non-addressable devices") and a collective process_allgather called on
  the primary alone DEADLOCKS the pod. Three in-process save attempts failed this way, each costing a
  ~14-min recompile restart. The fix is to DECOUPLE the save:
    PART 1 (in-process, in train_fastdvlm_tpu.py: dump_local_shards): every host dumps ONLY its own local
            addressable shards (pure local reads, NO collective) to out_dir (= /dev/shm, RAM, because the
            worker root disk is ~97% full). This is deliberately DUMB so it can never deadlock / never fail
            the run. Files: shards-step{STEP:06d}.proc{P}.pkl  (one per host, P = jax.process_index()).
    PART 2 (THIS script, EXTERNAL): collect all per-host shard pkls, reassemble the full weights, write HF
            safetensors, ship to HF/Vultr. If THIS has a bug, fix it and re-run on the already-dumped
            shards while training keeps running — NO restart, NO recompile, NO from-scratch.

Each pkl is a list of (dotpath:str, global_shape:tuple, [(start,stop) per dim], shard_numpy). Reassembly:
  full = np.empty(global_shape); for every shard across every proc: full[slices] = shard_numpy.
(Verified: a replicated leaf yields shards that each cover the full slice; a dp-sharded leaf yields
disjoint row-blocks; placing each at its global index reconstructs the original bit-identically.)

USAGE (run on a box that can reach HF, after collecting the per-host pkls into one dir):
  python stitch_and_ship_checkpoint.py \
    --shards-dir DIR_WITH_proc0..3_pkls --step 843 \
    --source-model-dir ~/models/boltzmann-final \
    --out-dir /tmp/ckpt-step000843 \
    [--hf-repo KMK040412/fastdvlm-aw-guiowlvit] [--hf-path-prefix fast-dvlm-kd-tpu]
Only deps: numpy, safetensors, huggingface_hub (NO jax/torch needed). Pure, restart-free.
"""
import argparse
import glob
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import numpy as np


def _hf_name_and_transform(dotpath: str):
    """Map an nnx state dot-path -> (hf_tensor_name, transform, also_lm_head). Rules mirror
    export_hf_safetensors in train_fastdvlm_tpu.py exactly."""
    if dotpath.endswith(".kernel"):
        base = dotpath[: -len(".kernel")] + ".weight"
        tf = "conv3d" if "patch_embed.proj" in dotpath else "linear"
        return base, tf, False
    if dotpath.endswith(".scale"):  # RMSNorm scale -> weight (norm1/norm2/merger/deepstack norms)
        return dotpath[: -len(".scale")] + ".weight", "none", False
    if dotpath.endswith(".embedding"):
        if "embed_tokens" in dotpath:
            return "model.language_model.embed_tokens.weight", "none", True  # tied -> also lm_head.weight
        return dotpath[: -len(".embedding")] + ".weight", "none", False      # pos_embed.embedding
    return dotpath, "none", False  # .bias / .weight (q_norm/k_norm/layernorms) pass through unchanged


def _apply_transform(arr: np.ndarray, tf: str) -> np.ndarray:
    if tf == "linear":
        arr = arr.transpose(1, 0)
    elif tf == "conv3d":
        arr = arr.transpose(4, 3, 0, 1, 2)
    return np.ascontiguousarray(arr)


def reassemble(shards_dir: Path, step: int) -> dict:
    """Collect every per-host pkl for this step, reassemble each param to its full global array."""
    pkls = sorted(glob.glob(str(shards_dir / f"shards-step{step:06d}.proc*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"no shard pkls for step {step} in {shards_dir}")
    print(f"[stitch] {len(pkls)} per-host shard files: {[Path(p).name for p in pkls]}")
    full: dict[str, np.ndarray] = {}
    filled: dict[str, np.ndarray] = {}  # bool mask of coverage (sanity)
    for p in pkls:
        with open(p, "rb") as f:
            payload = pickle.load(f)
        for entry in payload:
            dotpath, gshape, idx, data = entry
            # dotpath may be a str (clean) or a tuple of key-reprs; normalize either way.
            if isinstance(dotpath, (tuple, list)):
                parts = []
                for k in dotpath:
                    m = re.search(r"idx=(\d+)", str(k))
                    parts.append(m.group(1) if m else str(getattr(k, "key", k)))
                dotpath = ".".join(parts)
            gshape = tuple(int(d) for d in gshape)
            if dotpath not in full:
                full[dotpath] = np.empty(gshape, dtype=np.asarray(data).dtype)
                filled[dotpath] = np.zeros(gshape[0], dtype=bool) if gshape else None
            sl = tuple(slice(a, b) for (a, b) in idx)
            full[dotpath][sl] = data
            if filled[dotpath] is not None and idx:
                filled[dotpath][idx[0][0]:idx[0][1]] = True
    # coverage sanity: every row of axis-0 must be filled
    for k, m in filled.items():
        if m is not None and not m.all():
            raise RuntimeError(f"[stitch] '{k}' has UNFILLED rows {int((~m).sum())}/{m.size} — missing shards")
    print(f"[stitch] reassembled {len(full)} params; total params bytes={sum(a.nbytes for a in full.values())/1e9:.2f}GB")
    return full


def to_hf_tensors(full: dict) -> dict:
    tensors: dict[str, np.ndarray] = {}
    for dotpath, arr in full.items():
        name, tf, also_lm = _hf_name_and_transform(dotpath)
        tensors[name] = _apply_transform(arr, tf)
        if also_lm:
            tensors["lm_head.weight"] = tensors[name]
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--source-model-dir", required=True, help="for config/tokenizer files to copy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--hf-path-prefix", default="fast-dvlm-kd-tpu")
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = reassemble(Path(args.shards_dir), args.step)
    tensors = to_hf_tensors(full)

    # copy the non-weight bundle files from the source checkpoint (config, tokenizer, processor, ...)
    src = Path(args.source_model_dir).expanduser()
    for name in ["config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                 "video_preprocessor_config.json", "vit_lora_merged_config.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    from safetensors.numpy import save_file
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "jax_export_step": str(args.step)})
    (out / "checkpoint_manifest.json").write_text(json.dumps(
        {"step": args.step, "format": "hf_safetensors", "n_tensors": len(tensors),
         "source_model_dir": str(src), "reassembled_from": "decoupled per-host shard dump"}, indent=2))
    print(f"[stitch] wrote {out/'model.safetensors'} ({len(tensors)} tensors)")

    if args.hf_repo:
        from huggingface_hub import HfApi
        token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(repo_id=args.hf_repo, exist_ok=True)
        path_in_repo = f"{args.hf_path_prefix.rstrip('/')}/checkpoint-step{args.step:06d}" if args.hf_path_prefix \
            else f"checkpoint-step{args.step:06d}"
        api.upload_folder(repo_id=args.hf_repo, folder_path=str(out), path_in_repo=path_in_repo, token=token)
        print(f"[ship] uploaded -> {args.hf_repo}/{path_in_repo}")


if __name__ == "__main__":
    main()
