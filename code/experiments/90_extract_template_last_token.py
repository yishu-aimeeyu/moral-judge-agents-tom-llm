#!/usr/bin/env python3
"""
90_extract_template_last_token.py -- Lax 9/29 T5 input: final-token hidden state at EVERY layer,
for every story under every FACTORIAL_TEMPLATES prompt. Inference only.

81 keeps every token at 5 depths (~90 GB per model if extended to every layer); T5 needs only the
last prompt token -- the position the rating is read from -- so this keeps [n_stories, n_layers, H].
Prompt text and chat formatting are 81.build_text + the same chat rule as 03_behavioral (--chat auto).

Output: <out>/<model_tag>__<template>__chat<on|off>.npz
  last [n, n_layers, H] float16, story_id [n], layer_idx, n_tokens [n]
~60 MB per 7B model x template. Default out: $TOM_TPL_DIR or /orcd/scratch/orcd/007/$USER/acts_templates

  python code/experiments/90_extract_template_last_token.py --models Qwen/Qwen2.5-7B --limit 4   # smoke
"""
import os, sys, argparse, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_MODELS = ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
                  "unsloth/Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def extract(model_name, rows, prompts, chat, out_dir, dtype):
    import torch
    from transformers import AutoTokenizer, AutoModel
    beh = _load("beh03", "code/03_behavioral.py")
    x81 = _load("x81", "code/experiments/81_extract_token_trajectories.py")
    tdt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=tdt, device_map="auto",
                                      trust_remote_code=True).eval()
    use_chat = {"on": True, "off": False, "auto": bool(tok.chat_template)}[chat]
    if use_chat and not tok.chat_template:
        raise SystemExit(f"{model_name}: --chat on but the tokenizer has no chat template")
    tag = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    for prompt in prompts:
        out = os.path.join(out_dir, f"{tag}__{prompt}__chat{'on' if use_chat else 'off'}.npz")
        if os.path.exists(out):
            print(f"[skip] {out}", flush=True)
            continue
        H, ntok = [], []
        with torch.no_grad():
            for r in rows:
                p, _ = x81.build_text(r, prompt, beh, None)
                full = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True) if use_chat else p
                enc = tok(full, return_tensors="pt", add_special_tokens=not use_chat)
                hs = model(**{k: v.to(model.device) for k, v in enc.items()},
                           output_hidden_states=True).hidden_states
                h = torch.stack([L[0, -1] for L in hs])                       # [n_layers, H]
                if not torch.isfinite(h).all():
                    raise SystemExit(f"{tag}: non-finite activations; rerun with --dtype float32")
                H.append(h.to(torch.float16).cpu().numpy())
                ntok.append(enc["input_ids"].shape[1])
        np.savez(out, last=np.stack(H), story_id=np.array([r["story_id"] for r in rows]),
                 layer_idx=np.arange(len(H[0])), n_tokens=np.array(ntok), model=model_name,
                 prompt=prompt, chat=np.array(use_chat))
        print(f"  -> {out}  {np.stack(H).shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompts", nargs="+", default=None, help="default: 03 FACTORIAL_TEMPLATES")
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default=os.environ.get(
        "TOM_TPL_DIR", f"/orcd/scratch/orcd/007/{os.environ.get('USER', '')}/acts_templates"))
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import stage_landmarks
    rows = stage_landmarks.load_rows()
    if a.limit:
        rows = rows[:a.limit]
    prompts = a.prompts or _load("beh03", "code/03_behavioral.py").FACTORIAL_TEMPLATES
    print(f"{len(rows)} stories x {len(prompts)} prompts x {len(a.models)} models, chat={a.chat}", flush=True)
    for m in a.models:
        extract(m, rows, prompts, a.chat, a.out, a.dtype)
