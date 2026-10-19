#!/usr/bin/env python3
"""
81_extract_token_trajectories.py -- D1: hidden states at EVERY token, for decoding over time.

01_extract_activations.py keeps three positions per story (belief_end, action_end, final).
That is enough to say "intent is decodable at the end", not enough to say WHEN it becomes
decodable or whether it plateaus. This script keeps every token at a few layers, so
82_temporal_decoding.py can build accuracy-over-time curves and temporal-generalization
matrices (train at time t, test at time t').

Each story is run under one or more PROMPT CONDITIONS:
  story            -- the bare story text (what 01_extract_activations.py sees)
  <template>       -- story + a rating question from 03_behavioral.py (human_verbatim,
                      blame_w1, wrong_w2, punish_w1, ...) or participant_exact (84_*.py)
Having several prompt conditions over identical stories is what lets 82 train the intent
decoder on one wording and test it on another, separating "the model encodes intent" from
"the model encodes this question's phrasing".

--chat {auto,on,off} controls whether the tokenizer's chat template wraps the prompt.
03_behavioral.py applies it whenever the tokenizer ships one, which includes some BASE
models; `auto` reproduces that, `off` gives the raw-text condition for the base-vs-instruct
comparison. The choice is part of the output filename, never implicit.

--variant no_foreshadow deletes the foreshadow sentences (stage_landmarks.py), so harm and
no-harm versions are identical text until the outcome sentence.

Output: <out>/<model_tag>__<variant>__<prompt>__chat<on|off>.{H.npy,npz}
  .H.npy     [total_tokens, n_layers, hidden] float16, stories concatenated (mmap-able)
  tok_ptr    [n_stories+1]   story i owns rows tok_ptr[i]:tok_ptr[i+1]
  char_s/e   [total_tokens]  char span of each token IN THE STORY TEXT (-1 = not story)
  story_id   [n_stories]
  layer_idx  [n_layers]      indices into the hidden-state stack (0 = embeddings)
"""
import os, sys, argparse, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct",
]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_text(row, prompt, beh, exact):
    """-> (prompt string before chat formatting, char offset of the story inside it)."""
    story = row["text"].strip()
    if prompt == "story":
        return story, 0
    if prompt == "participant_exact":
        p, _, _ = exact.build_exact_prompt(row)
    else:
        p, _, _ = beh.build_prompt(row["text"], prompt, row["source"])
    return p, p.index(story)


def select_layers(n_stack, rel):
    return sorted({int(round(r * (n_stack - 1))) for r in rel})


def extract(model_name, rows, prompts, chat, layers_rel, out_dir, dtype, variant,
            revision=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    beh = _load("beh03", "code/03_behavioral.py")
    exact = _load("exact84", "code/experiments/84_participant_exact_prompts.py") \
        if "participant_exact" in prompts else None
    tdt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=tdt,
                                      device_map="auto", trust_remote_code=True).eval()
    use_chat = {"on": True, "off": False, "auto": bool(tok.chat_template)}[chat]
    if use_chat and not tok.chat_template:
        raise SystemExit(f"{model_name}: --chat on but the tokenizer has no chat template")
    tag = model_name.split("/")[-1] + (f"@{revision}" if revision else "")
    os.makedirs(out_dir, exist_ok=True)

    for prompt in prompts:
        stem = os.path.join(out_dir, f"{tag}__{variant}__{prompt}__chat{'on' if use_chat else 'off'}")
        out = stem + ".npz"
        if os.path.exists(out):
            print(f"[skip] {out}", flush=True)
            continue
        H, cs, ce, ptr, keep = [], [], [], [0], None
        with torch.no_grad():
            for r in rows:
                p, story_off = build_text(r, prompt, beh, exact)
                story_len = len(r["text"].strip())
                if use_chat:
                    full = tok.apply_chat_template([{"role": "user", "content": p}],
                                                   tokenize=False, add_generation_prompt=True)
                    story_off += full.index(p)
                else:
                    full = p
                enc = tok(full, return_tensors="pt", return_offsets_mapping=True,
                          add_special_tokens=not use_chat)
                offs = enc.pop("offset_mapping")[0].tolist()
                hs = model(**{k: v.to(model.device) for k, v in enc.items()},
                           output_hidden_states=True).hidden_states
                if keep is None:
                    keep = select_layers(len(hs), layers_rel)
                    print(f"{tag} [{prompt}] layers {keep} of {len(hs)}  chat={use_chat}",
                          flush=True)
                h = torch.stack([hs[L][0] for L in keep], dim=1)          # [T, L, H]
                if not torch.isfinite(h).all():
                    raise SystemExit(f"{tag}: non-finite activations; rerun with --dtype float32")
                H.append(h.to(torch.float16).cpu().numpy())
                # BPE merges across the story boundary (".\n\n"), so a token counts as story
                # if it overlaps the story span; its span is clipped to the story.
                for s, e in offs:
                    s2, e2 = s - story_off, e - story_off
                    inside = e > s and e2 > 0 and s2 < story_len
                    cs.append(max(s2, 0) if inside else -1)
                    ce.append(min(e2, story_len) if inside else -1)
                ptr.append(ptr[-1] + h.shape[0])
        np.save(stem + ".H.npy", np.concatenate(H))
        np.savez(out, tok_ptr=np.array(ptr), char_s=np.array(cs), char_e=np.array(ce),
                 story_id=np.array([r["story_id"] for r in rows]),
                 text=np.array([r["text"].strip() for r in rows]),
                 layer_idx=np.array(keep), model=model_name, prompt=prompt,
                 variant=variant, chat=np.array(use_chat))
        print(f"  -> {out}  ({ptr[-1]} tokens)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompts", nargs="+", default=["story", "human_verbatim", "blame_w1",
                                                      "wrong_w2", "punish_w1"])
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--layers-rel", nargs="+", type=float, default=[0.25, 0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--sources", nargs="+", default=["YS2008", "YS2009", "YS2011"])
    ap.add_argument("--limit", type=int, default=None, help="first N stories (smoke test)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--revision", default=None)
    ap.add_argument("--variant", default="original", choices=["original", "no_foreshadow"])
    # ~1.5 GB per 7B model x prompt; home quota cannot hold the full grid
    ap.add_argument("--out", default=os.environ.get(
        "TOM_TRAJ_DIR", f"/orcd/scratch/orcd/007/{os.environ.get('USER', '')}/acts_traj"))
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import stage_landmarks
    rows = stage_landmarks.load_rows()
    if a.variant == "no_foreshadow":
        rows = stage_landmarks.no_foreshadow(rows)
    rows = [r for r in rows if r["source"] in a.sources]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} stories x {len(a.prompts)} prompts x {len(a.models)} models "
          f"[{a.variant}]", flush=True)
    for m in a.models:
        extract(m, rows, a.prompts, a.chat, a.layers_rel, a.out, a.dtype, a.variant, a.revision)
