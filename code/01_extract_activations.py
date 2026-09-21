#!/usr/bin/env python3
"""
01_extract_activations.py  --  NO TRAINING. Forward passes only.

For each open-weight model, run every stimulus through a single forward pass and
save the hidden state at every layer. We store two pooling variants:
  - last-token hidden state  (standard for decoder-only probing)
  - mean-pooled over tokens  (robustness comparison)

IMPORTANT: representational analysis (Levels 2-3) requires OPEN-WEIGHT models,
because closed APIs (GPT, Claude, Gemini) do not expose hidden states. Use the
HuggingFace models below. Closed APIs can still be used for the behavioral level
(see 03_behavioral.py).

Output: outputs/acts/<model_tag>.npz  with arrays
  last  : [n_stories, n_layers, hidden]
  mean  : [n_stories, n_layers, hidden]
  story_id : [n_stories]  (aligned index into the master CSV)

`--revision` extends this to intermediate pretraining checkpoints (E8): the revision is part
of the filename (`pythia-1.4b@step1000.npz`) because activations from two checkpoints of the
same repo are different data, not different runs of the same data. With `--layers-rel` only
the requested relative depths are kept, and `layer_idx` records which ones, so a subset can
never be misread as layers 0..k. With `--evict-cache` each revision's weights are deleted
after extraction, which is what makes a 26-checkpoint sweep fit on disk at all.

Run on a GPU box (lab cluster / Colab). This script does not run in the chat sandbox.
"""
import os, csv, argparse, numpy as np

# Suggested model ladder (scale axis within & across families). Edit freely.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B",   "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B",
    # add instruct variants to test tuning effect, e.g. "...-Instruct"
]

def load_stimuli(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    return rows

def load_clause_offsets(path):
    """story_id -> {belief_end, action_end, ...} character offsets, or {} if unavailable."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["story_id"]] = {k: int(r[k]) for k in
                              ("belief_start", "belief_end", "action_start",
                               "action_end", "outcome_start", "outcome_end")}
        out[r["story_id"]]["method"] = r["method"]
    return out


def _token_at_char(offsets, char_end):
    """
    Index of the LAST token that ends at or before char_end.

    Character offsets cannot be precomputed as token indices because every tokenizer
    segments differently, so the mapping is resolved here, per model, from
    return_offsets_mapping. Special tokens carry (0, 0) and are skipped.
    """
    best = None
    for t, (s, e) in enumerate(offsets):
        if e == 0 and s == 0:
            continue
        if e <= char_end:
            best = t
        else:
            break
    return best


def model_tag(model_name, revision=None):
    """Filename stem. A revision is part of the identity of the activations, not metadata."""
    tag = model_name.split("/")[-1]
    return f"{tag}@{revision}" if revision else tag


def evict_from_cache(model_name, revision):
    """Delete this revision's snapshot from the HF cache.

    E8 walks ~26 Pythia revisions at 2.93 GB each, and the OLMo extension walks ~20 at 14 GB.
    Retaining them would be ~76 GB and ~280 GB respectively, against 124 GB of home quota, so
    eviction is load-bearing rather than tidy-minded: without it the trajectory cannot finish.
    Failure to evict is reported and does not abort a run whose activations are already saved.
    """
    try:
        from huggingface_hub import scan_cache_dir
        repo_id = model_name
        cache = scan_cache_dir()
        hashes = [rev.commit_hash
                  for repo in cache.repos if repo.repo_id == repo_id
                  for rev in repo.revisions
                  if revision is None or revision in rev.refs or rev.commit_hash == revision]
        if not hashes:
            print(f"  [evict] nothing cached for {repo_id}@{revision}", flush=True)
            return
        op = cache.delete_revisions(*hashes)
        freed = op.expected_freed_size_str
        op.execute()
        print(f"  [evict] {repo_id}@{revision}: freed {freed}", flush=True)
    except Exception as e:
        print(f"  [evict] WARNING could not evict {model_name}@{revision}: {e}", flush=True)


def select_layers(n_stack, rel_depths):
    """Indices into the hidden-state stack at the requested relative depths.

    Index 0 is the embedding output and index n_stack-1 the final layer, the same convention
    `24_rsa_cka.py` uses when it maps REL_DEPTHS onto the cached activation array, so a layer
    subset extracted here lines up with a full extraction read by the existing RSA code.
    """
    return sorted({int(round(r * (n_stack - 1))) for r in rel_depths})


def extract_for_model(model_name, rows, out_dir, batch_size=8, max_len=512,
                      clause_offsets=None, merge_existing=False, revision=None,
                      layers_rel=None, dtype="float16", evict=False):
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_name, revision=revision, output_hidden_states=True,
        torch_dtype=torch_dtype, device_map="auto",
        trust_remote_code=True)
    model.eval()
    keep = None          # resolved on the first batch, once the stack depth is known

    want_clause = bool(clause_offsets)
    texts = [r["text"] for r in rows]
    last_all, mean_all = [], []
    belief_all, action_all = [], []
    n_missing = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            brows = rows[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len,
                      return_offsets_mapping=want_clause)
            offs = enc.pop("offset_mapping").tolist() if want_clause else None
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc)
            hs = torch.stack(out.hidden_states, dim=1)  # [B, L, T, H]
            if layers_rel:
                if keep is None:
                    keep = select_layers(hs.size(1), layers_rel)
                    print(f"  layers: keeping {keep} of {hs.size(1)} "
                          f"(rel depths {layers_rel})", flush=True)
                hs = hs[:, keep]
            mask = enc["attention_mask"].unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
            # last non-pad token index per sequence
            lengths = enc["attention_mask"].sum(1) - 1
            idx = lengths.view(-1,1,1,1).expand(-1, hs.size(1), 1, hs.size(-1))
            last = hs.gather(2, idx).squeeze(2)                       # [B,L,H]
            mean = (hs*mask).sum(2) / mask.sum(2).clamp(min=1)        # [B,L,H]
            last_all.append(last.float().cpu().numpy())
            mean_all.append(mean.float().cpu().numpy())

            if want_clause:
                for key, sink in (("belief_end", belief_all), ("action_end", action_all)):
                    pos = []
                    for b, r in enumerate(brows):
                        co = clause_offsets.get(r["story_id"])
                        t = _token_at_char(offs[b], co[key]) if co else None
                        if t is None:
                            t = int(lengths[b].item())   # fall back to the final token
                            n_missing += 1
                        pos.append(t)
                    p = torch.tensor(pos, device=hs.device)
                    gi = p.view(-1,1,1,1).expand(-1, hs.size(1), 1, hs.size(-1))
                    sink.append(hs.gather(2, gi).squeeze(2).float().cpu().numpy())

    last_all = np.concatenate(last_all); mean_all = np.concatenate(mean_all)
    tag = model_tag(model_name, revision)
    os.makedirs(out_dir, exist_ok=True)
    arrays = dict(last=last_all, mean=mean_all,
                  story_id=np.array([r["story_id"] for r in rows]))
    if want_clause:
        arrays["belief_last"] = np.concatenate(belief_all)
        arrays["action_last"] = np.concatenate(action_all)
    if keep is not None:
        # which rows of the layer axis these are, so a subset is never mistaken for layers 0..k
        arrays["layer_idx"] = np.array(keep)
    if revision:
        arrays["revision"] = np.array(revision)

    # A checkpoint at random initialisation is the one case where the forward pass can
    # overflow fp16 silently, and E8's step-0 gate is built on exactly that checkpoint. A
    # non-finite activation would reach the RDM as a garbage distance rather than as an error.
    n_bad = sum(int((~np.isfinite(v)).sum()) for k, v in arrays.items()
                if k not in ("story_id", "layer_idx", "revision"))
    if n_bad:
        raise SystemExit(f"{tag}: {n_bad} non-finite activation values at dtype={dtype}. "
                         f"Re-run this revision with --dtype float32 before using it.")

    out_path = os.path.join(out_dir, f"{tag}.npz")
    if merge_existing and os.path.exists(out_path):
        old = np.load(out_path, allow_pickle=True)
        old_ids = [str(s) for s in old["story_id"]]
        new_ids = [r["story_id"] for r in rows]
        id_to_new = {sid: i for i, sid in enumerate(new_ids)}
        merged = {}
        for key in old.files:
            if key == "story_id":
                merged[key] = old[key]
                continue
            arr = np.array(old[key], copy=True)
            for sid, j in id_to_new.items():
                if sid not in old_ids:
                    raise SystemExit(f"{tag}: {sid} not in existing npz — cannot merge")
                arr[old_ids.index(sid)] = arrays[key][j]
            merged[key] = arr
        np.savez_compressed(out_path, **merged)
        print(f"{tag}: MERGED {len(new_ids)} stories into {out_path}", flush=True)
    else:
        np.savez_compressed(out_path, **arrays)
        extra = f" +belief_last/action_last (fallbacks: {n_missing})" if want_clause else ""
        print(f"{tag}: saved {last_all.shape} (n,layers,hidden){extra}", flush=True)

    if evict:
        del model
        try:
            import gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        evict_from_cache(model_name, revision)

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(here, "..", "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "acts"))
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--clause-offsets", default=None,
                    help="dataset/master/clause_offsets.csv -- additionally save "
                         "belief_last and action_last pooling, so intent can be probed at "
                         "the belief clause, BEFORE the harm is mentioned in the text")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan only (no weight download / no GPU)")
    ap.add_argument("--story-ids", nargs="*", default=None,
                    help="restrict extraction to these story_ids (surgical re-extract)")
    ap.add_argument("--merge-existing", action="store_true",
                    help="with --story-ids, patch rows into the existing npz")
    # --- checkpoint trajectory (E8). Default None keeps every existing call byte-identical.
    ap.add_argument("--revision", default=None,
                    help="HF revision/branch, e.g. step0 / step143000 for Pythia, or an "
                         "OLMo-2 stage tag. Output is written as <model>@<revision>.npz")
    ap.add_argument("--layers-rel", nargs="*", type=float, default=None,
                    help="keep only these relative depths (0.0=embeddings, 1.0=final), e.g. "
                         "0 0.2 0.4 0.6 0.8 1.0. Cuts the activation file ~4x for E8.")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--evict-cache", action="store_true",
                    help="delete this revision's HF snapshot after extracting, so a long "
                         "checkpoint sweep does not accumulate every revision on disk")
    a = ap.parse_args()
    rows = load_stimuli(a.csv)
    if a.story_ids:
        want = set(a.story_ids)
        rows = [r for r in rows if r["story_id"] in want]
        missing = want - {r["story_id"] for r in rows}
        if missing:
            raise SystemExit(f"story-ids not in master: {sorted(missing)}")
        print(f"restricted to {len(rows)} stories: {sorted(want)}")
    clause = load_clause_offsets(a.clause_offsets)
    if a.clause_offsets:
        print(f"clause offsets: {len(clause)} stories from {a.clause_offsets}")
    if a.dry_run:
        print(f"=== ACTIVATION EXTRACTION PLAN (dry-run) ===")
        print(f"stimuli={len(rows)}  out={a.out}")
        if a.revision:
            print(f"revision={a.revision}  ->  {model_tag(a.models[0], a.revision)}.npz")
        if a.layers_rel:
            print(f"layers-rel={a.layers_rel}  dtype={a.dtype}  evict={bool(a.evict_cache)}")
        print(f"{'model':42} {'~VRAM(bf16)':>12}")
        for m in a.models:
            # rough: params from name * 2 GB + overhead
            import re
            mm = re.search(r"(\d+\.?\d*)\s*[bB]\b", m)
            gb = (float(mm.group(1)) * 2 + 2) if mm else float("nan")
            print(f"{m:42} {gb:10.0f}GB" if mm else f"{m:42} {'?':>12}")
        print(f"\nLaunch:\n  JOBNAME=acts PART=mit_preemptable bash engaging/submit_gpu.sh "
              f"\"python code/01_extract_activations.py --models {' '.join(a.models)}\"")
    else:
        for m in a.models:
            extract_for_model(m, rows, a.out, clause_offsets=clause,
                              merge_existing=bool(a.merge_existing),
                              revision=a.revision, layers_rel=a.layers_rel,
                              dtype=a.dtype, evict=bool(a.evict_cache))
