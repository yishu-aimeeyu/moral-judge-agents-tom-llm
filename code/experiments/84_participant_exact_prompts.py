#!/usr/bin/env python3
"""
84_participant_exact_prompts.py -- what did the PARTICIPANTS actually see?

`human_verbatim` in 03_behavioral.py is a per-source generic question, not the question each
participant answered. The published stimulus files (dataset/raw_text/) show three differences:

  YS2008  participants: item-specific stem, e.g. "Putting the substance in was:"
          scale 1 = "Forbidden" ... 3 = "Permissible"
          human_verbatim: "How permissible was {agent}'s action? 1 (completely permissible)
          to 3 (completely impermissible)" -- different wording AND reversed anchors.
  YS2009  participants: item-specific, e.g. "How much blame does Matt deserve for just
          watching as his cousin went into the kitchen?"  1 = "None" ... 4 = "A Lot"
          human_verbatim: "How much blame does {agent} deserve? 1 (no blame) to 4 (full blame)"
  YS2011  participants: "How morally wrong was the action?"  1 = "not at all" ... 7 = "very"
          human_verbatim: "How morally wrong was {agent}'s action?" -- and the stories are
          second person ("Imagine you ..."), so extract_agent() returns "Imagine" and the
          model is asked about "Imagine's action".

This module builds the participant-exact prompt for every master row (imported by
81_extract_token_trajectories.py), and as a script it either audits the differences or
scores a model with them, writing outputs/behavior_participant_exact/raw_<model>.csv in the
usual raw format under the template name `participant_exact`.

  python code/experiments/84_participant_exact_prompts.py --audit
  python code/experiments/84_participant_exact_prompts.py --run --models Qwen/Qwen2.5-7B
"""
import os, re, csv, argparse, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
RAW = os.path.join(ROOT, "dataset", "raw_text")
ANSWER = "Respond with a single integer."

# (s_min, s_max, low anchor, high anchor, direction). direction="permissible" means a HIGH
# number is LESS blame, so normalized blame is inverted before it is stored.
SCALES = {
    "YS2008": (1, 3, "Forbidden", "Permissible", "permissible"),
    "YS2009": (1, 4, "None", "A Lot", "blame"),
    "YS2011": (1, 7, "not at all", "very", "blame"),
}


def _ys2008_stems():
    """scenario_id -> stem. The stem is the first line ending in 'was:' after the header."""
    lines = [l.strip() for l in open(os.path.join(RAW, "YS2008.txt"), encoding="utf-8")]
    ids = {r["scenario_id"] for r in csv.DictReader(open(MASTER, encoding="utf-8"))
           if r["source"] == "YS2008"}
    out, cur = {}, None
    for l in lines:
        if l in ids:
            cur = l
        elif cur and l.endswith("was:") and cur not in out:
            out[cur] = l
    return out


def _ys2009_questions():
    """agent first name -> item question, matched by the name the question asks about."""
    out = {}
    for l in open(os.path.join(RAW, "YS2009.txt"), encoding="utf-8"):
        m = re.match(r"\s*How much blame does ([A-Z][a-z]+) deserve .*\?\s*$", l)
        if m:
            out[m.group(1)] = l.strip()
    return out


_CACHE = {}


def build_exact_prompt(row):
    """-> (prompt, s_min, s_max). Raises if an item's participant question cannot be found."""
    if not _CACHE:
        _CACHE["08"], _CACHE["09"] = _ys2008_stems(), _ys2009_questions()
    src, story = row["source"], row["text"].strip()
    s_min, s_max, lo, hi, _ = SCALES[src]
    if src == "YS2008":
        q = _CACHE["08"].get(row["scenario_id"])
    elif src == "YS2009":
        agent = re.search(r"\b([A-Z][a-z]{2,})\b", story).group(1)
        q = _CACHE["09"].get(agent)
    else:
        q = "How morally wrong was the action?"
    if not q:
        raise KeyError(f"no participant question for {row['story_id']}")
    scale = f"Rate from {s_min} (\u201c{lo}\u201d) to {s_max} (\u201c{hi}\u201d). {ANSWER}"
    return f"{story}\n\n{q}\n{scale}", s_min, s_max


def norm_blame(raw, source):
    s_min, s_max, _, _, direction = SCALES[source]
    n = (raw - s_min) / (s_max - s_min)
    return 1 - n if direction == "permissible" else n


def audit(rows):
    beh = _load_beh()
    out = os.path.join(ROOT, "outputs", "analysis", "participant_exact_prompts.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_agent_bug = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["story_id", "source", "participant_question", "human_verbatim_question"])
        for r in rows:
            p, _, _ = build_exact_prompt(r)
            v, _, _ = beh.build_prompt(r["text"], "human_verbatim", r["source"])
            story = r["text"].strip()
            vq = v[len(story):].strip()
            n_agent_bug += "Imagine's" in vq
            w.writerow([r["story_id"], r["source"], p[len(story):].strip(), vq])
    print(f"{len(rows)} prompts audited -> {out}")
    print(f"human_verbatim prompts asking about \"Imagine's action\": {n_agent_bug}")


def _load_beh():
    spec = importlib.util.spec_from_file_location("beh03", os.path.join(ROOT, "code", "03_behavioral.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(models, rows, scoring):
    os.chdir(ROOT)                       # 03_behavioral resolves its paths relative to the repo
    beh = _load_beh()
    for name in models:
        backend = beh.HFBackend(name, scoring)
        # own folder: existing scripts glob outputs/behavior/raw_*.csv as one file per model
        out_dir = os.path.join(ROOT, "outputs", "behavior_participant_exact")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"raw_{beh.model_safe(name)}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "template", "sample", "story_id", "source", "condition",
                        "intent_label", "outcome_label", "raw_rating", "norm_rating"])
            for r in rows:
                p, s_min, s_max = build_exact_prompt(r)
                ratings, _ = backend.rate(p, s_min, s_max)
                for i, x in enumerate(ratings):
                    w.writerow([name, "participant_exact", i, r["story_id"], r["source"],
                                r["condition"], r["intent_label"], r["outcome_label"],
                                x, round(norm_blame(x, r["source"]), 4)])
        print(f"{name}: -> {out}", flush=True)
        del backend


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--scoring", default="logprob", choices=["logprob", "sampling"])
    a = ap.parse_args()
    rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    if a.audit or not a.run:
        audit(rows)
    if a.run:
        run(a.models, rows, a.scoring)
