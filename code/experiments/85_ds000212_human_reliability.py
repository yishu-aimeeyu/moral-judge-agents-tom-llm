#!/usr/bin/env python3
"""
85_ds000212_human_reliability.py -- R2-human: split-half reliability of REAL participants.

No per-participant ratings exist publicly for the YS2008 / YS2009 stories (the papers report
cell means only, n = 10-17). The closest open data is OpenNeuro ds000212 (Young, Chakroff,
Wasserman, Saxe, Dungan, Brown, Koster-Hale; CC0): 39 adults (NT and ASD), 60 scenarios,
intentional vs accidental, rated 1-4 in the scanner -- the Young & Saxe (2011) design that
the master's YS2011 items come from. Its events files carry item number + key press per
trial, but not item text, so items cannot yet be joined to master story_ids.

What this gives: the human reliability that model reliability (83, R2) should be read against.
  * split-half (Spearman-Brown) of item means, per domain, raw and within-condition
  * per-item SD across participants (human "story consistency", R3)
  * intentional - accidental contrast per domain, NT vs ASD

  python code/experiments/85_ds000212_human_reliability.py --fetch   # ~240 small tsv files
  python code/experiments/85_ds000212_human_reliability.py
"""
import os, io, argparse, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "dataset", "human_reference", "ds000212")
OUT = os.path.join(ROOT, "outputs", "analysis", "reliability")
S3 = "https://s3.amazonaws.com/openneuro.org/ds000212"
DOMAIN = {"PH": "physical_harm", "PS": "psychological_harm", "I": "incest",
          "P": "pathogen", "N": "neutral"}


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    parts = pd.read_csv(io.StringIO(_get(f"{S3}/participants.tsv")), sep="\t")
    parts.to_csv(os.path.join(RAW, "participants.tsv"), sep="\t", index=False)
    rows = []
    for sub in parts["participant_id"]:
        for run in range(1, 7):
            try:
                ev = pd.read_csv(io.StringIO(_get(
                    f"{S3}/{sub}/func/{sub}_task-dis_run-{run:02d}_events.tsv")), sep="\t")
            except Exception:
                continue
            ev["subject"], ev["run"] = sub, run
            rows.append(ev)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(os.path.join(RAW, "dis_events_all.csv"), index=False)
    print(f"{d['subject'].nunique()} subjects, {len(d)} trials -> {RAW}/dis_events_all.csv")


def tidy():
    d = pd.read_csv(os.path.join(RAW, "dis_events_all.csv"))
    g = pd.read_csv(os.path.join(RAW, "participants.tsv"), sep="\t").set_index("participant_id")["group"]
    code = d["condition"].str.split("_").str[1]                  # PHA, PSI, IA, ...
    d["intent"] = np.where(code.str.endswith("I"), "intentional", "accidental")
    d["domain"] = code.str[:-1].map(DOMAIN)
    d["group"] = d["subject"].map(g)
    d["rating"] = pd.to_numeric(d["key"], errors="coerce")
    d = d[d["rating"].between(1, 4)]
    # one rating per subject x item x intent (the design shows each item in one condition)
    return d


def split_half(W, n, rng):
    k = W.shape[1]
    rs = []
    for _ in range(n):
        p = rng.permutation(k)
        a, b = np.nanmean(W[:, p[:k // 2]], 1), np.nanmean(W[:, p[k // 2:]], 1)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 3:
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            rs.append(2 * r / (1 + r))
    rs = np.array(rs)
    return rs.mean(), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def analyse(d, n_splits):
    rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    rows, cons = [], []
    for grp in ("NT", "ASD", "all"):
        dg = d if grp == "all" else d[d["group"] == grp]
        for dom in ["all"] + sorted(d["domain"].dropna().unique()):
            dd = dg if dom == "all" else dg[dg["domain"] == dom]
            dd = dd.assign(cell=dd["item"].astype(str) + "_" + dd["intent"])
            W = dd.pivot_table(index="cell", columns="subject", values="rating")
            cond = W.index.str.split("_").str[1]
            for mode in ("raw", "within_condition"):
                M = W if mode == "raw" else W.sub(W.groupby(cond).transform("mean"))
                m, lo, hi = split_half(M.to_numpy(float), n_splits, rng)
                rows.append(dict(group=grp, domain=dom, mode=mode, n_cells=len(W),
                                 n_subjects=W.shape[1], split_half_sb=round(m, 3),
                                 lo=round(lo, 3), hi=round(hi, 3)))
            mi = dd.groupby("intent")["rating"].mean()
            rows[-1]["intent_minus_accident_1to4"] = round(mi.get("intentional", np.nan)
                                                           - mi.get("accidental", np.nan), 3)
    for (item, intent), g in d[d["group"] == "NT"].groupby(["item", "intent"]):
        cons.append(dict(item=item, intent=intent, domain=g["domain"].iloc[0], n=len(g),
                         mean=round(g["rating"].mean(), 3), sd=round(g["rating"].std(), 3)))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "R2_human_ds000212_split_half.csv"), index=False)
    c = pd.DataFrame(cons).sort_values("sd", ascending=False)
    c.to_csv(os.path.join(OUT, "R3_human_ds000212_item_consistency.csv"), index=False)
    print(r.to_string(index=False))
    return r, c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--splits", type=int, default=1000)
    a = ap.parse_args()
    if a.fetch or not os.path.exists(os.path.join(RAW, "dis_events_all.csv")):
        fetch()
    analyse(tidy(), a.splits)
