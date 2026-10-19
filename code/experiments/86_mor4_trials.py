#!/usr/bin/env python3
"""
86_mor4_trials.py -- T1.5: tidy trial table for the MOR4 fMRI behaviour (Young & Saxe 2008 design).

Source: /orcd/data/saxe/001/mindhive/saxelab2/MOR4_2/behavioural/SAX_MOR4_XX.MOR4.<run>.mat
  theStories (8 story numbers, 1-48), theOrder (condition code 1-8), key, RT.

Condition codes: SAX_MOR4_01/02.SPMinputs.txt list, per session, the 8 condition names in code order
with their onsets. Onsets map to trial positions (8, 30, ..., 162 s); this script re-derives theOrder from
the onsets for every session in those two files and asserts it equals the run file's theOrder, so
code k == k-th SPM name is checked, not assumed.
  BT/BG = belief level, OT/OG = outcome level, BF/OF = belief-first / foreshadow-first order.
  Which of T/G is the negative level is NOT stated in any file found; it is not decoded here.

Story numbers -> master story_id: no stimulus file or experiment script was found under MOR4_2/,
MOR4_3/ or saxelab2/ (see REPORT). story_id is left empty.

Keys: subjects used 50/51/52 (49 appears twice); subject 02 used 85/86/87. rating_1to4 = key - 48
(work-order convention); 85/86/87 are mapped to 2/3/4 and flagged. key == 0 -> no response.

  python code/experiments/86_mor4_trials.py
"""
import os, re, glob
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = "/orcd/data/saxe/001/mindhive/saxelab2/MOR4_2/behavioural"
OUT = os.path.join(ROOT, "outputs", "lax_0929")
ONSETS = [8, 30, 52, 74, 96, 118, 140, 162]
COND = ["BT_OT_BF", "BT_OG_BF", "BG_OT_BF", "BG_OG_BF",
        "BT_OT_OF", "BT_OG_OF", "BG_OT_OF", "BG_OG_OF"]
ALT_KEYS = {85: 50, 86: 51, 87: 52}


def spm_sessions(path):
    txt = open(path, errors="replace").read()
    pairs = re.findall(r"(B[TG]_O[TG]_[BO]F)\s+(\d+)", txt)
    sess = [pairs[i:i + 8] for i in range(0, len(pairs), 8)]
    out = []
    for s in sess:
        assert [n for n, _ in s] == COND, s
        pos = {int(o): COND.index(n) + 1 for n, o in s}
        out.append([pos[o] for o in ONSETS])
    return out


def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "SAX_MOR4_*.MOR4.[0-9].mat"))):
        m = re.search(r"SAX_MOR4_(\d+)\.MOR4\.(\d)\.mat$", f)
        subj, run = m.group(1), int(m.group(2))
        d = sio.loadmat(f, squeeze_me=True)
        for t, (s, c, k, rt) in enumerate(zip(d["theStories"], d["theOrder"], d["key"], d["RT"]), 1):
            k = int(k)
            kk = ALT_KEYS.get(k, k)
            rows.append(dict(subject=subj, run=run, trial=t, story_num=int(s), story_id="",
                             condition=int(c), condition_name=COND[int(c) - 1],
                             belief_level=COND[int(c) - 1][1], outcome_level=COND[int(c) - 1][4],
                             order=COND[int(c) - 1][6:],
                             rating_raw=k, rating_1to4=(kk - 48) if k else np.nan,
                             rt=float(rt) if k else np.nan,
                             alt_keycodes=k in ALT_KEYS))
    return pd.DataFrame(rows)


def verify_conditions(df):
    out = []
    for sub in ["01", "02"]:
        sess = spm_sessions(os.path.join(SRC, f"SAX_MOR4_{sub}.SPMinputs.txt"))
        runs = sorted(df.loc[df.subject == sub, "run"].unique())
        assert len(sess) == len(runs), (sub, len(sess), runs)
        for r, order in zip(runs, sess):
            got = df[(df.subject == sub) & (df.run == r)].sort_values("trial").condition.tolist()
            out.append(dict(subject=sub, run=r, spm_order=order, mat_order=got, match=order == got))
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_runs()
    ver = verify_conditions(df)
    print(ver.to_string(index=False))
    assert ver.match.all(), "SPMinputs onsets disagree with theOrder"
    df.to_csv(os.path.join(OUT, "mor4_trials.csv"), index=False)

    df["missing"] = df.rating_raw == 0
    ms = df.groupby("subject").agg(n_trials=("missing", "size"), n_missing=("missing", "sum"),
                                   runs=("run", lambda r: ",".join(map(str, sorted(r.unique())))))
    ms["frac_missing"] = ms.n_missing / ms.n_trials
    mc = df.pivot_table(index="subject", columns="condition_name", values="missing", aggfunc="mean")[COND]
    pd.concat([ms, mc.add_prefix("miss_")], axis=1).to_csv(os.path.join(OUT, "mor4_missingness.csv"))
    kd = df.rating_raw.value_counts().sort_index().rename("n").to_frame()
    kd.to_csv(os.path.join(OUT, "mor4_key_distribution.csv"))

    r = df[~df.missing]
    cm = r.groupby(["condition", "condition_name"]).rating_1to4.agg(["mean", "std", "count"])
    subj_cm = r.groupby(["subject", "condition_name"]).rating_1to4.mean().unstack()[COND]
    cm.to_csv(os.path.join(OUT, "mor4_condition_means.csv"))
    subj_cm.to_csv(os.path.join(OUT, "mor4_condition_means_by_subject.csv"))

    print(ms.to_string())
    print(kd.T.to_string())
    print(cm.round(2).to_string())
    print(f"\n{len(df)} trials, {df.subject.nunique()} subjects, {df.missing.mean():.1%} missing; "
          f"{(ms.frac_missing == 1).sum()} subjects with no responses at all")


if __name__ == "__main__":
    main()
