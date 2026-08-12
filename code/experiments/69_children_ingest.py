#!/usr/bin/env python3
"""E7e: ingest the open ToM booklet dataset and build a second, independent child anchor.

SOURCE. Sotomayor-Enriquez, Gweon, Saxe & Richardson, "Theory of Mind Booklet Task & Open
Dataset", OSF https://osf.io/g5zpv/ (public; n=321 children aged 3-12, two booklet versions,
including 64 children assessed longitudinally and 68 autistic children).

WHY IT IS WORTH THE TROUBLE. The existing child anchor is digitized off a published figure
(Cushman et al. 2013, Fig. 3), so its values carry read error of roughly +/-0.03-0.05 and cannot
be re-binned or error-barred. This dataset is item-level and raw, so the same developmental
quantity can be computed with real n per bin and real confidence intervals, from a different
sample, different stimuli and a different response format. Agreement between two such different
measurements is worth much more than either alone.

WHAT MAPS ONTO WHAT, AND WHAT DOES NOT.
  * The booklet asks two blame questions in a false-belief frame -- "Is Allie being
    mean/naughty?" and "Should Allie get in trouble?", correct answer No, because Allie took
    Sue's snack believing it was her own -- and the same two questions in a true-belief frame,
    where Melanie knowingly eats Mitch's cookie and the correct answer is Yes.
  * naughty items are a WRONGNESS judgment and trouble items are a PUNISHMENT judgment, which is
    exactly the distinction Cushman's Naughty and Punish series draw. Both series are produced
    here so the comparison is like-for-like, and `methods_child_measure.md`'s argument for
    preferring wrongness carries over unchanged.
  * THE CONTRAST IS NOT THE SAME CONTRAST. Cushman's child bands give attempted minus
    accidental: intent present and no harm, against no intent and harm. The booklet's
    true-belief item is INTENTIONAL harm -- intent and harm together. So what this anchor
    measures is intentional minus accidental, the other diagonal of the 2x2. It is an
    intent-based blame contrast and it is developmentally informative, but it must not be
    substituted for the attempted-accidental number, and this script never labels it as such.
  * The response is a binary forced choice, not a 7-point rating, so a proportion here is a
    proportion of children choosing to blame, not a mean blame magnitude. Comparing the two
    anchors is comparing shapes, not units.

DERIVATION. Under the mapping above, blame on a false-belief (accidental-type) act is the
proportion of children who answer INCORRECTLY, i.e. 1 - accuracy: a child who exculpates Allie is
scored correct, and a child who calls her naughty is the one attributing blame. On a true-belief
(intentional) act, blame is accuracy itself. Hence:
      blame_accidental  = 1 - accuracy(false-belief items)
      blame_intentional =     accuracy(true-belief items)
      contrast          = blame_intentional - blame_accidental

Only Booklet 2 contains true-belief moral items, so the contrast is available only in that
sample; Booklet 1 contributes the accidental limb, at much larger n.

Usage:
  python code/experiments/69_children_ingest.py --fetch     # download from OSF (idempotent)
  python code/experiments/69_children_ingest.py --build     # parse, derive, write CSVs + README
  python code/experiments/69_children_ingest.py --plot      # new figure; touches no existing one
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = os.path.join(ROOT, "dataset", "human_reference", "children")
RAW = os.path.join(DEST, "raw")
FIGDIR = os.path.join(ROOT, "outputs", "figures_final")
OSF_NODE = "g5zpv"

WORKBOOKS = [("ToMBooklet1-Data.xlsx", "booklet1"), ("ToMBooklet2-Data.xlsx", "booklet2")]

# Cushman's bands, reused so the two anchors are binned identically rather than similarly.
BANDS = [("child_3", 3.0, 4.0), ("child_4_5", 4.0, 6.0), ("child_6_7", 6.0, 8.0),
         ("child_8plus", 8.0, 13.0)]
CUSHMAN_NAUGHTY = {  # cushman_child_bands_PROPOSED.csv, presented_first
    "child_4_5": (0.455, 0.695), "child_6_7": (0.200, 0.700), "child_8plus": (0.060, 0.690)}
CUSHMAN_PUNISH = {   # cushman_child_bands_PUNISH.csv, presented_first
    "child_4_5": (0.590, 0.680), "child_6_7": (0.415, 0.535), "child_8plus": (0.220, 0.410)}


# ------------------------------------------------------------------ fetch
def fetch():
    os.makedirs(RAW, exist_ok=True)
    got = []

    def walk(url):
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for it in d.get("data", []):
            a = it["attributes"]
            if a["kind"] == "folder":
                walk(it["relationships"]["files"]["links"]["related"]["href"])
            else:
                out = os.path.join(RAW, a["name"])
                if os.path.exists(out) and os.path.getsize(out) == a.get("size"):
                    print(f"  have {a['name']}")
                else:
                    urllib.request.urlretrieve(it["links"]["download"], out)
                    print(f"  fetched {a['name']} ({os.path.getsize(out)} B)")
                got.append((a["name"], os.path.getsize(out)))

    walk(f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/?page[size]=100")
    print(f"{len(got)} files in {os.path.relpath(RAW, ROOT)}")
    return got


# ------------------------------------------------------------------ parse
def _sheet(wb, name):
    ws = wb[name]
    rows = ws.iter_rows(values_only=True)
    hdr = [(str(h).strip() if h is not None else "") for h in next(rows)]
    out = []
    for r in rows:
        out.append({hdr[i]: r[i] for i in range(min(len(hdr), len(r)))})
    return out


def _f(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def measure_of(question):
    """naughty items are a wrongness judgment; trouble items a punishment judgment.

    Two categories are deliberately NOT pooled into either series:

    * RESTITUTION. `ToMS_2AFC_16.9` reads "Now Mitch has a cookie. Should Melanie STILL get in
      trouble?", which asks whether compensating the victim cancels desert -- a different
      question from whether the act deserves punishment at all. Pooling it into the punishment
      series drags that series down hard (accuracy 0.18 at age 3 against 0.72 for the plain
      trouble item at the same age) and would have made the punishment anchor look like a
      developmental story about intent when it was partly about restitution.
    * THIRD-PARTY ANGER. "Is it fair that mother is mad with J?", "Is J mad at Isa?" and "Is
      mother mad at Isa?" are judgments about someone else's emotional reaction, not the child's
      own blame attribution. They are kept in the item-level table and excluded from the anchor.
    """
    q = (question or "").lower()
    if "still get in trouble" in q or "now mitch has a cookie" in q:
        return "restitution"
    if "naughty" in q or "mean/" in q:
        return "naughty"
    if "trouble" in q:
        return "trouble"
    return "other"


def frame_of(concept):
    c = (concept or "")
    if "False Belief" in c:
        return "false_belief"
    if "True Belief" in c:
        return "true_belief"
    return None


def load_all():
    try:
        import openpyxl
    except ImportError:
        raise SystemExit(
            "openpyxl is required to read the OSF workbooks: pip install openpyxl\n"
            "(code/requirements.txt is committed but empty, so it will not install this for you.)")
    items, people = [], {}
    for fname, book in WORKBOOKS:
        p = os.path.join(RAW, fname)
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        info = _sheet(wb, "Participant Info")
        for r in info:
            sid = r.get("SubID")
            if not sid:
                continue
            people[str(sid)] = dict(
                sub=str(sid), booklet=book, project=str(r.get("Project") or ""),
                age=_f(r.get("Age")), gender=str(r.get("Gender") or ""),
                # Booklet 2's sheet has no ASD column; its projects are neurotypical samples.
                asd=(1 if _f(r.get("ASD Diagnosis")) == 1.0 else 0),
                long_id=str(r.get("LongID") or "NA"))
        for r in _sheet(wb, "Data"):
            sid = r.get("subID")
            if not sid:
                continue
            frame = frame_of(r.get("Concept-super") == "Moral Reasoning"
                             and r.get("Concept") or "")
            if frame is None or str(r.get("Qtype")) != "2AFC":
                continue
            # Headers are stripped in _sheet, so the source's "Include? " loses its trailing
            # space. Both spellings are looked up rather than assuming which one survived.
            ans = _f(r.get("Answer (0 /1)"))
            inc = _f(r.get("Include?", r.get("Include? ")))
            err = _f(r.get("Exp Error?"))
            items.append(dict(
                sub=str(sid), booklet=book, q_id=str(r.get("Question_ID")),
                question=str(r.get("Question") or ""), frame=frame,
                measure=measure_of(r.get("Question")), correct=ans,
                include=inc, exp_error=err))
        wb.close()
    return items, people


def usable(it):
    """The dataset's own exclusion flags, honoured rather than reimplemented."""
    return (it["correct"] in (0.0, 1.0) and it["include"] == 1.0 and it["exp_error"] == 0.0)


def band_of(age):
    if age is None:
        return None
    for name, lo, hi in BANDS:
        if lo <= age < hi:
            return name
    return None


def wilson(k, n, z=1.96):
    """Wilson interval: the bands have small n, where a normal approximation misbehaves."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------ build
def build(first_session_only=False):
    items, people = load_all()
    keep = [it for it in items if usable(it)]
    n_flagged_out = len(items) - len(keep)

    # A longitudinally-assessed child contributes several sessions under one LongID, which are
    # not independent observations. The primary tables use every session (each is a valid
    # age-accuracy observation) and --first-session-only re-runs on one session per child, so the
    # dependence can be shown to change nothing rather than assumed to.
    if first_session_only:
        # LongID is a CROSS-BOOKLET pointer, not a within-sheet repeat: a Booklet 1 participant
        # whose LongID is "Booklet2-030" is the same child, seen again at an older age. Both
        # sheets carry 64 such pointers, and 385 sessions - 64 links = 321 children, which is the
        # n the dataset reports. Keying on the sorted pair collapses each link to one child.
        allow, seen = set(), set()
        for sub, p in sorted(people.items(), key=lambda kv: (kv[1]["age"] or 99)):
            lid = p["long_id"]
            key = tuple(sorted((sub, lid))) if lid in people else (sub,)
            if key not in seen:
                seen.add(key)
                allow.add(sub)
        keep = [it for it in keep if it["sub"] in allow]

    rows_item, rows_band, rows_year = [], [], []
    nt = {s: p for s, p in people.items() if p["asd"] == 0}

    # ---- item-level accuracy by age band
    agg = defaultdict(lambda: [0, 0])
    for it in keep:
        p = nt.get(it["sub"])
        if p is None:
            continue
        b = band_of(p["age"])
        if b is None:
            continue
        agg[(it["booklet"], it["q_id"], it["question"], it["frame"], it["measure"], b)][0] += \
            int(it["correct"])
        agg[(it["booklet"], it["q_id"], it["question"], it["frame"], it["measure"], b)][1] += 1
    for (book, qid, q, frame, meas, b), (k, n) in sorted(agg.items()):
        lo, hi = wilson(k, n)
        rows_item.append(dict(booklet=book, q_id=qid, question=q, frame=frame, measure=meas,
                              age_band=b, n=n, n_correct=k, accuracy=round(k / n, 4),
                              ci95_lo=round(lo, 4), ci95_hi=round(hi, 4)))

    # ---- age-binned anchor, per measure
    def summarise(bucket_fn, label_field):
        out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for it in keep:
            p = nt.get(it["sub"])
            if p is None or it["measure"] not in ("naughty", "trouble"):
                continue
            bucket = bucket_fn(p["age"])
            if bucket is None:
                continue
            cell = out[(bucket, it["measure"])][it["frame"]]
            cell[0] += int(it["correct"])
            cell[1] += 1
        res = []
        for (bucket, meas), fr in sorted(out.items(), key=lambda kv: str(kv[0])):
            fb_k, fb_n = fr.get("false_belief", [0, 0])
            tb_k, tb_n = fr.get("true_belief", [0, 0])
            acc_fb = fb_k / fb_n if fb_n else None
            acc_tb = tb_k / tb_n if tb_n else None
            blame_acc = (1 - acc_fb) if acc_fb is not None else None
            blame_int = acc_tb if acc_tb is not None else None
            fb_lo, fb_hi = wilson(fb_k, fb_n)
            tb_lo, tb_hi = wilson(tb_k, tb_n)
            res.append({
                label_field: bucket, "measure": meas,
                "n_obs_false_belief": fb_n, "n_obs_true_belief": tb_n,
                "accuracy_false_belief": None if acc_fb is None else round(acc_fb, 4),
                "accuracy_true_belief": None if acc_tb is None else round(acc_tb, 4),
                "blame_accidental": None if blame_acc is None else round(blame_acc, 4),
                "blame_accidental_ci95_lo": None if fb_n == 0 else round(1 - fb_hi, 4),
                "blame_accidental_ci95_hi": None if fb_n == 0 else round(1 - fb_lo, 4),
                "blame_intentional": None if blame_int is None else round(blame_int, 4),
                "blame_intentional_ci95_lo": None if tb_n == 0 else round(tb_lo, 4),
                "blame_intentional_ci95_hi": None if tb_n == 0 else round(tb_hi, 4),
                "contrast_intentional_minus_accidental":
                    None if (blame_acc is None or blame_int is None)
                    else round(blame_int - blame_acc, 4)})
        return res

    rows_band = summarise(band_of, "age_band")
    rows_year = summarise(lambda a: (None if a is None or not (3 <= a < 13)
                                     else f"age_{int(a)}"), "age_year")

    n_kids = len({it["sub"] for it in keep})
    n_nt = len({it["sub"] for it in keep if it["sub"] in nt})
    prov = dict(n_sessions_with_moral_2afc=n_kids, n_neurotypical_sessions=n_nt,
                n_moral_2afc_observations=len(keep),
                n_excluded_by_source_flags=n_flagged_out,
                n_dropped_as_repeat_session=(len(items) - n_flagged_out - len(keep)),
                first_session_only=first_session_only)
    return rows_item, rows_band, rows_year, prov


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"  wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


# ------------------------------------------------------------------ plot
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_rows = list(csv.DictReader(open(os.path.join(DEST, "children_child_anchor.csv"))))
    order = [b for b, _, _ in BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for ax, meas, cush, cname in ((axes[0], "naughty", CUSHMAN_NAUGHTY, "Naughty (wrongness)"),
                                  (axes[1], "trouble", CUSHMAN_PUNISH, "Punish")):
        sub = {r["age_band"]: r for r in band_rows if r["measure"] == meas}
        xs = [b for b in order if b in sub]
        x = list(range(len(xs)))

        def series(key):
            out = []
            for b in xs:
                v = sub[b].get(key, "")
                out.append(float(v) if v not in ("", None) else float("nan"))
            return out

        acc, inten = series("blame_accidental"), series("blame_intentional")
        acc_lo, acc_hi = series("blame_accidental_ci95_lo"), series("blame_accidental_ci95_hi")
        int_lo, int_hi = series("blame_intentional_ci95_lo"), series("blame_intentional_ci95_hi")
        ax.errorbar(x, acc, yerr=[[a - lo for a, lo in zip(acc, acc_lo)],
                                  [hi - a for a, hi in zip(acc, acc_hi)]],
                    marker="o", color="#c0392b", capsize=3, lw=2,
                    label="booklet: blame(accidental) = 1 − acc(false belief)")
        ax.errorbar(x, inten, yerr=[[a - lo for a, lo in zip(inten, int_lo)],
                                    [hi - a for a, hi in zip(inten, int_hi)]],
                    marker="s", color="#1f6f3f", capsize=3, lw=2,
                    label="booklet: blame(intentional) = acc(true belief)")
        cx = [i for i, b in enumerate(xs) if b in cush]
        if cx:
            ax.plot(cx, [cush[xs[i]][0] for i in cx], marker="o", ls="--", color="#c0392b",
                    alpha=0.45, lw=1.6, label="Cushman 2013: blame(accidental)")
            ax.plot(cx, [cush[xs[i]][1] for i in cx], marker="^", ls="--", color="#8e44ad",
                    alpha=0.55, lw=1.6, label="Cushman 2013: blame(attempted)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([b.replace("child_", "").replace("8plus", "8+").replace("_", "–")
                            for b in xs])
        ax.set_xlabel("age band (years)")
        ax.set_ylabel("proportion attributing blame")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="grey", lw=0.7, ls=":")
        ax.set_title(f"{cname}\nbooklet (solid, 95% Wilson CI) vs digitized Cushman (dashed)",
                     fontsize=10)
        ax.legend(fontsize=7.0, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.25)
    fig.suptitle("Two independent child anchors. Dashed lines are the existing digitized "
                 "Cushman shape; solid lines are the OSF booklet dataset (n per point in "
                 "children_child_anchor.csv).\nThe booklet's intent limb is INTENTIONAL harm, "
                 "not attempted harm, so the green and purple series are different contrasts "
                 "and are not interchangeable.", fontsize=8.5)
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, "child_anchor_booklet_vs_cushman.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p, ROOT)}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--first-session-only", action="store_true")
    a = ap.parse_args()
    if not any((a.fetch, a.build, a.plot)):
        ap.error("pass --fetch, --build and/or --plot")
    os.makedirs(DEST, exist_ok=True)
    if a.fetch:
        fetch()
    if a.build:
        item, band, year, prov = build(a.first_session_only)
        sfx = "_first_session_only" if a.first_session_only else ""
        _write(os.path.join(DEST, f"children_moral_item_accuracy{sfx}.csv"), item)
        _write(os.path.join(DEST, f"children_child_anchor{sfx}.csv"), band)
        _write(os.path.join(DEST, f"children_child_anchor_by_year{sfx}.csv"), year)
        print("  provenance:", prov)
        json.dump(prov, open(os.path.join(DEST, f"ingest_provenance{sfx}.json"), "w"), indent=1)
    if a.plot:
        plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
