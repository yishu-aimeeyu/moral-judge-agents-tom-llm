"""
stage_landmarks.py -- where in each story does intent / outcome information appear?

Stages are derived from the stimuli themselves, not from a belief-verb pattern: a span
carries INTENT information where the story differs from its intent-flipped partner (same
scenario, same outcome), and OUTCOME information where it differs from its outcome-flipped
partner. clause_offsets.csv is not used because its belief span sometimes starts at the
second belief sentence (e.g. YS2009_01 diverges ~65 chars before belief_start).

For every YS2008 / YS2009 story the order is the same:
  background | foreshadow (outcome) | belief (intent) | action | outcome sentence | question
YS2011 has no outcome partner (both cells are harm), so only the intent landmarks exist.

The foreshadow means outcome is already in the text during the belief stage. The
`no_foreshadow` variant deletes those sentences, so harm and no-harm versions are identical
text until the outcome sentence: anything outcome-like decoded before then must come from
the model's own expectations, not the prompt.
"""
import csv, difflib, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
STAGES = ["background", "foreshadow", "belief", "action", "outcome", "question"]
_FLIP_I = {"guilty": "innocent", "innocent": "guilty"}
_FLIP_O = {"harm": "no_harm", "no_harm": "harm"}


def _diff_blocks(a, b, merge_gap=25):
    """Char spans of `a` that differ from `b`; spans closer than merge_gap are merged."""
    out = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == "equal":
            continue
        i2 = max(i2, i1 + 1)
        if out and i1 - out[-1][1] < merge_gap:
            out[-1] = (out[-1][0], i2)
        else:
            out.append((i1, i2))
    return out


def _sentence_bounds(text, s, e):
    """Expand [s, e) outward to whole sentences."""
    a = max(text.rfind(". ", 0, s), text.rfind("? ", 0, s), text.rfind("! ", 0, s))
    a = 0 if a < 0 else a + 2
    ends = [i for i in (text.find(". ", e - 1), text.find("? ", e - 1)) if i >= 0]
    b = (min(ends) + 1) if ends else len(text)
    return a, b


def load_rows(path=MASTER):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["text"] = r["text"].strip()
    return rows


def landmarks(rows):
    """story_id -> dict of char positions (story text coordinates).

    intent_on / intent_off : first / last char of the intent-differing spans
    fore_on / fore_off     : outcome-differing spans before the belief, sentence-expanded
                             (None for YS2011)
    outcome_on             : sentence start of the first outcome-differing span after the belief
    story_end              : len(text)
    """
    key = {(r["source"], r["scenario_id"], r["intent_label"], r["outcome_label"]): r["text"]
           for r in rows}
    out = {}
    for r in rows:
        t = r["text"]
        ip = key.get((r["source"], r["scenario_id"], _FLIP_I[r["intent_label"]], r["outcome_label"]))
        op = key.get((r["source"], r["scenario_id"], r["intent_label"], _FLIP_O[r["outcome_label"]]))
        lm = dict(story_end=len(t), intent_on=None, intent_off=None,
                  fore_on=None, fore_off=None, outcome_on=None)
        if ip:
            bi = _diff_blocks(t, ip)
            lm["intent_on"], lm["intent_off"] = bi[0][0], bi[-1][1]
        if op and ip:
            bo = _diff_blocks(t, op)
            fore = [b for b in bo if b[0] < lm["intent_on"]]
            late = [b for b in bo if b[0] >= lm["intent_off"]]
            if fore:
                lm["fore_on"] = _sentence_bounds(t, *fore[0])[0]
                lm["fore_off"] = _sentence_bounds(t, *fore[-1])[1]
            if late:
                lm["outcome_on"] = _sentence_bounds(t, *late[0])[0]
        # "complete" = every stage boundary exists and is in order; a no_foreshadow story is
        # complete without a foreshadow (that stage then has zero length)
        ok = lm["intent_on"] is not None and lm["outcome_on"] is not None and \
            lm["intent_off"] <= lm["outcome_on"] and \
            (lm["fore_on"] is None or lm["fore_on"] <= lm["intent_on"])
        lm["complete"] = bool(ok)
        out[r["story_id"]] = lm
    return out


def no_foreshadow(rows):
    """Copy of rows with the foreshadow sentences deleted (rows without one are dropped)."""
    lm = landmarks(rows)
    out = []
    for r in rows:
        m = lm[r["story_id"]]
        if not m["complete"] or m["fore_on"] is None or m["fore_off"] > m["intent_on"]:
            continue
        t = r["text"]
        r2 = dict(r, text=(t[:m["fore_on"]] + t[m["fore_off"]:].lstrip()).strip())
        out.append(r2)
    return out
