#!/usr/bin/env python3
"""E8e -- did it grow or did it click? Sigmoid vs free-changepoint on beta(t). [CPU]

Fits two shapes to each beta trajectory in log(step) and compares them by AICc, with a
bootstrap CI on the changepoint location. Both models have four free parameters, so the
comparison is driven by residual sum of squares; AICc is used anyway because n is about 25
and the small-sample correction is not negligible. Rules fixed in E8_PRESPEC.md section 6:

  smooth      y = b + L / (1 + exp(-k*(x - x0)))
  breakpoint  y = a + s1*(x - c) for x <= c, a + s2*(x - c) for x > c, continuous at c
  bar         |dAICc| >= 2 to prefer a model; below 2 the shapes are INDISTINGUISHABLE and
              neither "gradual" nor "sudden" is claimed
  changepoint 1000 bootstrap resamples over checkpoints, percentile CI. A CI spanning more
              than half the fitted log-step range is reported as UNLOCATED -- P4 requires
              both that the breakpoint model wins and that its changepoint is located

`step0` is excluded from the fit: log10(0) is undefined, and step 0 is the gate anchor
rather than a trajectory point. It is drawn on the E8d figure with its null band.

WHY THE CAVEAT MATTERS HERE. Apparent sharp emergence in LLMs is frequently an artifact of a
thresholded measure (Schaeffer et al., *Are Emergent Abilities of Large Language Models a
Mirage?*) -- discontinuous scoring manufactures phase transitions that continuous measures
dissolve. Standardised beta is continuous and unthresholded, which is why a breakpoint found
here would be a stronger claim than the usual one. It is also why a smooth curve is a real
result and is reported as plainly: the write-up does not prefer the more dramatic shape.

The fit is run at EVERY cell, not only the pre-designated primary one, and the distribution
of verdicts is reported. If the primary cell disagrees with the majority of cells, the
prespec makes that disagreement the headline rather than a footnote.

Outputs -> outputs/rsa/checkpoint_shape_fits.csv
"""
import argparse
import csv
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")

AICC_BAR = 2.0          # E8_PRESPEC.md section 6
N_BOOT_CP = 1000
MIN_SEG = 3             # points required on each side of a changepoint
TARGETS = ["outcome", "intent"]
SEED = 0


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def aicc(sse, n, n_params):
    """n_params counts the mean function's parameters; sigma adds one."""
    k = n_params + 1
    if n - k - 1 <= 0 or sse <= 0:
        return float("inf")
    return n * np.log(sse / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)


def fit_piecewise(x, y, min_seg=MIN_SEG):
    """Continuous two-segment linear fit. Linear in the parameters once c is fixed, so c is
    found by an exhaustive search over interior candidates rather than by a gradient method
    that could land in a local optimum."""
    best = None
    order = np.argsort(x)
    xs = x[order]
    cands = xs[min_seg - 1: len(xs) - min_seg + 1]
    for c in np.unique(cands):
        A = np.column_stack([np.ones_like(x), np.minimum(x - c, 0.0),
                             np.maximum(x - c, 0.0)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(c), beta)
    sse, c, beta = best
    return dict(sse=sse, changepoint=c, intercept=float(beta[0]),
                slope_lo=float(beta[1]), slope_hi=float(beta[2]), n_params=4)


def fit_sigmoid(x, y):
    """Logistic in log(step). Multi-start, because a logistic fit to a flat series has a
    ridge of near-equivalent (k, x0) and a single start can land anywhere on it."""
    from scipy.optimize import curve_fit

    def f(xx, b, L, k, x0):
        return b + L / (1.0 + np.exp(-np.clip(k * (xx - x0), -60, 60)))

    span = max(x.max() - x.min(), 1e-6)
    best = None
    for x0 in np.linspace(x.min(), x.max(), 6):
        for k in (0.5, 2.0, 8.0):
            p0 = [float(y[0]), float(y[-1] - y[0]) or 1e-3, k, float(x0)]
            try:
                p, _ = curve_fit(f, x, y, p0=p0, maxfev=20000)
            except Exception:
                continue
            sse = float(((y - f(x, *p)) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, p)
    if best is None:
        return dict(sse=float("inf"), n_params=4, b=np.nan, L=np.nan, k=np.nan, x0=np.nan)
    sse, p = best
    return dict(sse=sse, n_params=4, b=float(p[0]), L=float(p[1]),
                k=float(p[2]), x0=float(p[3]), midpoint=float(p[3]))


def bootstrap_changepoint(x, y, n_boot, rng):
    """Percentile CI on c, resampling checkpoints with replacement."""
    out = []
    for _ in range(n_boot):
        sel = rng.integers(0, len(x), len(x))
        xb, yb = x[sel], y[sel]
        if len(np.unique(xb)) < 2 * MIN_SEG:
            continue
        try:
            out.append(fit_piecewise(xb, yb)["changepoint"])
        except Exception:
            continue
    if len(out) < 50:
        return float("nan"), float("nan"), len(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), len(out)


def shape_for_series(steps, betas, rng):
    """Both fits, the AICc comparison, and the changepoint CI, for one trajectory."""
    m = np.array(steps) >= 1                  # step 0 is the gate anchor, not a fit point
    x = np.log10(np.array(steps, dtype=float)[m])
    y = np.array(betas, dtype=float)[m]
    n = len(x)
    if n < 2 * MIN_SEG + 2:
        return None

    pw = fit_piecewise(x, y)
    sg = fit_sigmoid(x, y)
    a_pw, a_sg = aicc(pw["sse"], n, 4), aicc(sg["sse"], n, 4)
    d = a_pw - a_sg                            # negative favours the breakpoint model
    sst = float(((y - y.mean()) ** 2).sum())

    lo, hi, n_ok = bootstrap_changepoint(x, y, N_BOOT_CP, rng)
    span = x.max() - x.min()
    ci_width = hi - lo if hi == hi else float("nan")
    located = bool(ci_width == ci_width and ci_width < 0.5 * span)

    if abs(d) < AICC_BAR:
        verdict = "indistinguishable"
    elif d < 0:
        verdict = "breakpoint" if located else "breakpoint (changepoint UNLOCATED)"
    else:
        verdict = "smooth"

    return dict(
        n_points=n, log_step_min=round(float(x.min()), 3),
        log_step_max=round(float(x.max()), 3),
        sse_piecewise=round(pw["sse"], 6), sse_sigmoid=round(sg["sse"], 6),
        r2_piecewise=round(1 - pw["sse"] / sst, 4) if sst > 0 else "",
        r2_sigmoid=round(1 - sg["sse"] / sst, 4) if sst > 0 else "",
        aicc_piecewise=round(a_pw, 3), aicc_sigmoid=round(a_sg, 3),
        delta_aicc=round(d, 3),
        changepoint_log10=round(pw["changepoint"], 4),
        changepoint_step=int(round(10 ** pw["changepoint"])),
        changepoint_ci_lo=round(lo, 4) if lo == lo else "",
        changepoint_ci_hi=round(hi, 4) if hi == hi else "",
        changepoint_ci_width=round(ci_width, 4) if ci_width == ci_width else "",
        half_fitted_range=round(0.5 * span, 4),
        changepoint_located=located, n_boot_cp_ok=n_ok,
        slope_before=round(pw["slope_lo"], 5), slope_after=round(pw["slope_hi"], 5),
        sigmoid_midpoint_log10=round(sg.get("midpoint", np.nan), 4)
        if sg.get("midpoint") == sg.get("midpoint") else "",
        sigmoid_rate_k=round(sg.get("k", np.nan), 4)
        if sg.get("k") == sg.get("k") else "",
        verdict=verdict,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default=os.path.join(RSA_DIR, "checkpoint_betas.csv"))
    ap.add_argument("--out", default=os.path.join(RSA_DIR, "checkpoint_shape_fits.csv"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.betas):
        raise SystemExit(f"missing {a.betas} — run E8d first")
    rows = list(csv.DictReader(open(a.betas)))

    series = {}
    for r in rows:
        if r["predictor"] not in TARGETS:
            continue
        key = (r["model"], r["position"], int(r["layer"]), r["rel_depth_requested"],
               r["spec"], r["predictor"])
        series.setdefault(key, []).append((int(r["step"]), float(r["beta"]),
                                           r["is_primary_cell"] in ("True", "true", "1")))

    if a.dry_run:
        print("=== E8e SHAPE TEST PLAN (dry-run) ===")
        print(f"series      : {len(series)}  (cell x spec x {TARGETS})")
        any_len = len(next(iter(series.values())))
        print(f"points/series: {any_len} checkpoints, step 0 excluded from the fit")
        print(f"bar         : |dAICc| >= {AICC_BAR}; changepoint CI from {N_BOOT_CP} boots")
        return

    guard = _load("_run_guard.py", "run_guard")
    guard.require_clean_tree(os.path.basename(__file__), allow_dirty=a.allow_dirty)

    rng = np.random.default_rng(SEED)
    out = []
    for key, pts in sorted(series.items()):
        model, position, layer, req, spec, predictor = key
        pts = sorted(set(pts))
        steps = [p[0] for p in pts]
        betas = [p[1] for p in pts]
        res = shape_for_series(steps, betas, rng)
        if res is None:
            continue
        out.append(dict(model=model, position=position, layer=layer,
                        rel_depth_requested=req, spec=spec, predictor=predictor,
                        is_primary_cell=pts[0][2], **res))

    if not out:
        raise SystemExit("no series long enough to fit")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"\n=== E8e: {len(out)} series fitted ===")
    for pred in TARGETS:
        sub = [r for r in out if r["predictor"] == pred and r["spec"] == "primary"]
        from collections import Counter
        c = Counter(r["verdict"].split(" (")[0] for r in sub)
        print(f"\n{pred} (spec=primary, {len(sub)} cells): " +
              ", ".join(f"{k}={v}" for k, v in c.most_common()))
        prim = [r for r in sub if r["is_primary_cell"]]
        if prim:
            p = prim[0]
            print(f"  PRIMARY CELL: {p['verdict']}  dAICc={p['delta_aicc']:+.2f} "
                  f"(negative favours breakpoint)")
            print(f"    R2  piecewise={p['r2_piecewise']}  sigmoid={p['r2_sigmoid']}")
            print(f"    changepoint at step ~{p['changepoint_step']} "
                  f"(log10 {p['changepoint_log10']}), 95% CI "
                  f"[{p['changepoint_ci_lo']}, {p['changepoint_ci_hi']}] vs half-range "
                  f"{p['half_fitted_range']} -> "
                  f"{'located' if p['changepoint_located'] else 'UNLOCATED'}")
            maj = c.most_common(1)[0][0]
            if p["verdict"].split(" (")[0] != maj:
                print(f"    NOTE: the primary cell says '{p['verdict']}' while the majority "
                      f"of cells say '{maj}'. Per the prespec that disagreement is the "
                      f"headline, not a footnote.")
    print(f"\n-> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
