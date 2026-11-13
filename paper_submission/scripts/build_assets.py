#!/usr/bin/env python3
"""Build manuscript-only figures and LaTeX tables from authoritative project outputs."""

from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace("+", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def model_key(name: str) -> str:
    """Normalize hub prefixes and punctuation without collapsing model identity."""
    basename = str(name).rsplit("/", 1)[-1]
    basename = re.sub(
        r"^(huggingfaceh4|unsloth|allenai|mistralai)[-_]",
        "",
        basename,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^a-z0-9]", "", basename.lower())


def short_name(name: str) -> str:
    substitutions = {
        "OLMo-2-1124-7B-Instruct": "OLMo-2-7B-I",
        "OLMo-2-1124-7B": "OLMo-2-7B",
        "HuggingFaceH4-zephyr-7b-beta": "Zephyr-7B",
        "Mistral-7B-Instruct-v0-3": "Mistral-7B-I",
        "Mistral-7B-v0-3": "Mistral-7B",
        "unsloth-gemma-2-9b-it": "Gemma-2-9B-I",
        "unsloth-gemma-2-9b": "Gemma-2-9B",
        "Llama-3-1-Tulu-3-8B": "Tulu-3-8B",
        "unsloth-Meta-Llama-3-1-8B-Instruct": "Llama-3.1-8B-I",
        "unsloth-Meta-Llama-3-1-8B": "Llama-3.1-8B",
        "Qwen2.5-14B-Instruct": "Qwen2.5-14B-I",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B-I",
        "Qwen2.5-3B-Instruct": "Qwen2.5-3B-I",
        "Qwen2.5-1-5B-Instruct": "Qwen2.5-1.5B-I",
        "Qwen2.5-0-5B-Instruct": "Qwen2.5-0.5B-I",
        "Qwen2.5-1-5B": "Qwen2.5-1.5B",
        "Qwen2.5-0-5B": "Qwen2.5-0.5B",
        "gpt-5_5": "GPT-5.5",
        "gpt-5_4-mini": "GPT-5.4-mini",
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt-4o": "GPT-4o",
        "claude-opus-4-6": "Claude Opus 4.6",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        "gemini-3_5-flash": "Gemini 3.5 Flash",
        "gemini-2_5-flash": "Gemini 2.5 Flash",
    }
    if name in substitutions:
        return substitutions[name]
    normalized = {model_key(source): target for source, target in substitutions.items()}
    return normalized.get(model_key(name), name)


def copy_verified_figures() -> list[str]:
    mappings = {
        ROOT / "outputs/figures_final/tom_vs_contrast.png": "tom_vs_contrast.png",
        ROOT / "outputs/figures_final/interaction_forest.png": "interaction_forest.png",
        ROOT / "outputs/figures_final/checkpoint_dissection.png": "checkpoint_dissection.png",
        ROOT / "outputs/figures_final/layerwise_curves.png": "layerwise_curves.png",
        ROOT / "outputs/figures_final/gap_over_surface_span_matched.png": "gap_over_surface.png",
        ROOT / "outputs/figures_final/item_level_dissociation.png": "item_level_dissociation.png",
        ROOT / "outputs/figures_final/w3_steering_dose_OLMo.png": "w3_steering_OLMo.png",
        ROOT / "outputs/figures_final/w3_steering_dose_Qwen.png": "w3_steering_Qwen.png",
        ROOT / "outputs/figures_final/w3_prose_rating.png": "w3_prose_rating.png",
        ROOT / "outputs/figures_final/w4_curriculum.png": "w4_curriculum.png",
        ROOT / "outputs/figures_final/w7_bruneau_selectivity.png": "w7_bruneau_selectivity.png",
        ROOT / "outputs/figures_final/crosscultural_placement.png": "crosscultural_placement.png",
        ROOT / "outputs/figures_final/child_anchor_booklet_vs_cushman.png": "child_anchor_booklet_vs_cushman.png",
        ROOT / "outputs/rsa/checkpoint_emergence_by_position.png": "e8_pythia.png",
        ROOT / "outputs/rsa/checkpoint_trajectory_OLMo-2-1124-7B_joined.png": "e8_olmo.png",
        ROOT / "outputs/rsa/checkpoint_stage_rsa.png": "e2_checkpoint_stage_rsa.png",
        ROOT / "outputs/rsa/rsa_vs_behavior_scatter.png": "rsa_vs_behavior.png",
        ROOT / "presentation_figures/mechanistic/rsa_vs_behavior_scatter.png": "rsa_vs_behavior.png",
        ROOT / "outputs/experiments/w8a_amnesic_OLMo-2-1124-7B-Instruct.png": "w8a_OLMo.png",
        ROOT / "outputs/experiments/w8a_amnesic_Qwen2.5-7B-Instruct.png": "w8a_Qwen.png",
        ROOT / "outputs/experiments/w8a_amnesic_Llama-3.1-Tulu-3-8B.png": "w8a_Tulu.png",
        ROOT / "outputs/experiments/w8a_nuisance_specificity.png": "w8a_nuisance_specificity.png",
        ROOT / "outputs/experiments/e6_orthogonalized_specificity.png": "e6_specificity.png",
        ROOT / "outputs/experiments/e7b_residual_control.png": "e7b_residual_control.png",
    }
    copied: list[str] = []
    for source, destination in mappings.items():
        target = FIGURES / destination
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            copied.append(destination)
    return copied


def plot_behavioral_overview(open_rows: list[dict[str, str]], closed_rows: list[dict[str, str]]) -> None:
    open_models = [row for row in open_rows if row["type"] != "human"]
    open_models.sort(key=lambda row: number(row["contrast"]))
    closed_rows = sorted(closed_rows, key=lambda row: number(row["contrast"]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 8.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    panels = [
        (axes[0], open_models, "Open-weight models", "open"),
        (axes[1], closed_rows, "Closed-API models", "closed"),
    ]
    colors = {
        "base": "#8c8c8c",
        "instruct": "#2f5aa8",
        "OpenAI": "#2d6a4f",
        "Anthropic": "#7b2cbf",
        "Google": "#d97706",
    }
    for axis, rows, title, protocol in panels:
        ys = list(range(len(rows)))
        for y, row in zip(ys, rows):
            estimate = number(row["contrast"])
            low = number(row["ci_lo"])
            high = number(row["ci_hi"])
            key = row.get("type", "") if protocol == "open" else row.get("provider", "")
            axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                ms=5.5,
                capsize=2.2,
                color=colors.get(key, "#555555"),
                ecolor=colors.get(key, "#555555"),
                lw=1.2,
            )
        axis.set_yticks(ys)
        axis.set_yticklabels([short_name(row["model"]) for row in rows], fontsize=8)
        axis.axvline(0, color="#222222", lw=0.9)
        axis.axvline(0.666, color="#b03a2e", lw=1.1, ls="--")
        axis.set_xlim(-0.82, 0.78)
        axis.grid(axis="x", color="#dddddd", lw=0.6)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Attempted − accidental normalized condemnation", fontsize=9.5)
        axis.text(
            0.666,
            -1.1,
            "adult reference +0.666",
            color="#b03a2e",
            fontsize=7.5,
            ha="right",
            va="bottom",
        )
    axes[0].text(
        -0.80,
        len(open_models) - 0.2,
        "gray = base; blue = instruct",
        fontsize=8,
        color="#555555",
    )
    axes[1].text(
        -0.80,
        len(closed_rows) - 0.2,
        "green = OpenAI; purple = Anthropic; orange = Google",
        fontsize=7.5,
        color="#555555",
    )
    fig.suptitle(
        "Behavioral contrast under separate open-weight and closed-API protocols",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "behavioral_overview.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "behavioral_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_longtables(
    open_rows: list[dict[str, str]],
    closed_rows: list[dict[str, str]],
    tom_rows: list[dict[str, str]],
    stage_rows: list[dict[str, str]],
) -> None:
    lines: list[str] = [
        "% Generated by scripts/build_assets.py. Do not edit by hand.",
        r"\begin{longtable}{@{}llrrrrl@{}}",
        r"\caption{Open-weight behavioral results. Human reference shapes are reported separately in \cref{tab:human-reference}.}\label{tab:open-full}\\",
        r"\toprule",
        r"Model & Type & Contrast & 95\% low & 95\% high & Degenerate & Cell order \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Type & Contrast & 95\% low & 95\% high & Degenerate & Cell order \\",
        r"\midrule",
        r"\endhead",
    ]
    mixed = {
        model_key(row["model"]): row
        for row in read_csv(ROOT / "outputs/stats/mixed_effects_2x2.csv")
        if row.get("spec") == "primary" and row.get("model") != "HUMAN (Young 2007)"
    }
    for row in [item for item in open_rows if item["type"] != "human"]:
        name = row["model"]
        lookup = mixed.get(model_key(name), {})
        lines.append(
            "{} & {} & {:+.3f} & {:+.3f} & {:+.3f} & {} & {} \\\\".format(
                tex_escape(short_name(name)),
                tex_escape(row["type"]),
                number(row["contrast"]),
                number(row["ci_lo"]),
                number(row["ci_hi"]),
                "yes" if row.get("degenerate") == "True" else "no",
                tex_escape(lookup.get("cell_order", "not available")),
            )
        )
    lines += [r"\bottomrule", r"\end{longtable}", ""]

    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Complete clean closed-API behavioral reruns (v3; provisional pending archival provenance). Gemini-2.5-Pro was partial and excluded.}",
        r"\label{tab:closed-full}",
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"Model & Contrast & 95\% low & 95\% high & CI excludes 0 \\",
        r"\midrule",
    ]
    for row in closed_rows:
        lines.append(
            "{} & {:+.3f} & {:+.3f} & {:+.3f} & {} \\\\".format(
                tex_escape(short_name(row["model"])),
                number(row["contrast"]),
                number(row["ci_lo"]),
                number(row["ci_hi"]),
                "yes" if row.get("sig_vs_0") == "yes" else "no",
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    engaged = [
        row
        for row in tom_rows
        if row.get("engaged") == "True" and row.get("mtype") == "instruct"
    ]
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Joint BigToM and moral-judgment results for engaged open instruct models.}",
        r"\label{tab:bigtom-engaged}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Model & BigToM false belief & Moral contrast \\",
        r"\midrule",
    ]
    for row in sorted(engaged, key=lambda item: number(item["contrast"])):
        lines.append(
            "{} & {:.3f} & {:+.3f} \\\\".format(
                tex_escape(short_name(row["model"].split("_", 1)[-1].replace("_", "-"))),
                number(row["bigtom_false"]),
                number(row["contrast"]),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Post-training checkpoint dissection.}",
        r"\label{tab:stages}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Family & Stage & Contrast & $\beta_{\mathrm{intent}}$ & $\beta_{\mathrm{outcome}}$ & Rating SD \\",
        r"\midrule",
    ]
    for row in stage_rows:
        lines.append(
            "{} & {} & {:+.3f} & {:+.3f} & {:+.3f} & {:.3f} \\\\".format(
                tex_escape(row["family"]),
                tex_escape(row["stage"]),
                number(row["contrast"]),
                number(row["b_intent"]),
                number(row["b_outcome"]),
                number(row["rating_std"]),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    (GENERATED / "result_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def write_asset_manifest(copied: list[str]) -> None:
    all_assets = sorted(path.name for path in FIGURES.glob("*") if path.is_file())
    text = [
        "# Paper asset manifest",
        "",
        "Generated by `scripts/build_assets.py` from project-local, post-repair artifacts.",
        "",
        "## Assets present",
        *[f"- `{name}`" for name in all_assets],
        "",
        "## Copied in this run",
        *([f"- `{name}`" for name in copied] or ["- None; existing copies were retained."]),
        "",
        "The manuscript-specific `behavioral_overview` figure is generated directly from "
        "`outputs/master_all_models_digitized_openonly.csv` and "
        "`outputs/agents/stats/contrast_by_model.csv`.",
    ]
    (PAPER / "ASSET_MANIFEST.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    open_rows = read_csv(ROOT / "outputs/master_all_models_digitized_openonly.csv")
    closed_rows = read_csv(ROOT / "outputs/agents/stats/contrast_by_model.csv")
    tom_rows = read_csv(ROOT / "outputs/tom_benchmarks/tom_vs_contrast.csv")
    stage_rows = read_csv(ROOT / "outputs/experiments/checkpoint_dissection.csv")
    copied = copy_verified_figures()
    plot_behavioral_overview(open_rows, closed_rows)
    write_longtables(open_rows, closed_rows, tom_rows, stage_rows)
    write_asset_manifest(copied)
    print(f"Built paper assets in {PAPER}")


if __name__ == "__main__":
    main()
