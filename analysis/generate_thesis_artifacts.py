"""Generate the paper-facing tables, statistics, diagrams, and plots.

All reported artifacts are derived from one versioned result directory.  The
script also checks key uniqueness and paired-seed invariants so stale or mixed
experiment exports fail before they reach the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "thesis_final_v7"

PROPOSED = "ECB-HD-CR"
METHOD_ORDER = [
    PROPOSED,
    "D3QN-EH",
    "ECB-Static",
    "Two-Timescale-DRL",
    "ICR-HHODRL",
    "CHHFO",
]
BASELINES = [method for method in METHOD_ORDER if method != PROPOSED]
DIRECT_TRIO = ["D3QN-EH", "ECB-Static", "Two-Timescale-DRL"]
# Plan B: report each direct baseline separately (they are no longer equivalent).
DISPLAY_METHODS = [
    PROPOSED,
    "D3QN-EH",
    "ECB-Static",
    "Two-Timescale-DRL",
    "ICR-HHODRL",
    "CHHFO",
]
ABLATION_ORDER = [
    PROPOSED,
    "ECB-HD-CR-NoCluster",
    "ECB-HD-CR-NoAggregation",
    "ECB-HD-CR-NoRoleCredit",
    "ECB-HD-CR-NoFeedback",
]
REGIME_ORDER = [
    PROPOSED,
    "ECB-Static",
    "ECB-HD-CR-NoFeedback",
    "ECB-HD-CR-NoRoleCredit",
    "ECB-HD-CR-NoCluster",
    "ECB-HD-CR-NoAggregation",
]
MISMATCH_ORDER = [
    PROPOSED,
    "ECB-HD-CR-NoFeedback",
    "ECB-Static",
    "Two-Timescale-DRL",
    "D3QN-EH",
    "ECB-HD-CR-NoCluster",
]

METHOD_LABEL = {
    PROPOSED: "ECB-HD-CR",
    "D3QN-EH": "D3QN-EH",
    "ECB-Static": "ECB-Static",
    "Two-Timescale-DRL": "Two-Timescale-DRL",
    "ICR-HHODRL": "ICR-HHODRL",
    "CHHFO": "CHHFO",
    "ECB-HD-CR-NoCluster": "NoCluster",
    "ECB-HD-CR-NoAggregation": "NoAggregation",
    "ECB-HD-CR-NoRoleCredit": "NoRoleCredit",
    "ECB-HD-CR-NoFeedback": "NoFeedback",
}

SCENARIO_LABEL = {
    "centered_bs": "centered BS",
    "scarce_high_load": "scarce high load",
    "small_battery_bursty": "small battery/high load",
    "high_pressure": "high pressure",
    "harvest_uncertain": "moderate harvest/load",
    "edge_bs_high_load": "edge-BS high load",
    "critical_scarcity": "critical scarcity",
    "shift_large_field_edge": "regime shift",
    "credit_mismatch_edge": "credit mismatch",
}

COLORS = {
    PROPOSED: "#146C94",
    "D3QN-EH": "#2A9D8F",
    "ECB-Static": "#6C757D",
    "Two-Timescale-DRL": "#E9C46A",
    "ICR-HHODRL": "#F4A261",
    "CHHFO": "#C94C4C",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_phase(results: Path, phase: str) -> pd.DataFrame:
    path = results / phase / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing result export: {path}")
    frame = pd.read_csv(path)
    key = ["scenario", "method", "run_id"]
    if phase == "nsrdb":
        key.insert(0, "harvest_source")
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate result keys in {path}:\n{frame.loc[duplicates, key]}")
    return frame


def _aggregate(frame: pd.DataFrame, extra_groups: list[str] | None = None) -> pd.DataFrame:
    groups = list(extra_groups or []) + ["scenario", "method"]
    agg_map = {
        "n": ("run_id", "count"),
        "completion_mean": ("completion_rate", "mean"),
        "completion_std": ("completion_rate", "std"),
        "fnd_mean": ("FND", "mean"),
        "fnd_std": ("FND", "std"),
        "scheduling_energy_mean": ("total_energy", "mean"),
        "scheduling_energy_std": ("total_energy", "std"),
        "background_energy_mean": ("background_energy", "mean"),
        "physical_energy_mean": ("total_physical_energy", "mean"),
        "energy_per_task_mean": ("energy_per_completed_task", "mean"),
        "energy_per_task_std": ("energy_per_completed_task", "std"),
        "control_overhead_mean": ("control_overhead_pct", "mean"),
        "budget_deviation_mean": ("budget_deviation", "mean"),
        "utility_mean": ("utility", "mean"),
    }
    if "pre_switch_completion" in frame.columns:
        agg_map["pre_switch_completion_mean"] = ("pre_switch_completion", "mean")
        agg_map["post_switch_completion_mean"] = ("post_switch_completion", "mean")
        agg_map["post_switch_fnd_mean"] = ("post_switch_fnd", "mean")
    return (
        frame.groupby(groups, sort=False)
        .agg(**agg_map)
        .reset_index()
    )


def _holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    total = len(p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(p)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def _paired_tests(frame: pd.DataFrame, alternative: str) -> pd.DataFrame:
    records: list[dict] = []
    for scenario in frame["scenario"].drop_duplicates():
        scenario_frame = frame[frame["scenario"] == scenario]
        proposed = scenario_frame[scenario_frame["method"] == PROPOSED].set_index("run_id")
        for baseline in BASELINES:
            comparator = scenario_frame[scenario_frame["method"] == baseline].set_index("run_id")
            joined = proposed[["completion_rate", "FND"]].join(
                comparator[["completion_rate", "FND"]],
                how="inner",
                lsuffix="_proposed",
                rsuffix="_baseline",
            )
            if len(joined) != len(proposed) or len(joined) != len(comparator):
                raise ValueError(f"unpaired report seeds for {scenario} / {baseline}")
            record = {"scenario": scenario, "baseline": baseline, "n": len(joined)}
            for metric in ["completion_rate", "FND"]:
                left = joined[f"{metric}_proposed"].to_numpy(dtype=float)
                right = joined[f"{metric}_baseline"].to_numpy(dtype=float)
                delta = left - right
                if np.allclose(delta, 0.0, rtol=0.0, atol=1e-14):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(
                        left,
                        right,
                        alternative=alternative,
                        zero_method="wilcox",
                        method="auto",
                    )
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                record[f"{metric}_mean_difference"] = float(np.mean(delta))
                record[f"{metric}_median_difference"] = float(np.median(delta))
                record[f"{metric}_w"] = statistic
                record[f"{metric}_p_raw"] = p_value
            records.append(record)

    tests = pd.DataFrame(records)
    for metric in ["completion_rate", "FND"]:
        tests[f"{metric}_p_holm"] = _holm_adjust(
            tests[f"{metric}_p_raw"].tolist()
        )
    return tests


def _metric_duplicates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    metrics = ["completion_rate", "FND", "total_energy", "energy_per_completed_task"]
    for phase, frame in frames.items():
        keys = ["scenario", "run_id"]
        if "harvest_source" in frame.columns:
            keys.insert(0, "harvest_source")
        rounded = frame.copy()
        rounded[metrics] = rounded[metrics].round(12)
        for group_key, group in rounded.groupby(keys, sort=False):
            duplicate_mask = group.duplicated(metrics, keep=False)
            duplicate_group = group.loc[duplicate_mask]
            if len(duplicate_group) < 2:
                continue
            for values, tied in duplicate_group.groupby(metrics, sort=False):
                if len(tied) < 2:
                    continue
                key_values = group_key if isinstance(group_key, tuple) else (group_key,)
                records.append(
                    {
                        "phase": phase,
                        "group": " / ".join(map(str, key_values)),
                        "methods": " | ".join(sorted(tied["method"].tolist())),
                        "completion_rate": values[0],
                        "FND": values[1],
                        "total_energy": values[2],
                        "energy_per_completed_task": values[3],
                    }
                )
    return pd.DataFrame(records)


def _assert_direct_trio_differentiated(frame: pd.DataFrame, phase: str) -> None:
    """Plan B: require differentiation on discriminating scenarios only.

    Under centered/moderate geometries the hard static quota often does not bind,
    so D3QN-EH and ECB-Static may still match; edge and critical must separate.
    """
    keys = ["scenario", "run_id"]
    if "harvest_source" in frame.columns:
        keys.insert(0, "harvest_source")
    metrics = [
        "completion_rate",
        "FND",
        "total_energy",
        "energy_per_completed_task",
    ]
    # Only enforce on scenarios where the paper claims separation.
    discriminating = {
        "edge_bs_high_load",
        "critical_scarcity",
        "credit_mismatch_edge",
    }
    direct = frame[
        frame["method"].isin(DIRECT_TRIO) & frame["scenario"].isin(discriminating)
    ]
    if direct.empty:
        return
    for group_key, group in direct.groupby(keys, sort=False):
        present = set(group["method"])
        if len(present) < 2:
            continue
        values = group.set_index("method")[metrics].to_numpy(dtype=float)
        if values.shape[0] >= 2 and np.allclose(
            values, values[0], rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"direct trio still display-equivalent in {phase}: {group_key}"
            )


def _find(stats: pd.DataFrame, scenario: str, method: str, **extra: str) -> pd.Series:
    mask = (stats["scenario"] == scenario) & (stats["method"] == method)
    for column, value in extra.items():
        mask &= stats[column] == value
    rows = stats.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {scenario}/{method}/{extra}, got {len(rows)}")
    return rows.iloc[0]


def _pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _centered_table(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Centered-BS comparison (10 evaluation seeds). $E_{\mathrm{sched}}$ contains task-routing and control traffic; $E_{\mathrm{phys}}$ additionally contains background sensing, idle-listening, and synchronization energy.\label{tab:centered_results}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabularx}{\textwidth}{lCCCCCCC}",
        r"\toprule",
        r"Method & Completion & FND & $E_{\mathrm{sched}}$ (J) & $E_{\mathrm{phys}}$ (J) & $E_{\mathrm{sched}}$/task ($\mu$J) & Control (\%) & $D_{\mathrm{budget}}$\\",
        r"\midrule",
    ]
    for method in DISPLAY_METHODS:
        row = _find(stats, "centered_bs", method)
        if method in {PROPOSED, "ECB-Static"}:
            budget = f"{row.budget_deviation_mean:.3f}"
        else:
            budget = "---"
        lines.append(
            " & ".join(
                [
                    METHOD_LABEL[method],
                    _pm(row.completion_mean, row.completion_std),
                    _pm(row.fnd_mean, row.fnd_std, 1),
                    f"{row.scheduling_energy_mean:.3f}",
                    f"{row.physical_energy_mean:.3f}",
                    f"{row.energy_per_task_mean * 1e6:.2f}",
                    f"{row.control_overhead_mean:.2f}",
                    budget,
                ]
            )
            + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    _write(output / "centered_results.tex", lines)


def _stress_table(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Method comparison in the two discriminating stress scenarios (10 evaluation seeds). Energy per task is based on scheduling energy.\label{tab:stress_results}}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{llcCCC}",
        r"\toprule",
        r"Scenario & Method & Completion & FND & $E_{\mathrm{sched}}$/task ($\mu$J) & Utility\\",
        r"\midrule",
    ]
    scenarios = ["edge_bs_high_load", "critical_scarcity"]
    for scenario_index, scenario in enumerate(scenarios):
        for method_index, method in enumerate(DISPLAY_METHODS):
            row = _find(stats, scenario, method)
            scenario_cell = SCENARIO_LABEL[scenario] if method_index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        scenario_cell,
                        METHOD_LABEL[method],
                        _pm(row.completion_mean, row.completion_std, 5),
                        _pm(row.fnd_mean, row.fnd_std, 1),
                        _pm(row.energy_per_task_mean * 1e6, row.energy_per_task_std * 1e6, 2),
                        f"{row.utility_mean:.3f}",
                    ]
                )
                + r"\\"
            )
        if scenario_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    _write(output / "stress_results.tex", lines)


def _significance_table(tests: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Holm-adjusted one-sided paired Wilcoxon $p$-values for the primary endpoint, task completion ($H_1$: ECB-HD-CR is higher). Adjustment covers all 30 stress-scenario/baseline comparisons.\label{tab:paired_significance}}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{lCCCCC}",
        r"\toprule",
        r"Scenario & D3QN-EH & ECB-Static & Two-TS-DRL & ICR-HHODRL & CHHFO\\",
        r"\midrule",
    ]
    for scenario in [
        "scarce_high_load",
        "small_battery_bursty",
        "high_pressure",
        "harvest_uncertain",
        "edge_bs_high_load",
        "critical_scarcity",
    ]:
        values = []
        for baseline in [
            "D3QN-EH",
            "ECB-Static",
            "Two-Timescale-DRL",
            "ICR-HHODRL",
            "CHHFO",
        ]:
            row = tests[(tests.scenario == scenario) & (tests.baseline == baseline)].iloc[0]
            values.append(_p(row.completion_rate_p_holm))
        lines.append(" & ".join([SCENARIO_LABEL[scenario], *values]) + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    _write(output / "paired_significance.tex", lines)


def _ablation_table(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Mechanism-ablation changes relative to ECB-HD-CR (10 report seeds). Positive energy values mean that the variant consumes more. Absolute ECB-HD-CR values are reported once in Table~\ref{tab:stress_results}.\label{tab:ablation_results}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabularx}{\textwidth}{llCCCCC}",
        r"\toprule",
        r"Scenario & Variant & $\Delta$ completion (pp) & $\Delta$ FND (rounds) & $\Delta E_{\mathrm{sched}}$ (\%) & $\Delta E_{\mathrm{sched}}$/task (\%) & $D_{\mathrm{budget}}$\\",
        r"\midrule",
    ]
    for scenario_index, scenario in enumerate(["edge_bs_high_load", "critical_scarcity"]):
        proposed = _find(stats, scenario, PROPOSED)
        for method_index, method in enumerate(ABLATION_ORDER[1:]):
            row = _find(stats, scenario, method)
            scenario_cell = SCENARIO_LABEL[scenario] if method_index == 0 else ""
            completion_delta = 100.0 * (
                row.completion_mean - proposed.completion_mean
            )
            fnd_delta = row.fnd_mean - proposed.fnd_mean
            energy_delta = 100.0 * (
                row.scheduling_energy_mean / proposed.scheduling_energy_mean - 1.0
            )
            task_energy_delta = 100.0 * (
                row.energy_per_task_mean / proposed.energy_per_task_mean - 1.0
            )
            lines.append(
                " & ".join(
                    [
                        scenario_cell,
                        METHOD_LABEL[method],
                        f"{completion_delta:+.2f}",
                        f"{fnd_delta:+.1f}",
                        f"{energy_delta:+.2f}",
                        f"{task_energy_delta:+.2f}",
                        f"{row.budget_deviation_mean:.3f}",
                    ]
                )
                + r"\\"
            )
        if scenario_index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    _write(output / "ablation_results.tex", lines)


def _regime_table(stats: pd.DataFrame, raw: pd.DataFrame, output: Path) -> None:
    switch_round = 150
    if "switch_round" in raw.columns and raw["switch_round"].notna().any():
        switch_round = int(raw["switch_round"].dropna().iloc[0])
    fnd_min = int(raw["FND"].min())
    fnd_max = int(raw["FND"].max())
    survived = int((raw["FND"] >= switch_round).sum())
    total = len(raw)
    lines = [
        r"\begin{table}[H]",
        r"\caption{Regime-switch ablation over an 800-round horizon (10 evaluation seeds). Pre/post completion isolate the switch; post-switch FND is the first further death after the switch (or the horizon).\label{tab:regime_results}}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{lCCCCCC}",
        r"\toprule",
        r"Variant & Completion & Pre & Post & FND & Post-FND & $D_{\mathrm{budget}}$\\",
        r"\midrule",
    ]
    for method in REGIME_ORDER:
        row = _find(stats, "shift_large_field_edge", method)
        pre = getattr(row, "pre_switch_completion_mean", float("nan"))
        post = getattr(row, "post_switch_completion_mean", float("nan"))
        post_fnd = getattr(row, "post_switch_fnd_mean", float("nan"))
        lines.append(
            " & ".join(
                [
                    METHOD_LABEL[method],
                    _pm(row.completion_mean, row.completion_std),
                    f"{pre:.3f}" if np.isfinite(pre) else "---",
                    f"{post:.3f}" if np.isfinite(post) else "---",
                    _pm(row.fnd_mean, row.fnd_std, 1),
                    f"{post_fnd:.1f}" if np.isfinite(post_fnd) else "---",
                    f"{row.budget_deviation_mean:.3f}",
                ]
            )
            + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            rf"\par\footnotesize The switch occurs at round {switch_round}. "
            rf"Overall FND ranges {fnd_min}--{fnd_max}; "
            rf"{survived}/{total} method/seed pairs survive past the switch. "
            r"Post-FND reports the first additional death after the switch.",
            r"\end{table}",
        ]
    )
    _write(output / "regime_results.tex", lines)


def _mismatch_table(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Credit-mismatch edge scenario (10 evaluation seeds). The fixed initial credit is deliberately underscaled; a load/harvest step occurs at round 200. Full ECB feedback can re-scale the reference, whereas static and harvest-anchored variants cannot.\label{tab:mismatch_results}}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{lCCCCC}",
        r"\toprule",
        r"Method & Completion & FND & $E_{\mathrm{sched}}$/task ($\mu$J) & Utility & $D_{\mathrm{budget}}$\\",
        r"\midrule",
    ]
    for method in MISMATCH_ORDER:
        row = _find(stats, "credit_mismatch_edge", method)
        budget = (
            f"{row.budget_deviation_mean:.3f}"
            if method
            in {
                PROPOSED,
                "ECB-HD-CR-NoFeedback",
                "ECB-Static",
                "ECB-HD-CR-NoCluster",
            }
            else "---"
        )
        lines.append(
            " & ".join(
                [
                    METHOD_LABEL[method],
                    _pm(row.completion_mean, row.completion_std),
                    _pm(row.fnd_mean, row.fnd_std, 1),
                    _pm(row.energy_per_task_mean * 1e6, row.energy_per_task_std * 1e6, 2),
                    f"{row.utility_mean:.3f}",
                    budget,
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    _write(output / "mismatch_results.tex", lines)


def _scale_table(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Scale check at $N=120$ (10 evaluation seeds).\label{tab:scale_results}}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{llCCCC}",
        r"\toprule",
        r"Scenario & Method & Completion & FND & $E_{\mathrm{sched}}$ (J) & $E_{\mathrm{sched}}$/task ($\mu$J)\\",
        r"\midrule",
    ]
    for scenario_index, scenario in enumerate(["scarce_high_load", "edge_bs_high_load"]):
        for method_index, method in enumerate(DISPLAY_METHODS):
            row = _find(stats, scenario, method)
            scenario_cell = SCENARIO_LABEL[scenario] if method_index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        scenario_cell,
                        METHOD_LABEL[method],
                        _pm(row.completion_mean, row.completion_std),
                        _pm(row.fnd_mean, row.fnd_std, 1),
                        _pm(row.scheduling_energy_mean, row.scheduling_energy_std),
                        _pm(row.energy_per_task_mean * 1e6, row.energy_per_task_std * 1e6, 2),
                    ]
                )
                + r"\\"
            )
        if scenario_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    _write(output / "scale_results.tex", lines)


def _sensitivity_table(stats: pd.DataFrame, output: Path) -> None:
    critical = stats[stats["base_scenario"] == "critical_scarcity"].sort_values("gain")
    lines = [
        r"\begin{table}[H]",
        r"\caption{Sensitivity to the shared rate-adaptation gain in critical scarcity (evaluation-seed indices 0--4, a subset of the ten indices in Table~\ref{tab:stress_results}).\label{tab:sensitivity_results}}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{cCCCC}",
        r"\toprule",
        r"$\alpha_\lambda=\gamma_\mu$ & Completion & FND & $E_{\mathrm{sched}}$ (J) & $D_{\mathrm{budget}}$\\",
        r"\midrule",
    ]
    for row in critical.itertuples(index=False):
        lines.append(
            f"{row.gain:.1f} & {_pm(row.completion_mean, row.completion_std)} & "
            f"{_pm(row.fnd_mean, row.fnd_std, 1)} & {row.energy_mean:.3f} & "
            f"{row.budget_deviation_mean:.3f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    _write(output / "sensitivity_results.tex", lines)


def _nsrdb_table(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{ECB-HD-CR under measured-shape NSRDB input (10 evaluation seeds). Deltas are NSRDB minus the synthetic result generated from the identical evaluation seed; synthetic absolute values are not repeated.\label{tab:nsrdb_results}}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{lCCCCCCC}",
        r"\toprule",
        r"Scenario & NSRDB completion & NSRDB FND & $E_{\mathrm{sched}}$ (J) & $E_{\mathrm{phys}}$ (J) & $D_{\mathrm{budget}}$ & $\Delta$ completion (pp) & $\Delta$ FND\\",
        r"\midrule",
    ]
    for scenario in ["edge_bs_high_load", "critical_scarcity"]:
        synthetic = _find(
            stats, scenario, PROPOSED, harvest_source="synthetic"
        )
        nsrdb = _find(stats, scenario, PROPOSED, harvest_source="nsrdb")
        lines.append(
            " & ".join(
                [
                    SCENARIO_LABEL[scenario],
                    _pm(nsrdb.completion_mean, nsrdb.completion_std),
                    _pm(nsrdb.fnd_mean, nsrdb.fnd_std, 1),
                    _pm(nsrdb.scheduling_energy_mean, nsrdb.scheduling_energy_std),
                    f"{nsrdb.physical_energy_mean:.3f}",
                    f"{nsrdb.budget_deviation_mean:.3f}",
                    f"{100.0 * (nsrdb.completion_mean - synthetic.completion_mean):+.3f}",
                    f"{nsrdb.fnd_mean - synthetic.fnd_mean:+.1f}",
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    _write(output / "nsrdb_results.tex", lines)


def _plot_stress(frame: pd.DataFrame, figure_dir: Path) -> None:
    # Compact display for the trade-off figure (proposed + three direct + ICR + CHHFO is crowded).
    plot_methods = [PROPOSED, "D3QN-EH", "ECB-Static", "Two-Timescale-DRL", "ICR-HHODRL", "CHHFO"]
    short_labels = ["ECB", "D3QN", "Static", "2TS", "ICR", "CHHFO"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
    panels = [
        ("critical_scarcity", "completion_rate", "Completion rate", (0.65, 1.015)),
        ("critical_scarcity", "FND", "First-node-death round", (0, 525)),
        ("edge_bs_high_load", "energy_per_completed_task", "Scheduling energy/task ($\\mu$J)", None),
    ]
    for axis, (scenario, metric, ylabel, ylim) in zip(axes, panels):
        data = []
        for method in plot_methods:
            values = frame[(frame.scenario == scenario) & (frame.method == method)][metric]
            if metric == "energy_per_completed_task":
                values = values * 1e6
            data.append(values.to_numpy(dtype=float))
        boxes = axis.boxplot(
            data,
            tick_labels=short_labels,
            widths=0.62,
            patch_artist=True,
            medianprops={"color": "#202020", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
            flierprops={"markersize": 3, "alpha": 0.6},
        )
        for patch, method in zip(boxes["boxes"], plot_methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.82)
            patch.set_edgecolor("#303030")
        axis.set_ylabel(ylabel, fontsize=9)
        axis.tick_params(axis="x", labelrotation=28, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
        if ylim:
            axis.set_ylim(*ylim)
    axes[0].set_title("(a) Critical scarcity", fontsize=10)
    axes[1].set_title("(b) Critical scarcity", fontsize=10)
    axes[2].set_title("(c) Edge-BS high load", fontsize=10)
    fig.savefig(figure_dir / "stress_tradeoffs.svg", bbox_inches="tight")
    import cairosvg
    cairosvg.svg2pdf(url=str(figure_dir / "stress_tradeoffs.svg"),
                    write_to=str(figure_dir / "stress_tradeoffs.pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", "--thesis-dir", dest="thesis_dir", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    results = args.results.resolve()
    thesis = args.thesis_dir.resolve()
    generated = thesis / "generated"
    figures = thesis / "Figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    expected_pickle_count = 950  # Plan B: +60 mismatch + 84 scale (3->10 seeds)
    pickle_paths = list(((args.raw_results or results) / "runs").rglob("*.pkl"))
    pickle_count = len(pickle_paths)
    if pickle_count != expected_pickle_count:
        raise ValueError(
            f"expected {expected_pickle_count} raw run files, found {pickle_count} in {results}"
        )

    versions = set()
    for path in pickle_paths:
        with path.open("rb") as handle:
            record = pickle.load(handle)
        versions.add(record.get("config_snapshot", {}).get("experiment_version"))
    if versions != {"thesis_final_v7"}:
        raise ValueError(f"mixed or missing experiment versions: {versions}")

    frames = {
        phase: _load_phase(results, phase)
        for phase in [
            "centered",
            "stress",
            "ablation",
            "regime",
            "mismatch",
            "nsrdb",
            "sensitivity",
            "scale",
        ]
    }
    expected_rows = {
        "centered": 60,
        "stress": 360,
        "ablation": 150,
        "regime": 60,
        "mismatch": 60,
        "nsrdb": 240,
        "sensitivity": 50,
        "scale": 120,  # 2 scenarios × 6 methods × 10 seeds
    }
    for phase, expected in expected_rows.items():
        if len(frames[phase]) != expected:
            raise ValueError(
                f"expected {expected} exported rows for {phase}, found {len(frames[phase])}"
            )
        frame = frames[phase]
        if not np.allclose(
            frame["total_physical_energy"],
            frame["total_energy"] + frame["background_energy"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"physical-energy identity failed in phase {phase}")
        expected_completion = frame["completed_tasks"] / frame["total_tasks"].clip(lower=1)
        if not np.allclose(
            frame["completion_rate"], expected_completion, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"completion-rate identity failed in phase {phase}")

    # Common environment streams imply identical task counts for methods that
    # share a scenario and report seed.
    for phase in ["centered", "stress", "scale", "mismatch"]:
        mismatch = frames[phase].groupby(["scenario", "run_id"])["total_tasks"].nunique()
        if (mismatch != 1).any():
            raise ValueError(f"task-stream mismatch in phase {phase}")

    for phase in ["centered", "stress", "nsrdb", "scale"]:
        _assert_direct_trio_differentiated(frames[phase], phase)

    stats = {
        "centered": _aggregate(frames["centered"]),
        "stress": _aggregate(frames["stress"]),
        "ablation": _aggregate(frames["ablation"]),
        "regime": _aggregate(frames["regime"]),
        "mismatch": _aggregate(frames["mismatch"]),
        "nsrdb": _aggregate(frames["nsrdb"], ["harvest_source"]),
        "scale": _aggregate(frames["scale"]),
    }
    for phase, frame in stats.items():
        frame.to_csv(generated / f"paper_stats_{phase}.csv", index=False)

    stress_tests = _paired_tests(frames["stress"], alternative="greater")
    centered_tests = _paired_tests(frames["centered"], alternative="two-sided")
    stress_tests.to_csv(generated / "paired_tests_stress.csv", index=False)
    centered_tests.to_csv(generated / "paired_tests_centered.csv", index=False)

    duplicate_metrics = _metric_duplicates(frames)
    duplicate_metrics.to_csv(generated / "duplicate_metric_groups.csv", index=False)

    _centered_table(stats["centered"], generated)
    _stress_table(stats["stress"], generated)
    _significance_table(stress_tests, generated)
    _ablation_table(stats["ablation"], generated)
    _regime_table(stats["regime"], frames["regime"], generated)
    _mismatch_table(stats["mismatch"], generated)
    sensitivity_stats = pd.read_csv(results / "sensitivity" / "aggregated_stats.csv")
    _sensitivity_table(sensitivity_stats, generated)
    _nsrdb_table(stats["nsrdb"], generated)
    _scale_table(stats["scale"], generated)

    if not args.no_figures:
        _plot_stress(frames["stress"], figures)

    csv_paths = sorted(results.glob("*/summary.csv"))
    manifest = {
        "result_root": results.name,
        "raw_pickle_count": pickle_count,
        "experiment_versions": sorted(versions),
        "summary_sha256": {str(path.relative_to(results)): _sha256(path) for path in csv_paths},
        "generated_files": sorted(path.name for path in generated.iterdir()),
    }
    (generated / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Generated paper artifacts from {results} into {thesis}")


if __name__ == "__main__":
    main()
