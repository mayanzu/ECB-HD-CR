"""Shared result conversion and aggregation helpers for experiment runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_specs import PROPOSED


GAIN_COLUMNS = [
    "scenario",
    "baseline",
    "completion_relative_gain_pct",
    "energy_reduction_pct",
    "energy_per_task_reduction_pct",
    "fnd_gain_pct",
    "utility_gain_pct",
    "budget_deviation_reduction_pct",
]


def ci95(series: pd.Series) -> float:
    count = series.count()
    if count <= 1:
        return 0.0
    return 1.96 * float(series.std(ddof=1)) / np.sqrt(count)


def result_to_row(result: dict, scenario_name: str, include_extended: bool = False) -> dict:
    """Convert a per-run result dict into a stable CSV row."""
    total_energy = result["total_energy"]
    completed_tasks = max(1, result["completed_tasks"])
    row = {
        "scenario": scenario_name,
        "method": result["method"],
        "run_id": result["run_id"],
        "FND": result["fnd"],
        "HND": result["hnd"],
        "LND": result["lnd"],
        "completion_rate": result["completion_rate"],
        "completed_tasks": result["completed_tasks"],
        "total_tasks": result["total_tasks"],
        "total_energy": total_energy,
        "background_energy": result.get("background_energy", 0.0),
        "total_physical_energy": result.get(
            "total_physical_energy",
            total_energy + result.get("background_energy", 0.0),
        ),
        "energy_per_completed_task": total_energy / completed_tasks,
        "budget_deviation": result.get("budget_deviation", 0.0),
    }
    if include_extended:
        alive_node_rounds = max(1, result.get("alive_node_rounds", 1))
        horizon = max(1, len(result.get("round", [])))
        row.update(
            {
                "energy_per_alive_node_round": total_energy / alive_node_rounds,
                "control_energy": result.get("control_energy", 0.0),
                "control_overhead_pct": 100.0
                * result.get("control_energy", 0.0)
                / max(1e-12, total_energy),
                "utility": result["completion_rate"] * result["fnd"] / horizon,
                "task_count": max(1, result["total_tasks"]),
                "pre_switch_completion": result.get("pre_switch_completion", 0.0),
                "post_switch_completion": result.get("post_switch_completion", 0.0),
                "post_switch_fnd": result.get("post_switch_fnd", result["fnd"]),
                "switch_round": result.get("switch_round"),
            }
        )
    return row


def aggregate(summary: pd.DataFrame, include_extended: bool = False) -> pd.DataFrame:
    agg_spec = {
        "completion_mean": ("completion_rate", "mean"),
        "completion_std": ("completion_rate", "std"),
        "completion_ci95": ("completion_rate", ci95),
        "fnd_mean": ("FND", "mean"),
        "fnd_std": ("FND", "std"),
        "energy_mean": ("total_energy", "mean"),
        "energy_std": ("total_energy", "std"),
        "energy_per_task_mean": ("energy_per_completed_task", "mean"),
        "energy_per_task_std": ("energy_per_completed_task", "std"),
    }
    if include_extended:
        agg_spec.update(
            {
                "utility_mean": ("utility", "mean"),
                "utility_std": ("utility", "std"),
                "energy_per_alive_node_round_mean": (
                    "energy_per_alive_node_round",
                    "mean",
                ),
                "control_overhead_pct_mean": ("control_overhead_pct", "mean"),
                "budget_deviation_mean": ("budget_deviation", "mean"),
                "pre_switch_completion_mean": ("pre_switch_completion", "mean"),
                "post_switch_completion_mean": ("post_switch_completion", "mean"),
                "post_switch_fnd_mean": ("post_switch_fnd", "mean"),
            }
        )
    return summary.groupby(["scenario", "method"], sort=False).agg(**agg_spec).reset_index()


def relative_gains(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in agg.groupby("scenario", sort=False):
        proposed_rows = group[group["method"] == PROPOSED]
        if proposed_rows.empty:
            continue
        proposed = proposed_rows.iloc[0]
        for _, row in group[group["method"] != PROPOSED].iterrows():
            gain = {
                "scenario": scenario,
                "baseline": row["method"],
                "completion_relative_gain_pct": (
                    proposed["completion_mean"] / row["completion_mean"] - 1
                )
                * 100
                if row["completion_mean"] > 0
                else None,
                "energy_reduction_pct": (
                    1 - proposed["energy_mean"] / row["energy_mean"]
                )
                * 100
                if row["energy_mean"] > 0
                else None,
                "energy_per_task_reduction_pct": (
                    1
                    - proposed["energy_per_task_mean"]
                    / row["energy_per_task_mean"]
                )
                * 100
                if row["energy_per_task_mean"] > 0
                else None,
                "fnd_gain_pct": (proposed["fnd_mean"] / row["fnd_mean"] - 1) * 100
                if row["fnd_mean"] > 0
                else None,
            }
            if "utility_mean" in row and "utility_mean" in proposed:
                gain["utility_gain_pct"] = (
                    proposed["utility_mean"] / row["utility_mean"] - 1
                ) * 100 if row["utility_mean"] > 0 else None
            if "budget_deviation_mean" in row and "budget_deviation_mean" in proposed:
                gain["budget_deviation_reduction_pct"] = (
                    1
                    - proposed["budget_deviation_mean"]
                    / row["budget_deviation_mean"]
                ) * 100 if row["budget_deviation_mean"] > 0 else None
            rows.append(gain)
    return pd.DataFrame(rows, columns=GAIN_COLUMNS)


def write_outputs(rows: list[dict], output_dir: Path, include_extended: bool = False) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    agg = aggregate(summary, include_extended=include_extended)
    agg.to_csv(output_dir / "aggregated_stats.csv", index=False)
    relative_gains(agg).to_csv(output_dir / "relative_gains.csv", index=False)
