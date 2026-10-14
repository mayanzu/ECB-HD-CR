"""Run one versioned, internally consistent experiment set for the thesis.

Plan B (thesis_final_v5):
- differentiated direct baselines (hard budgets)
- regime-switch with post-switch survival
- scale check with 10 seeds
- credit_mismatch scenario for ECB feedback physical gains
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pickle
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_specs import (
    ABLATION_METHODS,
    BASELINE_METHODS,
    BASE_CONFIG,
    MISMATCH_METHODS,
    MISMATCH_SCENARIOS,
    REGIME_SHIFT_METHODS,
    REGIME_SHIFT_SCENARIOS,
    STRESS_SCENARIOS,
    scenario_map,
)
from experiments.result_utils import aggregate, relative_gains, result_to_row
from experiments.run_main_experiments import run_single_experiment


REPORT_SEED_OFFSET = 720000
REPORT_SEEDS = 10
SCALE_SEEDS = 10
OUTPUT = PROJECT_ROOT / "results" / "thesis_final_v7"
EXPERIMENT_VERSION = "thesis_final_v7"


def _result_path(job: dict) -> Path:
    return (
        Path(job["output_dir"])
        / f"{job['method']}_N{job['nodes']}_run{job['run_id']:02d}.pkl"
    )


def _execute(job: dict) -> dict:
    """Worker entry point with resumable per-run pickle output."""
    import torch

    torch.set_num_threads(1)
    path = _result_path(job)
    if job["resume"] and path.exists():
        with path.open("rb") as handle:
            result = pickle.load(handle)
        expected_version = job["config"].get("experiment_version")
        actual_version = result.get("config_snapshot", {}).get(
            "experiment_version"
        )
        identity = (result.get("method"), result.get("N"), result.get("run_id"))
        expected_identity = (job["method"], job["nodes"], job["run_id"])
        if actual_version != expected_version or identity != expected_identity:
            raise RuntimeError(
                f"resume file does not match job: {path}; "
                f"version {actual_version!r} vs {expected_version!r}, "
                f"identity {identity!r} vs {expected_identity!r}"
            )
    else:
        result = run_single_experiment(
            (
                job["method"],
                job["nodes"],
                job["run_id"],
                job["config"],
                job["output_dir"],
            )
        )
    return {"job": job, "result": result}


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_execute, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            job = record["job"]
            print(
                f"[{index:4d}/{len(jobs)}] {job['phase']} / "
                f"{job['scenario']} / {job['method']} / seed {job['run_id']}",
                flush=True,
            )
    return records


def _job(
    phase: str,
    scenario: str,
    method: str,
    run_id: int,
    config: dict,
    nodes: int,
    resume: bool,
) -> dict:
    output_dir = OUTPUT / "runs" / phase / scenario
    return {
        "phase": phase,
        "scenario": scenario,
        "method": method,
        "run_id": run_id,
        "config": config,
        "nodes": nodes,
        "output_dir": str(output_dir),
        "resume": resume,
    }


def _base_config() -> dict:
    config = dict(BASE_CONFIG)
    config["seed_offset"] = REPORT_SEED_OFFSET
    config["experiment_version"] = EXPERIMENT_VERSION
    return config


def _rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        job = record["job"]
        row = result_to_row(
            record["result"], job["scenario"], include_extended=True
        )
        row["phase"] = job["phase"]
        rows.append(row)
    return rows


def _write_standard(rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(
        ["scenario", "method", "run_id"]
    )
    summary.to_csv(directory / "summary.csv", index=False)
    stats = aggregate(summary, include_extended=True)
    stats.to_csv(directory / "aggregated_stats.csv", index=False)
    relative_gains(stats).to_csv(directory / "relative_gains.csv", index=False)


def _build_jobs(resume: bool) -> tuple[list[dict], dict[str, list[dict]]]:
    jobs = []
    groups: dict[str, list[dict]] = {}
    stress_map = scenario_map(STRESS_SCENARIOS)

    centered = _base_config()
    centered.update(
        {
            "E_init": 0.45,
            "B_max": 3.0,
            "base_energy_per_round": 0.00006,
            "energy_scale_factor": 0.00000008,
            "arrival_rate": 1.25,
        }
    )
    groups["centered"] = [
        _job("centered", "centered_bs", method, seed, centered, 80, resume)
        for method in BASELINE_METHODS
        for seed in range(REPORT_SEEDS)
    ]

    groups["stress"] = []
    for scenario in STRESS_SCENARIOS:
        config = _base_config()
        config.update(scenario["overrides"])
        groups["stress"].extend(
            _job(
                "stress",
                scenario["name"],
                method,
                seed,
                config,
                80,
                resume,
            )
            for method in BASELINE_METHODS
            for seed in range(REPORT_SEEDS)
        )

    groups["ablation"] = []
    for scenario_name in [
        "harvest_uncertain",
        "edge_bs_high_load",
        "critical_scarcity",
    ]:
        config = _base_config()
        config.update(stress_map[scenario_name]["overrides"])
        groups["ablation"].extend(
            _job(
                "ablation",
                scenario_name,
                method,
                seed,
                config,
                80,
                resume,
            )
            for method in ABLATION_METHODS[1:]
            for seed in range(REPORT_SEEDS)
        )

    regime = _base_config()
    regime["total_rounds"] = 800
    regime.update(REGIME_SHIFT_SCENARIOS[0]["overrides"])
    groups["regime"] = [
        _job(
            "regime",
            REGIME_SHIFT_SCENARIOS[0]["name"],
            method,
            seed,
            regime,
            80,
            resume,
        )
        for method in REGIME_SHIFT_METHODS[:6]
        for seed in range(REPORT_SEEDS)
    ]

    # Plan B: credit-mismatch / load-step scenario.
    groups["mismatch"] = []
    for scenario in MISMATCH_SCENARIOS:
        config = _base_config()
        config.update(scenario["overrides"])
        groups["mismatch"].extend(
            _job(
                "mismatch",
                scenario["name"],
                method,
                seed,
                config,
                80,
                resume,
            )
            for method in MISMATCH_METHODS
            for seed in range(REPORT_SEEDS)
        )

    groups["nsrdb"] = []
    nsrdb_csv = str(PROJECT_ROOT / "data" / "nsrdb_hefei.csv")
    for scenario_name in ["edge_bs_high_load", "critical_scarcity"]:
        config = _base_config()
        config.update(stress_map[scenario_name]["overrides"])
        config.update({"harvest_source": "nsrdb", "nsrdb_csv": nsrdb_csv})
        groups["nsrdb"].extend(
            _job(
                "nsrdb",
                scenario_name,
                method,
                seed,
                config,
                80,
                resume,
            )
            for method in BASELINE_METHODS
            for seed in range(REPORT_SEEDS)
        )

    groups["sensitivity"] = []
    for scenario_name in ["high_pressure", "critical_scarcity"]:
        for gain in [0.1, 0.2, 0.3, 0.5, 0.7]:
            config = _base_config()
            config.update(stress_map[scenario_name]["overrides"])
            config.update({"alpha_lambda": gain, "gamma_mu": gain})
            tagged = f"{scenario_name}__gain_{gain:.1f}"
            groups["sensitivity"].extend(
                _job(
                    "sensitivity",
                    tagged,
                    "ECB-HD-CR",
                    seed,
                    config,
                    80,
                    resume,
                )
                for seed in range(5)
            )

    groups["scale"] = []
    for scenario_name in ["scarce_high_load", "edge_bs_high_load"]:
        config = _base_config()
        config.update(stress_map[scenario_name]["overrides"])
        groups["scale"].extend(
            _job(
                "scale",
                scenario_name,
                method,
                seed,
                config,
                120,
                resume,
            )
            for method in BASELINE_METHODS
            for seed in range(SCALE_SEEDS)
        )

    for phase_jobs in groups.values():
        jobs.extend(phase_jobs)
    return jobs, groups


def _export(records: list[dict]) -> None:
    rows = _rows(records)
    by_phase = {
        phase: [row for row in rows if row["phase"] == phase]
        for phase in {row["phase"] for row in rows}
    }

    for phase in ["centered", "stress", "regime", "scale", "mismatch"]:
        if phase in by_phase:
            _write_standard(by_phase[phase], OUTPUT / phase)

    # Reuse the exact proposed-method stress rows in the ablation table.
    ablation_rows = list(by_phase.get("ablation", []))
    target = {"harvest_uncertain", "edge_bs_high_load", "critical_scarcity"}
    ablation_rows.extend(
        row
        for row in by_phase.get("stress", [])
        if row["method"] == "ECB-HD-CR" and row["scenario"] in target
    )
    _write_standard(ablation_rows, OUTPUT / "ablation")

    # Reuse the same stress rows as the synthetic side of the trace comparison.
    nsrdb_rows = []
    for row in by_phase.get("nsrdb", []):
        row = dict(row)
        row["harvest_source"] = "nsrdb"
        nsrdb_rows.append(row)
    for row in by_phase.get("stress", []):
        if row["scenario"] in {"edge_bs_high_load", "critical_scarcity"}:
            row = dict(row)
            row["harvest_source"] = "synthetic"
            nsrdb_rows.append(row)
    nsrdb_dir = OUTPUT / "nsrdb"
    nsrdb_dir.mkdir(parents=True, exist_ok=True)
    nsrdb_summary = pd.DataFrame(nsrdb_rows).sort_values(
        ["harvest_source", "scenario", "method", "run_id"]
    )
    nsrdb_summary.to_csv(nsrdb_dir / "summary.csv", index=False)
    stats_parts = []
    gain_parts = []
    for source, part in nsrdb_summary.groupby("harvest_source", sort=False):
        stats = aggregate(part, include_extended=True)
        stats.insert(0, "harvest_source", source)
        stats_parts.append(stats)
        gains = relative_gains(stats.drop(columns=["harvest_source"]))
        gains.insert(0, "harvest_source", source)
        gain_parts.append(gains)
    if stats_parts:
        pd.concat(stats_parts, ignore_index=True).to_csv(
            nsrdb_dir / "aggregated_stats.csv", index=False
        )
        pd.concat(gain_parts, ignore_index=True).to_csv(
            nsrdb_dir / "relative_gains.csv", index=False
        )

    sensitivity_dir = OUTPUT / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    if by_phase.get("sensitivity"):
        sensitivity = pd.DataFrame(by_phase["sensitivity"])
        sensitivity[["base_scenario", "gain"]] = sensitivity["scenario"].str.extract(
            r"(.+)__gain_([0-9.]+)"
        )
        sensitivity["gain"] = sensitivity["gain"].astype(float)
        sensitivity.to_csv(sensitivity_dir / "summary.csv", index=False)
        sensitivity.groupby(["base_scenario", "gain"]).agg(
            completion_mean=("completion_rate", "mean"),
            completion_std=("completion_rate", "std"),
            fnd_mean=("FND", "mean"),
            fnd_std=("FND", "std"),
            energy_mean=("total_energy", "mean"),
            energy_per_task_mean=("energy_per_completed_task", "mean"),
            budget_deviation_mean=("budget_deviation", "mean"),
        ).reset_index().to_csv(
            sensitivity_dir / "aggregated_stats.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="*",
        choices=[
            "centered",
            "stress",
            "ablation",
            "regime",
            "mismatch",
            "nsrdb",
            "sensitivity",
            "scale",
        ],
    )
    args = parser.parse_args()

    all_jobs, groups = _build_jobs(args.resume)
    if args.phases:
        selected = set(args.phases)
        jobs = [job for job in all_jobs if job["phase"] in selected]
    else:
        jobs = all_jobs
    print(
        f"Plan B runner: {len(jobs)} jobs, version={EXPERIMENT_VERSION}, "
        f"output={OUTPUT}",
        flush=True,
    )
    records = _run_jobs(jobs, args.workers)

    # A complete export needs all phases. With --resume, loading all records is
    # cheap and guarantees that derived tables are rebuilt from one result set.
    if args.phases:
        remaining = [
            job
            for phase, phase_jobs in groups.items()
            if phase not in set(args.phases)
            for job in phase_jobs
        ]
        if all(_result_path({**job, "resume": True}).exists() for job in remaining):
            records.extend(
                _run_jobs([{**job, "resume": True} for job in remaining], args.workers)
            )
        else:
            print(
                "Selected phases complete; full export deferred until all phases exist."
            )
            # Still export selected phases if present.
            _export(records)
            return
    _export(records)
    print(f"Final thesis results written to {OUTPUT}")


if __name__ == "__main__":
    main()
