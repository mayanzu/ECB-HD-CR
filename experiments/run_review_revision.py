"""Rerun the shared BS protocol on five prespecified paper settings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_main_experiments import run_single_experiment
from experiments.run_thesis_final import _build_jobs


def execute(job):
    import os
    os.environ["ECB_DEVICE"] = job[3].get("device", "cpu")
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = run_single_experiment(job)
    return result["method"], result["run_id"], time.perf_counter() - started


def build_jobs(seeds=10, smoke=False):
    original, _ = _build_jobs(False)
    wanted = {( "centered", "centered_bs", 80),
              ("stress", "edge_bs_high_load", 80),
              ("stress", "critical_scarcity", 80),
              ("scale", "edge_bs_high_load", 120),
              ("scale", "scarce_high_load", 120)}
    jobs = []
    for old in original:
        key = (old["phase"], old["scenario"], old["nodes"])
        if key not in wanted or old["run_id"] >= seeds:
            continue
        config = dict(old["config"])
        config.update(deployment_protocol="central_bs", record_diagnostics=True,
                      experiment_version="review_revision_v1", device="cpu")
        if smoke:
            config["total_rounds"] = 60
        out = ROOT/"results"/("review_revision_smoke" if smoke else "review_revision_v1")/"runs"/f"{old['scenario']}_N{old['nodes']}"
        jobs.append((old["method"], old["nodes"], old["run_id"], config, str(out)))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Seeds per cell (default: 10 full, 1 smoke)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (1 if args.smoke else 10)
    if args.workers < 1 or not 1 <= seeds <= 10:
        parser.error("--workers must be positive and --seeds must be between 1 and 10")
    jobs = build_jobs(seeds, args.smoke)
    if args.resume:
        pending = []
        for job in jobs:
            method, n, seed, config, out = job
            p = Path(out)/f"{method}_N{n}_run{seed:02d}.pkl"
            if p.exists():
                with p.open("rb") as f: record = pickle.load(f)
                if record["config_snapshot"] != config:
                    raise ValueError(f"resume config mismatch: {p}")
            else:
                pending.append(job)
        jobs = pending
    print(f"Jobs: {len(jobs)}, workers: {args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            method, seed, seconds = future.result()
            print(f"{i}/{len(jobs)} {method}, seed {seed}, {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
