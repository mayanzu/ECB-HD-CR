"""Recompute paper statistics from supplied evidence; no TeX installation."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def compare_csv(reference, regenerated):
    import numpy as np
    import pandas as pd
    a, b = pd.read_csv(reference), pd.read_csv(regenerated)
    if set(a.columns) != set(b.columns) or len(a) != len(b):
        raise ValueError('CSV shape/column mismatch: ' + reference.name)
    b = b[a.columns]
    keys = [key for key in ['phase','harvest_source','base_scenario','scenario','N','method','run_id','round','gain']
            if key in a.columns]
    if keys:
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
    for column in a:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            np.testing.assert_allclose(a[column], b[column], rtol=1e-10, atol=1e-12, equal_nan=True,
                                       err_msg=f'{reference.name}: {column}')
        elif not a[column].fillna('').astype(str).equals(b[column].fillna('').astype(str)):
            raise ValueError(f'CSV value mismatch: {reference.name}: {column}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    os.environ.setdefault('ECB_DEVICE', 'cpu')
    from verify_package import verify
    from restore_results import restore
    verify()
    restore()
    run('-m', 'unittest', 'discover', '-s', 'tests', '-v')

    from experiments import run_thesis_final as legacy
    jobs, _ = legacy._build_jobs(False)
    records = []
    for job in jobs:
        with legacy._result_path(job).open('rb') as handle:
            record = pickle.load(handle)
        if (record['method'], record['N'], record['run_id']) != (job['method'],job['nodes'],job['run_id']):
            raise ValueError('Legacy run identity mismatch')
        records.append({'job': job, 'result': record})
    legacy.OUTPUT = ROOT/'artifacts/legacy_exports'
    legacy._export(records)
    extra = ['--no-figures'] if args.no_figures else []
    run('analysis/generate_thesis_artifacts.py', '--results', legacy.OUTPUT,
        '--raw-results', ROOT/'results/thesis_final_v7', '--output', ROOT/'artifacts/legacy', *extra)

    from analysis.export_review_revision import legacy_audit, deployment_export
    deployed = ROOT/'artifacts/deployment'
    deployed.mkdir(parents=True, exist_ok=True)
    audit = legacy_audit(deployed, args.workers)
    if len(audit) != 110:
        raise ValueError('Expected 110 original proposed-method bound records')
    deployment_export(deployed, audit)
    compared = 0
    for version, destination in [('thesis_final_v7',legacy.OUTPUT), ('review_revision_v1',deployed)]:
        for reference in sorted((ROOT/'results'/version).rglob('*.csv')):
            generated = destination/reference.relative_to(ROOT/'results'/version)
            compare_csv(reference, generated)
            compared += 1
    if not args.no_figures:
        run('analysis/draw_review_figures.py', '--results', deployed, '--output', deployed)
    report = dict(raw_runs=1250, legacy_runs=950, deployment_runs=300, invariant_tests=18,
                  original_bound_audits=len(audit), compared_reference_csvs=compared,
                  numerical_reference_match=True, figures_generated=not args.no_figures,
                  comparison_rtol=1e-10, comparison_atol=1e-12,
                  outputs='artifacts/', published_reference_files_unchanged=True)
    (ROOT/'artifacts/REPRODUCTION_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
