"""Audit original runs, then export all newly executed deployment results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.optimality_bounds import objective, trace_upper_bound, WEIGHTS, ENERGY_REFERENCE_J
from scipy.stats import t as student_t


def audit_bound(path):
    with Path(path).open('rb') as f:
        result = pickle.load(f)
    parts = Path(path).relative_to(ROOT/'results/thesis_final_v7/runs').parts
    scenario = parts[1]
    upper = trace_upper_bound(result)
    achieved = objective(result)
    if achieved > upper + 1e-10:
        raise ValueError(f'Upper-bound violation: {path}')
    return dict(phase=parts[0], scenario=scenario, N=result['N'], run_id=result['run_id'],
                objective=achieved, upper_bound=upper, gap_upper=upper-achieved)


def legacy_audit(output, workers):
    paths = []
    for phase in ['centered', 'stress', 'regime', 'mismatch', 'scale']:
        paths.extend((ROOT/'results/thesis_final_v7/runs'/phase).rglob('ECB-HD-CR_N*.pkl'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_bound, [str(p) for p in paths]))
    frame = pd.DataFrame(rows).sort_values(['phase', 'scenario', 'N', 'run_id'])
    frame.to_csv(output/'legacy_bound_audit.csv', index=False)
    print(f'Audited {len(frame)} original proposed-method runs.', flush=True)
    return frame


def deployment_export(output, legacy):
    paths = list((ROOT/'results/review_revision_v1/runs').rglob('*.pkl'))
    if len(paths) != 300:
        raise ValueError(f'Expected 300 complete deployment runs; found {len(paths)}')
    bound_lookup = legacy.set_index(['scenario', 'N', 'run_id']).upper_bound.to_dict()
    rows = []; diagnostics = []
    for p in paths:
        with p.open('rb') as f:
            d = pickle.load(f)
        c = d['config_snapshot']
        if c.get('experiment_version') != 'review_revision_v1' or c.get('deployment_protocol') != 'central_bs':
            raise ValueError(f'Wrong revision record: {p}')
        scenario = p.parent.name.rsplit('_N', 1)[0]
        row = dict(scenario=scenario, N=d['N'], run_id=d['run_id'], method=d['method'],
                   completion=d['completion_rate'], FND=d['fnd'], scheduling_energy=d['total_energy'],
                   physical_energy=d['total_physical_energy'], delivered=d['completed_tasks'],
                   offered=d['total_tasks'], energy_per_task_uJ=1e6*d['total_energy']/max(d['completed_tasks'],1),
                   ecb_control_energy=d['control_energy'], deployment_energy=d['deployment_energy'],
                   control_failed_transactions=d['control_failed_transactions'], objective=objective(d))
        for key, value in d['deployment_energy_components'].items():
            row[f'{key}_energy'] = value
        row['all_control_share_pct'] = 100*(row['ecb_control_energy']+row['deployment_energy'])/max(row['scheduling_energy'],1e-15)
        if d['method'] == 'ECB-HD-CR':
            row['upper_bound'] = bound_lookup[(scenario, d['N'], d['run_id'])]
            row['gap_upper'] = row['upper_bound']-row['objective']
            if row['gap_upper'] < -1e-10:
                raise ValueError('Bound violation after deployment costs')
            for item in d['diagnostics']:
                entry = dict(item, scenario=scenario, N=d['N'], run_id=d['run_id'])
                entry['alive_after_round'] = d['alive_count'][item['round']]
                diagnostics.append(entry)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(['scenario','N','method','run_id'])
    if (frame.groupby(['scenario','N','run_id']).offered.nunique() != 1).any():
        raise ValueError('Methods do not share offered tasks')
    counts = frame.groupby(['scenario','N','method']).size()
    if len(counts) != 30 or not (counts == 10).all():
        raise ValueError('Incomplete scenario/method/seed matrix')
    np.testing.assert_allclose(frame.telemetry_energy+frame.commands_energy+frame.topology_energy,
                               frame.deployment_energy,rtol=1e-12,atol=1e-12)
    frame.to_csv(output/'deployment_per_seed.csv',index=False)
    paired=frame[frame.method=='ECB-HD-CR'].merge(
        frame[frame.method=='D3QN-EH'],on=['scenario','N','run_id'],suffixes=('_p','_d'))
    differences=[]
    for (scenario,n),group in paired.groupby(['scenario','N']):
        entry=dict(scenario=scenario,N=n,energy_reduction_pct=100*(1-
            group.energy_per_task_uJ_p.mean()/group.energy_per_task_uJ_d.mean()))
        for metric in ['completion','FND','objective']:
            delta=group[metric+'_p']-group[metric+'_d']
            half=student_t.ppf(.975,9)*delta.std(ddof=1)/np.sqrt(10)
            entry.update({metric+'_diff':delta.mean(),metric+'_ci95_low':delta.mean()-half,
                          metric+'_ci95_high':delta.mean()+half})
        differences.append(entry)
    pd.DataFrame(differences).to_csv(output/'paired_deployment_differences.csv',index=False)
    metrics=['completion','FND','energy_per_task_uJ','objective','all_control_share_pct',
             'deployment_energy','ecb_control_energy','control_failed_transactions','upper_bound','gap_upper']
    aggregate=frame.groupby(['scenario','N','method'])[metrics].agg(['mean','std'])
    aggregate.columns=['_'.join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output/'deployment_summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'deployment_diagnostics.csv',index=False)
    (output/'evaluation_contract.json').write_text(json.dumps(dict(
        weights=WEIGHTS,energy_reference_J=ENERGY_REFERENCE_J,
        bound='Tracewise hindsight relaxation; not an exact causal optimum',
        deployment_runs=len(frame),seeds_per_cell=10,
        protocol=dict(telemetry_base_bits=128,per_task_descriptor_bits=32,
                      routing_command_bits=64,periodic_topology_bits=64,
                      ecb_transaction_bits='256 uplink + 256 downlink'),
        omitted=['processor energy','contention','retransmissions','extra recovery signaling']),indent=2),encoding='utf-8')
    print(aggregate.loc[aggregate.index.get_level_values('method')=='ECB-HD-CR'].to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--legacy-only',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    out=ROOT/'results/review_revision_v1'; out.mkdir(parents=True,exist_ok=True)
    path=out/'legacy_bound_audit.csv'
    legacy=pd.read_csv(path) if path.exists() else legacy_audit(out,args.workers)
    if not args.legacy_only:
        deployment_export(out,legacy)


if __name__=='__main__':
    main()
