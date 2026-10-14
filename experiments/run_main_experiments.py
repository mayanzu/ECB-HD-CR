"""
Experiment Runner with Parallel Execution
Run main experiments, ablation studies, and sensitivity analysis
"""

import numpy as np
import os
import pickle
import random
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.network import EH_WSN
from core.control_plane import CentralControlPlane
from data.solar_model import SolarTraceGenerator
from data.task_model import TaskGenerator

# Import all algorithms
from algorithms.ecb_hd.ecb_hd_cr import ECB_HD_CR
from algorithms.ecb_hd.ecb_hd_cr_ablations import (
    ECB_HD_CR_NoAggregation,
    ECB_HD_CR_NoCluster,
    ECB_HD_CR_NoFeedback,
    ECB_HD_CR_NoRoleCredit,
)
from algorithms.baseline.d3qn_eh import D3QN_EH
from algorithms.baseline.ecb_static import ECB_Static
from algorithms.baseline.two_timescale_drl import TwoTimescaleDRL
from algorithms.baseline.chhfo import CHHFO
from algorithms.baseline.icr_hhodrl import ICR_HHODRL


def get_algorithm(method_name: str, network, config: dict):
    """Factory function to create algorithm instance"""
    algorithms = {
        'ECB-HD-CR': ECB_HD_CR,
        'ECB-HD-CR-NoCluster': ECB_HD_CR_NoCluster,
        'ECB-HD-CR-NoRoleCredit': ECB_HD_CR_NoRoleCredit,
        'ECB-HD-CR-NoAggregation': ECB_HD_CR_NoAggregation,
        'ECB-HD-CR-NoFeedback': ECB_HD_CR_NoFeedback,
        'D3QN-EH': D3QN_EH,
        'ECB-Static': ECB_Static,
        'Two-Timescale-DRL': TwoTimescaleDRL,
        'CHHFO': CHHFO,
        'ICR-HHODRL': ICR_HHODRL,
    }

    if method_name not in algorithms:
        raise ValueError(f"Unknown method: {method_name}")

    return algorithms[method_name](network, config)


def run_single_experiment(params):
    """
    Run a single experiment (for parallel execution)

    Args:
        params: Tuple of (method_name, N, run_id, config, output_dir)

    Returns:
        Result dictionary
    """
    method_name, N, run_id, config, output_dir = params

    seed_offset = int(config.get('seed_offset', 0))
    data_seed = int(config.get('data_seed_base', 1000)) + seed_offset + run_id
    topology_seed = int(config.get('topology_seed_base', 2000)) + seed_offset + run_id
    method_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(method_name))
    algorithm_seed = (
        int(config.get('algorithm_seed_base', 3000))
        + seed_offset
        + run_id * 997
        + method_seed
    )

    # Separate environment data RNGs from algorithm RNGs. This keeps topology,
    # harvest, and task streams reproducible while preventing an algorithm's
    # internal random draws from changing the next round's task arrivals.
    np.random.seed(algorithm_seed)
    random.seed(algorithm_seed)
    torch.manual_seed(algorithm_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(algorithm_seed)

    # Create network. The topology seed is tied to run_id so that each
    # independent run draws a different node layout (genuine per-seed variance).
    network = EH_WSN(
        N=N,
        M=config.get('field_size', 100.0),
        bs_position=config.get('bs_position', (50, 50)),
        seed=topology_seed,
    )

    # Energy constraints are configurable so exploratory runs can quickly
    # search for regimes where lifetime and completion-rate differences appear.
    E_init = config.get('E_init', 0.5)
    B_max = config.get('B_max', 5.0)
    for node in network.nodes:
        node.E_init = E_init
        node.battery = E_init
        node.B_max = B_max

    # Generate the common irradiance-derived base trace.  The NSRDB path maps
    # each round to the next hourly GHI observation but retains the same 60 s
    # equivalent integration window as the synthetic path; it is therefore a
    # measured-shape replay, not a one-hour energy integration.
    solar_gen = SolarTraceGenerator(seed=data_seed)
    total_rounds = config.get('total_rounds', 30000)
    if config.get('harvest_source', 'synthetic') == 'nsrdb':
        from data.nsrdb_loader import nsrdb_base_trace
        base_trace = nsrdb_base_trace(
            config['nsrdb_csv'],
            total_rounds,
            seconds_per_round=config.get('nsrdb_seconds_per_round', 60.0),
        )
    else:
        base_trace = solar_gen.generate_synthetic_trace(total_rounds)
    # Keep node-level shadowing identical when comparing synthetic and NSRDB
    # sources.  A separate RNG prevents the synthetic weather draws from
    # advancing the spatial-factor stream.
    spatial_gen = SolarTraceGenerator(seed=data_seed + 50000)
    spatial_traces = spatial_gen.add_spatial_variation(base_trace, N)

    # ENERGY SCALING: Reduce solar energy to create realistic scarcity
    # SOLAR ENERGY SCALING: Balance harvest vs consumption
    # Original energy is ~1M times too high (harvest >> consumption)
    # Scale down to balance harvest and consumption for realistic lifetime
    ENERGY_SCALE_FACTOR = config.get('energy_scale_factor', 0.0000001)
    spatial_traces = spatial_traces * ENERGY_SCALE_FACTOR

    # Non-stationary regime shift: at a mid-run round the operating regime
    # changes (a cloudy spell that dims harvest and/or a traffic surge). A
    # static energy budget set for the initial regime then mis-paces the
    # network, whereas the closed-loop credit tracks the new consumption.
    regime = config.get('regime_shift')
    regime_round = int(regime['round']) if regime else None
    regime_arrival_mult = float(regime.get('arrival_mult', 1.0)) if regime else 1.0
    if regime:
        peak_scale = float(regime.get('peak_scale', 1.0))
        if peak_scale != 1.0:
            spatial_traces[:, regime_round:] *= peak_scale
        if regime.get('reshuffle_spatial', False):
            perm = np.random.RandomState(data_seed + 70000).permutation(N)
            spatial_traces[:, regime_round:] = spatial_traces[perm, regime_round:]

    task_gen = TaskGenerator(
        num_nodes=N,
        arrival_rate=config.get('arrival_rate', 1.0),
        data_size_range=config.get('data_size_range', (100, 500)),
        seed=data_seed + 100000,
    )

    # Initialize algorithm
    algorithm = get_algorithm(method_name, network, config)
    algorithm.initialize()
    control_plane = (
        CentralControlPlane(network, algorithm, config)
        if config.get('deployment_protocol', 'legacy') == 'central_bs' else None
    )

    # Statistics tracking
    stats = {
        'method': method_name,
        'N': N,
        'run_id': run_id,
        'alive_count': [],
        'round': [],
        'fnd': None,
        'hnd': None,
        'lnd': None,
        'total_tasks': 0,
        'completed_tasks': 0,
        'total_energy': 0.0,
        'background_energy': 0.0,
        'data_seed': data_seed,
        'topology_seed': topology_seed,
        'algorithm_seed': algorithm_seed,
        'config_snapshot': dict(config),
        # Segmented endpoints (pre/post regime switch when configured).
        'pre_switch_tasks': 0,
        'pre_switch_completed': 0,
        'post_switch_tasks': 0,
        'post_switch_completed': 0,
        'post_switch_fnd': None,
        'switch_round': regime_round,
    }

    # Main simulation loop
    for t in range(total_rounds):
        network.current_round = t

        # Apply the traffic-surge component of a regime shift, if configured.
        if regime_round is not None and t == regime_round and regime_arrival_mult != 1.0:
            task_gen.arrival_rate *= regime_arrival_mult

        # Energy harvesting
        solar_harvests = spatial_traces[:, t]

        # Generate tasks
        tasks = task_gen.generate_tasks_for_round(t)
        stats['total_tasks'] += len(tasks)
        if regime_round is not None:
            if t < regime_round:
                stats['pre_switch_tasks'] += len(tasks)
            else:
                stats['post_switch_tasks'] += len(tasks)

        # Schedule tasks
        deployment_energy = 0.0
        saved_alive = None
        if control_plane is not None:
            deployment_energy, saved_alive = control_plane.begin_round(tasks, t, solar_harvests)
        try:
            results = algorithm.schedule_round(tasks, t, solar_harvests)
            completed = results.get('completed', 0)
            stats['completed_tasks'] += completed
            stats['total_energy'] += results.get('energy', 0.0) + deployment_energy
            if regime_round is not None:
                if t < regime_round:
                    stats['pre_switch_completed'] += completed
                else:
                    stats['post_switch_completed'] += completed
        except Exception as e:
            raise RuntimeError(
                f"{method_name} run {run_id} failed at round {t}"
            ) from e
        finally:
            if control_plane is not None:
                control_plane.end_round(saved_alive)

        # BASE ENERGY CONSUMPTION: Every alive node consumes energy per round
        # This includes sensing, idle listening, synchronization
        BASE_ENERGY_PER_ROUND = config.get('base_energy_per_round', 0.00005)
        for node in network.nodes:
            if node.is_alive:
                stats['background_energy'] += node.consume_up_to(
                    BASE_ENERGY_PER_ROUND
                )

        for node in network.nodes:
            node.check_alive(t)

        # Update network stats
        alive = network.update_alive_count()
        stats['alive_count'].append(alive)
        stats['round'].append(t)

        # Record FND, HND, LND
        if network.fnd is not None and stats['fnd'] is None:
            stats['fnd'] = network.fnd
        if network.hnd is not None and stats['hnd'] is None:
            stats['hnd'] = network.hnd
        if network.lnd is not None and stats['lnd'] is None:
            stats['lnd'] = network.lnd

    # Final stats
    if stats['fnd'] is None:
        stats['fnd'] = total_rounds
    if stats['hnd'] is None:
        stats['hnd'] = total_rounds
    if stats['lnd'] is None:
        stats['lnd'] = total_rounds

    # First death after the regime switch, or horizon if none.
    if regime_round is not None:
        if stats['fnd'] is not None and stats['fnd'] >= regime_round:
            stats['post_switch_fnd'] = stats['fnd']
        else:
            # If FND was before the switch, look for first further death after switch.
            post_deaths = [
                t for t, alive in zip(stats['round'], stats['alive_count'])
                if t >= regime_round and alive < N
            ]
            # Prefer the first round where alive drops relative to switch-time count.
            if stats['fnd'] < regime_round:
                switch_alive = None
                for t, alive in zip(stats['round'], stats['alive_count']):
                    if t == regime_round - 1:
                        switch_alive = alive
                        break
                if switch_alive is None:
                    switch_alive = N
                post_drop = [
                    t for t, alive in zip(stats['round'], stats['alive_count'])
                    if t >= regime_round and alive < switch_alive
                ]
                stats['post_switch_fnd'] = (
                    post_drop[0] if post_drop else total_rounds
                )
            else:
                stats['post_switch_fnd'] = stats['fnd']

    stats['completion_rate'] = (stats['completed_tasks'] / stats['total_tasks']
                                if stats['total_tasks'] > 0 else 0.0)
    stats['pre_switch_completion'] = (
        stats['pre_switch_completed'] / stats['pre_switch_tasks']
        if stats['pre_switch_tasks'] > 0 else 0.0
    )
    stats['post_switch_completion'] = (
        stats['post_switch_completed'] / stats['post_switch_tasks']
        if stats['post_switch_tasks'] > 0 else 0.0
    )
    stats['budget_deviation'] = (
        algorithm.stats.get('budget_deviation_sum', 0.0) /
        max(1, algorithm.stats.get('budget_deviation_count', 0))
        if hasattr(algorithm, 'stats') else 0.0
    )

    # Control-plane overhead of the credit mechanism (downlink quota/rates +
    # uplink bill/completion), charged to the radio energy model. Baselines
    # without a credit control plane report zero here. Also record the total
    # alive node-rounds so a survivor-bias-free energy metric can be computed.
    stats['control_energy'] = (
        algorithm.stats.get('control_energy', 0.0)
        if hasattr(algorithm, 'stats') else 0.0
    )
    stats['alive_node_rounds'] = int(sum(stats['alive_count']))
    stats['total_physical_energy'] = (
        stats['total_energy'] + stats['background_energy']
    )
    stats['deployment_energy'] = sum(control_plane.energy.values()) if control_plane else 0.0
    stats['deployment_energy_components'] = dict(control_plane.energy) if control_plane else {}
    stats['deployment_message_counts'] = dict(control_plane.messages) if control_plane else {}
    stats['control_failed_transactions'] = control_plane.failed_transactions if control_plane else 0
    if config.get('record_diagnostics', False):
        target = getattr(algorithm, 'cr', algorithm)
        stats['diagnostics'] = getattr(target, 'diagnostics', [])

    if len(stats['round']) != total_rounds:
        raise RuntimeError(
            f"{method_name} run {run_id} produced {len(stats['round'])} "
            f"rounds, expected {total_rounds}"
        )
    if not 0 <= stats['completed_tasks'] <= stats['total_tasks']:
        raise RuntimeError(
            f"invalid task counts for {method_name} run {run_id}: "
            f"{stats['completed_tasks']} / {stats['total_tasks']}"
        )
    if (
        not np.isfinite(stats['total_physical_energy'])
        or stats['total_physical_energy'] < 0
    ):
        raise RuntimeError(
            f"invalid energy total for {method_name} run {run_id}: "
            f"{stats['total_physical_energy']}"
        )

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    filename = Path(output_dir) / f'{method_name}_N{N}_run{run_id:02d}.pkl'
    temporary = filename.with_suffix(filename.suffix + '.tmp')

    with temporary.open('wb') as f:
        pickle.dump(stats, f)
    os.replace(temporary, filename)

    return stats
