"""Certified relaxation of the manuscript's direct/member-to-CH route model.

This is an upper bound, not an exact causal oracle or a learned-policy proof.
It drops energy causality, relay battery/CH-count/credit constraints, forwarding
costs and control costs, allows hindsight selection of delivered tasks, and
sets FND to the horizon. All relaxations enlarge the achievable objective.
"""
import numpy as np
from core.network import EH_WSN
from data.task_model import TaskGenerator

WEIGHTS = {"completion": 0.45, "lifetime": 0.35, "energy": 0.20}
ENERGY_REFERENCE_J = 1e-4


def objective(result):
    horizon = int(result["config_snapshot"]["total_rounds"])
    energy_per_task = float(result["total_energy"]) / max(int(result["completed_tasks"]), 1)
    return (WEIGHTS["completion"] * float(result["completion_rate"])
            + WEIGHTS["lifetime"] * float(result["fnd"]) / horizon
            - WEIGHTS["energy"] * energy_per_task / ENERGY_REFERENCE_J)


def bound_from_costs(costs, lifetime_upper_fraction=1.0):
    costs = np.sort(np.asarray(costs, dtype=float))
    if np.any(costs < 0) or not np.all(np.isfinite(costs)):
        raise ValueError("lower costs must be finite and nonnegative")
    if len(costs) == 0:
        return WEIGHTS["lifetime"] * lifetime_upper_fraction
    delivered = np.arange(1, len(costs) + 1)
    values = (WEIGHTS["completion"] * delivered / len(costs)
              + WEIGHTS["lifetime"] * lifetime_upper_fraction
              - WEIGHTS["energy"] * np.cumsum(costs) / delivered / ENERGY_REFERENCE_J)
    return max(float(np.max(values)), WEIGHTS["lifetime"] * lifetime_upper_fraction)


def trace_upper_bound(result):
    config = result["config_snapshot"]
    network = EH_WSN(N=result["N"], M=config.get("field_size", 100.0),
                     bs_position=config.get("bs_position", (50, 50)),
                     seed=result["topology_seed"])
    unit_lower = []
    for source in network.nodes:
        direct = network.E_tx(1, source.distance_to(network.bs))
        member = [network.E_tx(1, source.distance_to(relay)) + network.E_rx(1)
                  for relay in network.nodes if relay.id != source.id]
        unit_lower.append(min([direct] + member))
    generator = TaskGenerator(result["N"], config.get("arrival_rate", 1.0),
                              config.get("data_size_range", (100, 500)),
                              seed=result["data_seed"] + 100000)
    regime = config.get("regime_shift")
    costs = []
    for t in range(int(config["total_rounds"])):
        if regime and t == int(regime["round"]):
            generator.arrival_rate *= float(regime.get("arrival_mult", 1.0))
        costs.extend(task.data_size * unit_lower[task.source_node]
                     for task in generator.generate_tasks_for_round(t))
    if len(costs) != result["total_tasks"]:
        raise ValueError("replayed task count does not match stored experiment")
    return bound_from_costs(costs)
