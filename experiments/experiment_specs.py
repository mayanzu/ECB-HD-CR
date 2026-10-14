"""Shared experiment specifications for publication runners.

Keeping methods, base hyperparameters, and scenario overrides in one module
prevents the paper-facing runners from drifting apart.

Plan B (2026-07-11):
- direct baselines enforce cycle budgets (see baseline implementations)
- regime-switch parameters keep nodes alive past the switch
- scale check uses 10 seeds
- credit_mismatch scenario stresses wrong quota scale / load step
"""

from __future__ import annotations


PROPOSED = "ECB-HD-CR"

BASELINE_METHODS = [
    PROPOSED,
    "D3QN-EH",
    "ECB-Static",
    "Two-Timescale-DRL",
    "ICR-HHODRL",
    "CHHFO",
]

ABLATION_METHODS = [
    PROPOSED,
    "ECB-HD-CR-NoCluster",
    "ECB-HD-CR-NoRoleCredit",
    "ECB-HD-CR-NoAggregation",
    "ECB-HD-CR-NoFeedback",
]

REGIME_SHIFT_METHODS = [
    PROPOSED,
    "ECB-Static",
    "ECB-HD-CR-NoFeedback",
    "ECB-HD-CR-NoRoleCredit",
    "ECB-HD-CR-NoCluster",
    "ECB-HD-CR-NoAggregation",
    "D3QN-EH",
    "Two-Timescale-DRL",
    "ICR-HHODRL",
    "CHHFO",
]

MISMATCH_METHODS = [
    PROPOSED,
    "ECB-HD-CR-NoFeedback",
    "ECB-Static",
    "Two-Timescale-DRL",
    "D3QN-EH",
    "ECB-HD-CR-NoCluster",
]

BASE_CONFIG = {
    "total_rounds": 500,
    "lambda_0": 0.05,
    "mu_0": 0.03,
    "eta_c": 0.3,
    "xi": 0.5,
    "beta": 0.7,
    "T_L": 50,
    "lr": 0.001,
    "gamma": 0.9,
    # All learning-based methods use the same online exploration schedule.
    "epsilon_start": 0.3,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.99,
    "exploration_decay_interval": 10,
    "buffer_capacity": 10000,
    "batch_size": 64,
    "target_update_freq": 100,
    "reward_weights": {
        "w_complete": 10.0,
        "w_energy": 2.0,
        "w_delay": 1.0,
        "w_death": 0.5,
    },
    "hho_pop_size": 10,
    "hho_iter": 8,
    "K_reoptimize": 50,
    "fuzzy_transmit_threshold": 0.2,
    "aggregation_ratio": 0.25,
    "cluster_gain_threshold": 0.18,
    "ch_energy_reserve_ratio": 0.08,
    "ch_credit_multiplier": 2.4,
    "relay_credit_multiplier": 1.3,
    "control_msg_bits": 256,
    # Credit pacing: schedule only while remaining cycle quota covers the packet.
    # Closed-loop feedback keeps the quota on the bill scale; NoFeedback keeps a
    # harvest-anchored (mis-scaled) quota and therefore incurs a physical penalty.
    "enable_credit_pacing": True,
    # Tuned 2026-07-11 (pacing_tune refine, 4 seeds): edge completion ~0.9997,
    # Full-NoFeedback gap ~0.90, edge E/task reduction ~13.5% vs D3QN.
    "credit_pacing_overshoot": 3.5,
    "credit_share_boost": 2.2,
    # Two-Timescale-DRL hard cycle budget calibration (radio-scale).
    # Typical edge direct cycle bills are ~3e-3..1.5e-2 J; night harvest is ~0,
    # so the floor dominates and must sit below mean bill to bind.
    "two_timescale_hard_budget": True,
    "two_timescale_budget_scale": 200.0,
    "two_timescale_budget_floor": 0.0025,
    "two_timescale_safety_factor": 0.8,
    "two_timescale_harvest_init": 1e-6,
    # ECB-Static fixed hard credit (radio-scale). 0.1 J never binds under the
    # paper radio model; a few mJ forces quota-limited drops without clustering.
    "static_credit": 0.0035,
    "initial_credit": 0.1,
}

STRESS_SCENARIOS = [
    {
        "name": "scarce_high_load",
        "overrides": {
            "E_init": 0.35,
            "B_max": 2.0,
            "base_energy_per_round": 0.00008,
            "energy_scale_factor": 0.00000005,
            "arrival_rate": 1.5,
        },
    },
    {
        "name": "small_battery_bursty",
        "overrides": {
            "E_init": 0.30,
            "B_max": 1.4,
            "base_energy_per_round": 0.00008,
            "energy_scale_factor": 0.00000006,
            "arrival_rate": 1.8,
        },
    },
    {
        "name": "high_pressure",
        "overrides": {
            "E_init": 0.40,
            "B_max": 2.0,
            "base_energy_per_round": 0.00010,
            "energy_scale_factor": 0.00000004,
            "arrival_rate": 2.0,
        },
    },
    {
        "name": "harvest_uncertain",
        "overrides": {
            "E_init": 0.40,
            "B_max": 2.5,
            "base_energy_per_round": 0.00007,
            "energy_scale_factor": 0.00000005,
            "arrival_rate": 1.7,
        },
    },
    {
        "name": "edge_bs_high_load",
        "overrides": {
            "E_init": 0.36,
            "B_max": 2.0,
            "base_energy_per_round": 0.00009,
            "energy_scale_factor": 0.000000045,
            "arrival_rate": 1.9,
            "bs_position": (50.0, 150.0),
        },
    },
    {
        # Deliberately pushed past the network's lifetime limit so that an
        # interpretable performance distribution emerges.
        "name": "critical_scarcity",
        "overrides": {
            "E_init": 0.28,
            "B_max": 1.4,
            "base_energy_per_round": 0.00014,
            "energy_scale_factor": 0.000000025,
            "arrival_rate": 2.4,
            "bs_position": (50.0, 150.0),
        },
    },
]

# Plan B: nodes must commonly survive past the switch so post-switch endpoints
# are measurable. Pre-switch is milder; the switch still imposes a sharp stress.
REGIME_SHIFT_SCENARIOS = [
    {
        "name": "shift_large_field_edge",
        "overrides": {
            "field_size": 150.0,
            "E_init": 0.55,
            "B_max": 2.5,
            "base_energy_per_round": 0.00008,
            "energy_scale_factor": 0.00000006,
            "arrival_rate": 1.05,
            "data_size_range": (100, 500),
            "bs_position": (75.0, 225.0),
            "regime_shift": {
                "round": 150,
                "peak_scale": 0.22,
                "arrival_mult": 2.0,
                "reshuffle_spatial": True,
            },
        },
    },
]

# Plan B mismatch scenario: edge geometry + deliberately wrong fixed credit
# scale + mid-horizon load step. Full ECB can re-scale R(k); static/hard-budget
# baselines cannot.
MISMATCH_SCENARIOS = [
    {
        "name": "credit_mismatch_edge",
        "overrides": {
            "E_init": 0.40,
            "B_max": 2.0,
            "base_energy_per_round": 0.00009,
            "energy_scale_factor": 0.00000005,
            "arrival_rate": 1.6,
            "bs_position": (50.0, 150.0),
            # Far below a typical scheduling cycle bill under edge routing (~0.008 J).
            "initial_credit": 0.0015,
            "static_credit": 0.0015,
            "regime_shift": {
                "round": 200,
                "peak_scale": 0.45,
                "arrival_mult": 1.8,
                "reshuffle_spatial": False,
            },
        },
    },
]


def scenario_map(scenarios: list[dict]) -> dict[str, dict]:
    """Return scenarios keyed by name."""
    return {scenario["name"]: scenario for scenario in scenarios}
