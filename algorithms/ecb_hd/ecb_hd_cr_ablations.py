"""Ablation variants for ECB-HD-CR."""

from __future__ import annotations

from .ecb_hd_cr import ECB_HD_CR


class ECB_HD_CR_NoCluster(ECB_HD_CR):
    """Disable clustering and CH aggregation; nodes transmit directly."""

    def __init__(self, network, config: dict):
        cfg = dict(config)
        cfg["enable_clustering"] = False
        cfg["enable_aggregation"] = False
        cfg["enable_role_credit"] = False
        super().__init__(network, cfg)


class ECB_HD_CR_NoRoleCredit(ECB_HD_CR):
    """Use the same credit multiplier for all node roles."""

    def __init__(self, network, config: dict):
        cfg = dict(config)
        cfg["enable_role_credit"] = False
        super().__init__(network, cfg)


class ECB_HD_CR_NoAggregation(ECB_HD_CR):
    """Keep clustering/routing but forward the full payload without aggregation."""

    def __init__(self, network, config: dict):
        cfg = dict(config)
        cfg["enable_aggregation"] = False
        super().__init__(network, cfg)


class ECB_HD_CR_NoFeedback(ECB_HD_CR):
    """Disable bill-driven quota adaptation while retaining clustering and pacing.

    Credits stay harvest-anchored every cycle (wrong physical scale). Credit
    pacing remains on, so the mis-scaled quota becomes a hard delivery limit.
    This isolates closed-loop bill tracking from the rest of the architecture.
    """

    def __init__(self, network, config: dict):
        cfg = dict(config)
        cfg["enable_feedback"] = False
        cfg["enable_bill_state"] = True  # expose remaining quota to the state
        cfg["enable_bill_reward"] = False
        cfg["enable_control_overhead"] = False
        cfg["enable_credit_pacing"] = True
        super().__init__(network, cfg)
