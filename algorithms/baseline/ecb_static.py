"""ECB-Static baseline with fixed credit quotas and a hard cycle quota.

Unlike ECB-HD-CR, credits do not track realized bills. The fixed quota is
enforced as a hard per-cycle spending cap on direct transmissions so that the
baseline is no longer execution-equivalent to unconstrained D3QN-EH.
"""

import numpy as np

from algorithms.ecb_hd.ecb_hd_cr import ECB_HD_CR
from algorithms.ecb_hd.ecb_mechanism import ECB_Mechanism
from .base_algorithm import BaseAlgorithm


class ECB_Static(BaseAlgorithm):
    """Static-credit direct-routing baseline without bill-driven feedback."""

    def __init__(self, network, config: dict):
        super().__init__(network, config)
        cr_config = dict(config)
        cr_config.update(
            {
                "enable_clustering": False,
                "enable_role_credit": False,
                "enable_feedback": False,
                "enable_control_overhead": False,
                # Keep bill state so remaining credit is visible to the router,
                # but freeze rates and disable soft reward shaping.
                "enable_bill_state": True,
                "enable_bill_reward": False,
                # Hard fixed quota distinguishes this baseline from D3QN-EH.
                "hard_credit_gate": True,
            }
        )
        self.cr = ECB_HD_CR(network, cr_config)

        class StaticECB(ECB_Mechanism):
            """ECB with frozen rates."""

            def update_rates(self, cycle: int):
                return

        self.cr.ecb = StaticECB(
            network.N,
            lambda_0=config.get("lambda_0", 0.05),
            mu_0=config.get("mu_0", 0.03),
            eta_c=0.0,
            xi=config.get("xi", 0.5),
        )
        self._initial_credit = config.get(
            "static_credit", config.get("initial_credit", 0.0035)
        )

    def initialize(self):
        self.cr.initialize()
        self.cr.ecb.credits = np.ones(self.network.N) * self._initial_credit

    def schedule_round(self, tasks, current_round: int, solar_harvests: np.ndarray):
        self.stats["rounds_executed"] += 1

        results = self.cr.schedule_round(tasks, current_round, solar_harvests)

        # Re-pin the fixed quota after every cycle settlement performed inside CR.
        if self.cr.round_in_cycle == 0:
            self.cr.ecb.credits = np.ones(self.network.N) * self._initial_credit
            self.cr.ecb.bills = np.zeros(self.network.N)

        self.stats["total_tasks"] += len(tasks)
        self.stats["completed_tasks"] += results.get("completed", 0)
        self.stats["dropped_tasks"] += results.get("dropped", 0)
        self.stats["total_energy_consumed"] += results.get("energy", 0.0)
        self.stats["budget_deviation_sum"] = self.cr.stats.get(
            "budget_deviation_sum", 0.0
        )
        self.stats["budget_deviation_count"] = self.cr.stats.get(
            "budget_deviation_count", 0
        )

        return results
