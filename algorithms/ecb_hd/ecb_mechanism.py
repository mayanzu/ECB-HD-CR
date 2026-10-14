"""Reusable Energy Credit Bank mechanism."""

import numpy as np


class ECB_Mechanism:
    """Virtual credit accounts with dynamic penalty and saving rates."""

    def __init__(
        self,
        num_nodes: int,
        lambda_0: float = 0.05,
        mu_0: float = 0.03,
        eta_c: float = 0.3,
        xi: float = 0.5,
        alpha_lambda: float = None,
        gamma_mu: float = None,
        lambda_max: float = 0.5,
        mu_min: float = 0.01,
        rate_decay: float = 0.95,
        rate_growth: float = 1.05,
    ):
        self.num_nodes = num_nodes
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.eta_c = eta_c
        self.xi = xi
        # Rate-adaptation gains. They default to eta_c so the dynamics match the
        # original implementation, but are now named parameters (alpha_lambda,
        # gamma_mu) matching Equations (12)-(13), with saturation caps
        # [lambda_0, lambda_max] / [mu_min, mu_0] and geometric relaxation.
        self.alpha_lambda = eta_c if alpha_lambda is None else alpha_lambda
        self.gamma_mu = eta_c if gamma_mu is None else gamma_mu
        self.lambda_max = lambda_max
        self.mu_min = mu_min
        self.rate_decay = rate_decay
        self.rate_growth = rate_growth

        self.credits = np.zeros(num_nodes)
        self.bills = np.zeros(num_nodes)
        self.lambda_rates = np.full(num_nodes, lambda_0)
        self.mu_rates = np.full(num_nodes, mu_0)

        self.overdraft_history = []
        self.savings_history = []

    def update_rates(self, cycle: int):
        """Update dynamic interest rates from previous-cycle spending.

        Recursive (leaky-integrator) form, Equations (12)-(13): an overdraft
        raises the penalty rate in proportion to its relative severity
        o_i(k) = [delta_i(k)]^+ (capped at lambda_max); a clean cycle relaxes it
        geometrically back toward lambda_0. The saving reward rate is the
        mirror image, bounded in [mu_min, mu_0].
        """
        for i in range(self.num_nodes):
            overdraft = max(0, self.bills[i] - self.credits[i])
            savings = max(0, self.credits[i] - self.bills[i])

            if overdraft > 0:
                self.lambda_rates[i] = min(
                    self.lambda_max,
                    self.lambda_rates[i]
                    + self.alpha_lambda * (overdraft / (self.credits[i] + 1e-6)),
                )
            else:
                self.lambda_rates[i] = max(
                    self.lambda_0, self.lambda_rates[i] * self.rate_decay
                )

            if savings > 0:
                self.mu_rates[i] = max(
                    self.mu_min,
                    self.mu_rates[i]
                    - self.gamma_mu * (savings / (self.credits[i] + 1e-6)),
                )
            else:
                self.mu_rates[i] = min(self.mu_0, self.mu_rates[i] * self.rate_growth)

    def compute_reward(self, node_id: int, base_reward: float) -> float:
        """Apply ECB penalty/reward to a task-level base reward."""
        bill = self.bills[node_id]
        credit = self.credits[node_id]

        if bill > credit:
            overdraft = bill - credit
            penalty = self.lambda_rates[node_id] * overdraft / (credit + 1e-6)
            return base_reward - penalty

        savings = credit - bill
        reward = self.mu_rates[node_id] * savings / (credit + 1e-6)
        return base_reward + min(reward, 1.0)

    def record_consumption(self, node_id: int, energy: float):
        """Accumulate actual energy bill for the current credit cycle."""
        self.bills[node_id] += energy

    def record_bill(self, node_id: int, energy: float):
        """Compatibility alias for recording current-cycle energy spending."""
        self.record_consumption(node_id, energy)

    def update_credits(self, new_credits: np.ndarray):
        """Commit next-cycle credit quotas and clear current-cycle bills."""
        self.credits = new_credits.copy()
        self.bills = np.zeros(self.num_nodes)

    def get_remaining_credit(self, node_id: int) -> float:
        """Return remaining credit for node ``node_id``."""
        return max(0, self.credits[node_id] - self.bills[node_id])
