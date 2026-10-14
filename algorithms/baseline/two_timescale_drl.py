"""
Two-Timescale-DRL: DRL-based two-timescale approach without credit mechanism
Reference: Liu et al. (2024) IEEE TWC

Long timescale: Fixed budget allocation based on predicted harvest
Short timescale: D3QN scheduling
No closed-loop feedback from actual bills

Plan-B differentiation: the cycle budget is a hard spending cap. Once a node has
spent its allocated budget, remaining tasks in that cycle are dropped even if the
battery could still pay the radio cost. This separates the baseline from pure
battery-only D3QN-EH.
"""

import numpy as np
from core.d3qn import D3QN_Agent
from .base_algorithm import BaseAlgorithm


class TwoTimescaleDRL(BaseAlgorithm):
    """Two-timescale DRL without behavioral feedback."""

    def __init__(self, network, config: dict):
        super().__init__(network, config)

        # [battery, budget_remaining, predicted_harvest, batch_size, cycle phase]
        state_dim = 5
        # [drop, smallest-packet first, highest-priority first, FIFO]
        action_dim = 4

        self.d3qn = D3QN_Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.get("lr", 0.001),
            gamma=config.get("gamma", 0.9),
            epsilon_start=config.get(
                "two_timescale_epsilon_start",
                config.get("epsilon_start", 1.0),
            ),
            epsilon_end=config.get(
                "two_timescale_epsilon_end",
                config.get("epsilon_end", 0.01),
            ),
            epsilon_decay=config.get(
                "two_timescale_epsilon_decay",
                config.get("epsilon_decay", 0.995),
            ),
            buffer_capacity=config.get("buffer_capacity", 10000),
            batch_size=config.get("batch_size", 64),
            target_update_freq=config.get("target_update_freq", 100),
        )
        self.exploration_decay_interval = config.get(
            "exploration_decay_interval", 10
        )
        self.energy_floor = config.get("physical_energy_floor", 0.002)

        self.T_L = config.get("T_L", 50)
        self.current_cycle = 0
        self.round_in_cycle = 0

        self.budgets = np.zeros(network.N)
        self.spent = np.zeros(network.N)

        self.beta = config.get("beta", 0.7)
        # Start near the true harvest scale (~1e-6 J/round after scenario scaling),
        # not a placeholder 0.1 J which would make the first cycle budget unbounded.
        self.harvest_estimates = np.ones(network.N) * config.get(
            "two_timescale_harvest_init", 1e-6
        )
        self._cycle_harvest_buffer = []

        # Harvest-predicted budgets are typically far below radio bills. A
        # calibrated scale keeps the hard cap meaningful without forcing almost
        # every task to drop. Override with two_timescale_budget_scale if needed.
        self.safety_factor = config.get("two_timescale_safety_factor", 0.8)
        self.budget_scale = config.get("two_timescale_budget_scale", 200.0)
        self.budget_floor = config.get("two_timescale_budget_floor", 0.0025)
        self.hard_budget = config.get("two_timescale_hard_budget", True)

        self.w_complete = config.get("two_timescale_w_complete", 10.0)
        self.w_energy = config.get("two_timescale_w_energy", 2.0)
        self.w_budget_violation = config.get(
            "two_timescale_w_budget_violation", 5.0
        )

        self.training = True

    def initialize(self):
        # Use the floor until the first observed cycle; harvest EWMA alone is too
        # small at night and too large if initialized on an unscaled placeholder.
        self.budgets = np.full(self.network.N, self.budget_floor)
        self.spent = np.zeros(self.network.N)
        self._cycle_harvest_buffer = []

    def long_timescale_update(self, cycle_harvests: np.ndarray):
        self.harvest_estimates = (
            self.beta * self.harvest_estimates
            + (1 - self.beta) * cycle_harvests
        )
        predicted = (
            self.harvest_estimates
            * self.T_L
            * self.safety_factor
            * self.budget_scale
        )
        self.budgets = np.maximum(predicted, self.budget_floor)
        self.spent = np.zeros(self.network.N)
        self.current_cycle += 1
        self.round_in_cycle = 0

    def get_state(self, node_id: int, queue_size: int) -> np.ndarray:
        node = self.network.nodes[node_id]
        budget_remaining = max(0.0, self.budgets[node_id] - self.spent[node_id])
        return np.array(
            [
                node.battery / node.B_max,
                budget_remaining / (self.budgets[node_id] + 1e-6),
                self.harvest_estimates[node_id],
                min(queue_size / 10.0, 1.0),
                self.round_in_cycle / self.T_L,
            ]
        )

    def compute_reward(
        self,
        completed_count: int,
        energy_spent: float,
        dropped_count: int,
        budget_remaining: float,
        node_alive: bool,
    ) -> float:
        reward = self.w_complete * completed_count
        reward -= self.w_energy * energy_spent
        reward -= dropped_count
        if budget_remaining < 0:
            reward -= self.w_budget_violation * abs(budget_remaining)
        if not node_alive:
            reward -= 100.0
        return reward

    def get_valid_actions(self, node_id: int) -> np.ndarray:
        node = self.network.nodes[node_id]
        if not node.is_alive or node.battery < self.energy_floor:
            return np.array([True, False, False, False], dtype=bool)
        if self.hard_budget and self.spent[node_id] >= self.budgets[node_id]:
            # Budget exhausted for this cycle: only drop remains available.
            return np.array([True, False, False, False], dtype=bool)
        return np.array([False, True, True, True], dtype=bool)

    def schedule_round(self, tasks, current_round: int, solar_harvests: np.ndarray):
        self.stats["rounds_executed"] += 1
        self._cycle_harvest_buffer.append(solar_harvests.copy())

        for i, node in enumerate(self.network.nodes):
            if node.is_alive:
                node.harvest(solar_harvests[i])

        node_tasks = {i: [] for i in range(self.network.N)}
        for task in tasks:
            node_tasks[task.source_node].append(task)

        results = {"completed": 0, "dropped": 0, "energy": 0.0}

        for node_id in range(self.network.N):
            node = self.network.nodes[node_id]
            if not node.is_alive:
                results["dropped"] += len(node_tasks[node_id])
                continue

            tasks_at_node = node_tasks[node_id]
            if len(tasks_at_node) == 0:
                continue

            state = self.get_state(node_id, len(tasks_at_node))
            action = self.d3qn.select_action(
                state,
                valid_actions=self.get_valid_actions(node_id),
                eval_mode=not self.training,
            )

            if action == 0:
                results["dropped"] += len(tasks_at_node)
                completed_count = 0
                energy_spent = 0.0
            else:
                completed_count = 0
                energy_spent = 0.0
                if action == 1:
                    ordered_tasks = sorted(
                        tasks_at_node, key=lambda task: task.data_size
                    )
                elif action == 2:
                    ordered_tasks = sorted(
                        tasks_at_node,
                        key=lambda task: (-task.priority, task.task_id),
                    )
                else:
                    ordered_tasks = tasks_at_node

                for task in ordered_tasks:
                    dist_to_bs = node.distance_to(self.network.bs)
                    energy_cost = self.network.E_tx(task.data_size, dist_to_bs)
                    remaining = self.budgets[node_id] - self.spent[node_id]
                    if self.hard_budget and energy_cost > remaining + 1e-15:
                        # Hard cycle budget: skip unaffordable packets.
                        continue
                    if node.consume(energy_cost):
                        completed_count += 1
                        energy_spent += energy_cost
                        self.spent[node_id] += energy_cost
                    else:
                        continue

                results["completed"] += completed_count
                results["dropped"] += len(tasks_at_node) - completed_count
                results["energy"] += energy_spent
                self.record_energy(node_id, energy_spent)

            budget_remaining = self.budgets[node_id] - self.spent[node_id]
            reward = self.compute_reward(
                completed_count,
                energy_spent,
                len(tasks_at_node) - completed_count,
                budget_remaining,
                node.is_alive,
            )
            next_state = self.get_state(node_id, 0)

            if self.training:
                self.d3qn.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done=not node.is_alive,
                    next_valid_actions=self.get_valid_actions(node_id),
                )

        if self.training:
            self.d3qn.train_step()
            if current_round % self.exploration_decay_interval == 0:
                self.d3qn.decay_epsilon()

        self.round_in_cycle += 1
        if self.round_in_cycle >= self.T_L:
            cycle_harvests = np.mean(
                np.asarray(self._cycle_harvest_buffer), axis=0
            )
            self.long_timescale_update(cycle_harvests)
            self._cycle_harvest_buffer = []

        self.stats["total_tasks"] += len(tasks)
        self.stats["completed_tasks"] += results["completed"]
        self.stats["dropped_tasks"] += results["dropped"]
        self.stats["total_energy_consumed"] += results["energy"]

        return results
