"""
ECB-HD-CR: Energy Credit Bank with clustering and routing.

This variant keeps the ECB feedback loop but replaces direct-only task
transmission with HHO-selected cluster heads and CH aggregation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.d3qn import D3QN_Agent
from core.hho import HarrisHawksOptimizer
from .ecb_mechanism import ECB_Mechanism


class ECB_HD_CR:
    """Energy Credit Bank with clustering and routing."""

    def __init__(self, network, config: dict):
        self.network = network
        self.config = config

        self.ecb = ECB_Mechanism(
            network.N,
            lambda_0=config["lambda_0"],
            mu_0=config["mu_0"],
            eta_c=config["eta_c"],
            xi=config["xi"],
            alpha_lambda=config.get("alpha_lambda"),
            gamma_mu=config.get("gamma_mu"),
        )

        self.num_clusters = max(3, config.get("num_clusters", network.N // 20))
        self.K_reoptimize = config.get("K_reoptimize", config.get("T_L", 50))
        self.T_L = config.get("T_L", 50)
        self.current_cycle = 0
        self.round_in_cycle = 0

        self.hho = HarrisHawksOptimizer(
            dim=network.N,
            pop_size=config.get("hho_pop_size", 30),
            max_iter=config.get("hho_iter", 50),
            lb=0.0,
            ub=1.0,
        )

        # State: src battery, credit ratio, predicted harvest, dispersion,
        # distance to BS, distance to CH, CH battery, current batch size, role.
        self.d3qn = D3QN_Agent(
            state_dim=9,
            action_dim=3,  # drop fallback, direct BS, via CH
            lr=config.get("lr", 0.001),
            gamma=config.get("gamma", 0.9),
            epsilon_start=config.get("epsilon_start", 0.3),
            epsilon_end=config.get("epsilon_end", 0.01),
            epsilon_decay=config.get("epsilon_decay", 0.99),
            buffer_capacity=config.get("buffer_capacity", 10000),
            batch_size=config.get("batch_size", 64),
            target_update_freq=config.get("target_update_freq", 100),
        )
        self.exploration_decay_interval = config.get(
            "exploration_decay_interval", 10
        )

        reward_weights = config.get("reward_weights", {})
        self.w_complete = reward_weights.get("w_complete", 10.0)
        self.w_energy = reward_weights.get("w_energy", 2.0)
        self.w_drop = reward_weights.get("w_death", 0.5)

        self.beta = config.get("beta", 0.7)
        self.conf_window = config.get("conf_window", 20)
        self.harvest_estimates = np.zeros(network.N)
        self.harvest_history = [[] for _ in range(network.N)]

        self.cluster_heads: list[int] = []
        self.cluster_assignment: dict[int, int] = {}
        self.cluster_members: dict[int, list[int]] = {}
        self._cycle_harvest_buffer = []
        self._role_energy = np.zeros(network.N)
        self._last_credit_balance_ratio = np.ones(network.N)

        self.aggregation_ratio = config.get("aggregation_ratio", 0.25)
        self.cluster_gain_threshold = config.get("cluster_gain_threshold", 0.18)
        self.ch_energy_reserve_ratio = config.get("ch_energy_reserve_ratio", 0.08)
        self.cluster_member_weight = config.get("cluster_member_weight", 0.65)
        self.cluster_downstream_weight = config.get("cluster_downstream_weight", 0.45)
        self.cluster_balance_weight = config.get("cluster_balance_weight", 18.0)
        self.cluster_energy_weight = config.get("cluster_energy_weight", 650.0)
        self.cluster_credit_weight = config.get("cluster_credit_weight", 120.0)
        self.ch_credit_multiplier = config.get("ch_credit_multiplier", 2.4)
        self.relay_credit_multiplier = config.get("relay_credit_multiplier", 1.3)
        self.enable_clustering = config.get("enable_clustering", True)
        self.enable_role_credit = config.get("enable_role_credit", True)
        self.enable_aggregation = config.get("enable_aggregation", True)
        self.enable_feedback = config.get("enable_feedback", True)
        self.enable_bill_state = config.get("enable_bill_state", True)
        self.enable_bill_reward = config.get("enable_bill_reward", True)
        # Hard per-cycle credit gate: refuse transmissions that would exceed the
        # remaining virtual quota. Used by ECB-Static; the proposed method uses
        # credit pacing instead (see enable_credit_pacing).
        self.hard_credit_gate = config.get("hard_credit_gate", False)
        # Soft-to-hard pacing: schedule only while remaining cycle credit covers
        # the prospective packet. With closed-loop feedback the quota tracks the
        # bill scale, so pacing rarely starves delivery after warm-up. Without
        # feedback, harvest-anchored quotas sit on the wrong scale and pacing
        # produces a measurable physical penalty.
        self.enable_credit_pacing = config.get("enable_credit_pacing", True)
        # Allowed overshoot factor before the next packet is refused (1.0 = hard).
        self.credit_pacing_overshoot = float(
            config.get("credit_pacing_overshoot", 3.5)
        )
        # Multiplier on the closed-loop reference before health/role weighting.
        self.credit_share_boost = float(config.get("credit_share_boost", 2.2))

        # Network-wide reference per-cycle spending budget; the closed loop
        # EWMA-tracks realized consumption so the quota lives on the true bill
        # scale and follows non-stationary regime shifts.
        self._budget_ref = None

        # Control-plane message size of the ECB (per node, per cycle): one
        # downlink carrying (C_i, lambda_i, mu_i) and one uplink carrying the
        # settled bill/status. "A few tens of bytes each" -> 32 bytes = 256 bits by
        # default. This overhead is charged to the radio energy model.
        self.control_msg_bits = config.get("control_msg_bits", 256)
        self.enable_control_overhead = config.get("enable_control_overhead", True)
        self._last_control_energy = 0.0
        self.record_diagnostics = bool(config.get("record_diagnostics", False))
        self.diagnostics = []

        self.stats = {
            "total_energy_consumed": 0.0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "dropped_tasks": 0,
            "rounds_executed": 0,
            "budget_deviation_sum": 0.0,
            "budget_deviation_count": 0,
            "control_energy": 0.0,
        }

    def initialize(self):
        initial_credit = self.config.get("initial_credit", 0.1)
        self.ecb.credits = np.ones(self.network.N) * initial_credit
        self.ecb.bills = np.zeros(self.network.N)
        # Start the harvest predictor at zero and bootstrap it from the first
        # observed cycle (in _cycle_update), so the EWMA tracks the true harvest
        # scale instead of being pinned by a placeholder initial value.
        self.harvest_estimates = np.zeros(self.network.N)
        self.harvest_history = [[] for _ in range(self.network.N)]
        self.current_cycle = 0
        self.round_in_cycle = 0
        self._cycle_harvest_buffer = []
        self._role_energy = np.zeros(self.network.N)
        self._last_credit_balance_ratio = np.ones(self.network.N)
        self._optimize_clusters()

    def _optimize_clusters(self):
        if not self.enable_clustering:
            self.cluster_heads = list(range(self.network.N))
            self._assign_clusters()
            return

        alive_ids = [node.id for node in self.network.nodes if node.is_alive]
        if len(alive_ids) <= self.num_clusters:
            self.cluster_heads = alive_ids
            self._assign_clusters()
            return

        def fitness(solution):
            ch_indices = sorted(
                alive_ids, key=lambda node_id: solution[node_id]
            )[-self.num_clusters:]
            if not ch_indices:
                return np.inf

            member_dist = 0.0
            sizes = {ch: 0 for ch in ch_indices}
            energy_penalty = 0.0
            credit_penalty = 0.0

            for ch_id in ch_indices:
                ch = self.network.nodes[ch_id]
                energy_penalty += (
                    1.0 - ch.battery / ch.B_max
                ) * self.cluster_energy_weight
                credit_ratio = self._last_credit_balance_ratio[ch_id]
                credit_penalty += max(0.0, -credit_ratio) * self.cluster_credit_weight

            for node in self.network.nodes:
                if not node.is_alive:
                    continue
                if node.id in ch_indices:
                    sizes[node.id] += 1
                    continue
                nearest = min(
                    ch_indices,
                    key=lambda ch_id: node.distance_to(self.network.nodes[ch_id]),
                )
                member_dist += node.distance_to(self.network.nodes[nearest]) ** 2
                sizes[nearest] += 1

            effective_ratio = self.aggregation_ratio if self.enable_aggregation else 1.0
            downstream_dist = 0.0
            for ch_id, size in sizes.items():
                if size <= 0:
                    continue
                ch = self.network.nodes[ch_id]
                downstream_dist += size * effective_ratio * ch.distance_to(self.network.bs) ** 2

            return (
                self.cluster_member_weight * member_dist
                + self.cluster_downstream_weight * downstream_dist
                + self.cluster_balance_weight * np.std(list(sizes.values()))
                + energy_penalty
                + credit_penalty
            )

        best_solution, _ = self.hho.optimize(fitness)
        self.cluster_heads = sorted(
            alive_ids, key=lambda node_id: best_solution[node_id]
        )[-self.num_clusters:]
        if not self.cluster_heads:
            self.cluster_heads = [max(alive_ids, key=lambda i: self.network.nodes[i].battery)]
        self._assign_clusters()

    def _assign_clusters(self):
        self.cluster_assignment = {}
        self.cluster_members = {ch: [] for ch in self.cluster_heads}

        for node in self.network.nodes:
            if not node.is_alive:
                continue
            if node.id in self.cluster_heads:
                ch_id = node.id
            else:
                ch_id = min(
                    self.cluster_heads,
                    key=lambda ch: node.distance_to(self.network.nodes[ch]),
                )
            self.cluster_assignment[node.id] = ch_id
            self.cluster_members.setdefault(ch_id, []).append(node.id)

    def get_harvest_dispersion(self, node_id: int) -> float:
        history = self.harvest_history[node_id]
        if len(history) < 3:
            return 0.0
        return float(min(1.0, np.std(history) / (np.mean(history) + 1e-9)))

    def get_state(self, node_id: int, queue_size: int) -> np.ndarray:
        node = self.network.nodes[node_id]
        ch_id = self.cluster_assignment.get(node_id, node_id)
        ch_node = self.network.nodes[ch_id]
        credit_ratio = 1.0
        if self.enable_bill_state:
            credit_ratio = (
                self.ecb.credits[node_id] - self.ecb.bills[node_id]
            ) / (self.ecb.credits[node_id] + 1e-6)
        return np.array(
            [
                node.battery / node.B_max,
                np.clip(credit_ratio, -1.0, 1.0),
                self.harvest_estimates[node_id],
                self.get_harvest_dispersion(node_id),
                node.distance_to(self.network.bs) / 100.0,
                node.distance_to(ch_node) / 100.0,
                ch_node.battery / ch_node.B_max,
                min(queue_size / 10.0, 1.0),
                1.0 if node_id in self.cluster_heads else 0.0,
            ]
        )

    def _apply_bill_reward(self, node_id: int, base_reward: float) -> float:
        if not self.enable_bill_reward:
            return base_reward
        return self.ecb.compute_reward(node_id, base_reward)

    def get_valid_actions(self, node_id: int) -> np.ndarray:
        node = self.network.nodes[node_id]
        valid = np.ones(3, dtype=bool)
        if not node.is_alive or node.battery < 0.002:
            valid[1:] = False
            return valid
        # Soft default: do not drop tasks because of credit state. Hard-gate
        # and credit-pacing may still force drop when the virtual quota is gone.
        valid[0] = False
        if (
            (self.hard_credit_gate or self.enable_credit_pacing)
            and self._remaining_credit_budget(node_id) <= 0
        ):
            valid[:] = False
            valid[0] = True
            return valid
        ch_id = self.cluster_assignment.get(node_id)
        if ch_id is None or not self.network.nodes[ch_id].is_alive:
            valid[2] = False
        if node_id == ch_id:
            valid[2] = False
        if not self.enable_clustering:
            valid[2] = False
        return valid

    def _remaining_credit_budget(self, node_id: int) -> float:
        """Remaining virtual quota available for scheduling this cycle."""
        cap = self.ecb.credits[node_id] * self.credit_pacing_overshoot
        return max(0.0, cap - self.ecb.bills[node_id])

    def _credit_allows(self, node_id: int, energy: float) -> bool:
        """Return True if a prospective spend is allowed under the credit policy."""
        if not (self.hard_credit_gate or self.enable_credit_pacing):
            return True
        return energy <= self._remaining_credit_budget(node_id) + 1e-15

    def _route_action(self, node_id: int, queue_size: int) -> int:
        valid = self.get_valid_actions(node_id)
        if not valid[1] and not valid[2]:
            return 0
        if valid[1] and valid[2]:
            node = self.network.nodes[node_id]
            ch = self.network.nodes[self.cluster_assignment[node_id]]
            # Energy-aware gate: CH forwarding is only considered when the
            # estimated aggregated route has a clear energy margin.
            sample_bits = 300
            direct = self.network.E_tx(sample_bits, node.distance_to(self.network.bs))
            effective_ratio = (
                self.aggregation_ratio if self.enable_aggregation else 1.0
            )
            via_ch = (
                self.network.E_tx(sample_bits, node.distance_to(ch))
                + self.network.E_rx(sample_bits)
                + (
                    sample_bits * self.network.E_DA
                    if self.enable_aggregation
                    else 0.0
                )
                + effective_ratio
                * self.network.E_tx(sample_bits, ch.distance_to(self.network.bs))
            )
            if via_ch >= direct * (1.0 - self.cluster_gain_threshold):
                valid[2] = False
        state = self.get_state(node_id, queue_size)
        return self.d3qn.select_action(state, valid)

    def _cluster_downstream_cost(self, ch_id: int, payload_bits: int) -> float:
        if payload_bits <= 0:
            return 0.0
        ch_node = self.network.nodes[ch_id]
        effective_ratio = self.aggregation_ratio if self.enable_aggregation else 1.0
        aggregated_bits = max(1, int(payload_bits * effective_ratio))
        return (
            (
                payload_bits * self.network.E_DA
                if self.enable_aggregation
                else 0.0
            )
            + self.network.E_tx(aggregated_bits, ch_node.distance_to(self.network.bs))
        )

    def _can_accept_cluster_payload(
        self, ch_id: int, payload_bits: int, receive_bits: int = 0
    ) -> bool:
        ch_node = self.network.nodes[ch_id]
        if not ch_node.is_alive:
            return False
        reserve = self.ch_energy_reserve_ratio * ch_node.B_max
        receive_cost = self.network.E_rx(receive_bits)
        return ch_node.battery - reserve >= (
            receive_cost + self._cluster_downstream_cost(ch_id, payload_bits)
        )

    def schedule_round(self, tasks, current_round: int, solar_harvests: np.ndarray):
        self.stats["rounds_executed"] += 1
        self._cycle_harvest_buffer.append(solar_harvests.copy())

        if current_round > 0 and current_round % self.K_reoptimize == 0:
            self._optimize_clusters()

        for i, node in enumerate(self.network.nodes):
            self.harvest_history[i].append(float(solar_harvests[i]))
            if len(self.harvest_history[i]) > self.conf_window:
                self.harvest_history[i].pop(0)
            if node.is_alive:
                node.harvest(solar_harvests[i])

        node_tasks = defaultdict(list)
        for task in tasks:
            node_tasks[task.source_node].append(task)

        results = {"completed": 0, "dropped": 0, "energy": 0.0}
        cluster_payload = defaultdict(int)
        cluster_completed = defaultdict(list)
        cluster_dropped = defaultdict(int)
        transitions = []

        for node_id, tasks_at_node in node_tasks.items():
            node = self.network.nodes[node_id]
            if not node.is_alive:
                results["dropped"] += len(tasks_at_node)
                continue
            if not tasks_at_node:
                continue

            state = self.get_state(node_id, len(tasks_at_node))
            action = self._route_action(node_id, len(tasks_at_node))

            if action == 0:
                results["dropped"] += len(tasks_at_node)
                transitions.append((state, action, -1.0 * len(tasks_at_node), node_id))
                continue

            completed = 0
            decision_energy = 0.0
            ch_id = self.cluster_assignment.get(node_id, node_id)
            ch_node = self.network.nodes[ch_id]

            for task in tasks_at_node:
                if action == 1 or node_id == ch_id or not ch_node.is_alive:
                    dist = node.distance_to(self.network.bs)
                    energy = self.network.E_tx(task.data_size, dist)
                    if not self._credit_allows(node_id, energy):
                        continue
                    protects_pending_cluster = (
                        node_id == ch_id and cluster_payload[ch_id] > 0
                    )
                    reserved_downstream = (
                        self.ch_energy_reserve_ratio * node.B_max
                        + self._cluster_downstream_cost(
                            ch_id, cluster_payload[ch_id]
                        )
                        if protects_pending_cluster
                        else 0.0
                    )
                    if (
                        node.battery - energy >= reserved_downstream
                        and node.consume(energy)
                    ):
                        completed += 1
                        decision_energy += energy
                        results["completed"] += 1
                        results["energy"] += energy
                        self.ecb.record_bill(node_id, energy)
                        self._role_energy[node_id] += energy
                    else:
                        continue
                else:
                    dist = node.distance_to(ch_node)
                    e_tx = self.network.E_tx(task.data_size, dist)
                    e_rx = self.network.E_rx(task.data_size)
                    if not self._credit_allows(node_id, e_tx):
                        continue
                    projected_payload = cluster_payload[ch_id] + task.data_size
                    if not self._can_accept_cluster_payload(
                        ch_id, projected_payload, task.data_size
                    ):
                        direct_energy = self.network.E_tx(
                            task.data_size, node.distance_to(self.network.bs)
                        )
                        if not self._credit_allows(node_id, direct_energy):
                            continue
                        if node.consume(direct_energy):
                            completed += 1
                            decision_energy += direct_energy
                            results["completed"] += 1
                            results["energy"] += direct_energy
                            self.ecb.record_bill(node_id, direct_energy)
                            self._role_energy[node_id] += direct_energy
                        else:
                            continue
                    elif node.battery >= e_tx and ch_node.battery >= e_rx:
                        node.consume(e_tx)
                        ch_node.consume(e_rx)
                        completed += 1
                        previous_payload = cluster_payload[ch_id]
                        incremental_downstream = (
                            self._cluster_downstream_cost(
                                ch_id, previous_payload + task.data_size
                            )
                            - self._cluster_downstream_cost(ch_id, previous_payload)
                        )
                        decision_energy += e_tx + e_rx + incremental_downstream
                        cluster_payload[ch_id] += task.data_size
                        cluster_completed[ch_id].append(node_id)
                        self.ecb.record_bill(node_id, e_tx)
                        self.ecb.record_bill(ch_id, e_rx)
                        self._role_energy[node_id] += e_tx
                        self._role_energy[ch_id] += e_rx
                        results["energy"] += e_tx + e_rx
                    else:
                        continue

            dropped = len(tasks_at_node) - completed
            if action == 2 and node_id != ch_id:
                cluster_dropped[ch_id] += dropped
            else:
                results["dropped"] += dropped
            reward = (
                self.w_complete * completed
                - self.w_energy * decision_energy
                - self.w_drop * dropped
            )
            transitions.append(
                (state, action, self._apply_bill_reward(node_id, reward), node_id)
            )

        for ch_id, payload_bits in cluster_payload.items():
            ch_node = self.network.nodes[ch_id]
            if payload_bits <= 0 or not ch_node.is_alive:
                lost = len(cluster_completed[ch_id]) + cluster_dropped[ch_id]
                results["dropped"] += lost
                continue

            effective_ratio = self.aggregation_ratio if self.enable_aggregation else 1.0
            aggregated_bits = max(1, int(payload_bits * effective_ratio))
            downstream_cost = self._cluster_downstream_cost(ch_id, payload_bits)
            if ch_node.consume(downstream_cost):
                results["completed"] += len(cluster_completed[ch_id])
                results["dropped"] += cluster_dropped[ch_id]
                results["energy"] += downstream_cost
                self.ecb.record_bill(ch_id, downstream_cost)
                self._role_energy[ch_id] += downstream_cost
            else:
                results["dropped"] += len(cluster_completed[ch_id]) + cluster_dropped[ch_id]

        for state, action, reward, node_id in transitions:
            next_state = self.get_state(node_id, 0)
            done = not self.network.nodes[node_id].is_alive
            next_valid = self.get_valid_actions(node_id)
            self.d3qn.store_transition(
                state, action, reward, next_state, done, next_valid
            )

        loss = self.d3qn.train_step()
        if self.record_diagnostics:
            self.diagnostics.append({
                "round": int(current_round), "loss": None if loss is None else float(loss),
                "reference": self._budget_ref, "mean_credit": float(np.mean(self.ecb.credits)),
                "mean_bill": float(np.mean(self.ecb.bills)),
                "alive_nodes": int(sum(n.is_alive for n in self.network.nodes)),
                "completed": int(results["completed"]), "offered": int(len(tasks)),
                "mean_lambda": float(np.mean(self.ecb.lambda_rates)),
                "epsilon": float(self.d3qn.epsilon),
            })
        if current_round % self.exploration_decay_interval == 0:
            self.d3qn.decay_epsilon()

        self.round_in_cycle += 1
        if self.round_in_cycle >= self.T_L:
            cycle_harvests = np.mean(np.asarray(self._cycle_harvest_buffer), axis=0)
            self._cycle_update(cycle_harvests)
            results["energy"] += self._last_control_energy

        self.stats["total_tasks"] += len(tasks)
        self.stats["completed_tasks"] += results["completed"]
        self.stats["dropped_tasks"] += results["dropped"]
        self.stats["total_energy_consumed"] += results["energy"]
        return results

    def _charge_control_overhead(self) -> float:
        """Charge the per-cycle ECB control plane to the radio energy model.

        Each alive node receives one downlink quota/rate message and sends one
        uplink bill/completion message to the BS. The energy is deducted from
        node batteries and accumulated separately so its share of the total
        consumption can be reported.
        """
        if not self.enable_control_overhead:
            return 0.0

        ctrl_bits = self.control_msg_bits
        ctrl_energy = 0.0
        for node in self.network.nodes:
            if not node.is_alive:
                continue
            d_bs = node.distance_to(self.network.bs)
            e = self.network.E_rx(ctrl_bits) + self.network.E_tx(ctrl_bits, d_bs)
            if node.consume(e):
                ctrl_energy += e
                self.ecb.record_bill(node.id, e)
        self.stats["control_energy"] += ctrl_energy
        return ctrl_energy

    def _cycle_update(self, cycle_harvests: np.ndarray):
        self._last_control_energy = self._charge_control_overhead()

        if self.current_cycle == 0:
            # Bootstrap the predictor from the first observed cycle.
            self.harvest_estimates = cycle_harvests.copy()
        else:
            self.harvest_estimates = (
                self.beta * self.harvest_estimates + (1 - self.beta) * cycle_harvests
            )
        if self.enable_feedback:
            self.ecb.update_rates(self.current_cycle)

            # Preserve the settled cycle's signed balance for the next HHO
            # call. New quota commitment clears bills, so reading bills during
            # the later clustering call would make this term identically zero.
            self._last_credit_balance_ratio = (
                (self.ecb.credits - self.ecb.bills)
                / (self.ecb.credits + 1e-6)
            )
        else:
            # The NoFeedback ablation removes every bill-driven control path:
            # rates stay at their base values and HHO receives no overdraft
            # signal. Bills are retained only for the diagnostic below.
            self.ecb.lambda_rates.fill(self.ecb.lambda_0)
            self.ecb.mu_rates.fill(self.ecb.mu_0)
            self._last_credit_balance_ratio.fill(1.0)

        deviation = np.mean(
            np.abs(self.ecb.bills - self.ecb.credits) / (self.ecb.credits + 1e-6)
        )
        self.stats["budget_deviation_sum"] += float(deviation)
        self.stats["budget_deviation_count"] += 1

        base_credit = np.maximum(self.harvest_estimates * self.T_L, 1e-6)
        role_multiplier = np.ones(self.network.N)
        if self.enable_role_credit:
            for ch_id in self.cluster_heads:
                role_multiplier[ch_id] = self.ch_credit_multiplier
            active_relays = self._role_energy > np.percentile(self._role_energy, 65)
            role_multiplier[active_relays] = np.maximum(
                role_multiplier[active_relays], self.relay_credit_multiplier
            )

        if self.enable_feedback:
            # --- Adaptive fair-share, lifetime-pacing credit (closed loop) ---
            # (1) A network-wide reference budget tracks realized per-node
            #     consumption via a damped EWMA, so the quota lives on the true
            #     bill scale and follows non-stationary harvest/load shifts.
            active = self.ecb.bills > 0
            mean_bill = (
                float(np.mean(self.ecb.bills[active]))
                if np.any(active)
                else 1e-9
            )
            mean_bill = max(mean_bill, 1e-9)
            if self._budget_ref is None:
                self._budget_ref = mean_bill
            else:
                self._budget_ref = (
                    (1.0 - self.ecb.eta_c) * self._budget_ref
                    + self.ecb.eta_c * mean_bill
                )
            # (2) Battery-health tilt: depleting nodes get a tighter quota, so
            #     overspending them raises the penalty rate and the router shifts
            #     load to healthier nodes; CHs/relays keep role headroom.
            batteries = np.array([n.battery for n in self.network.nodes])
            b_max = self.network.nodes[0].B_max if self.network.nodes else 1.0
            health = 0.5 + np.clip(batteries / b_max, 0.0, 1.0)  # in [0.5, 1.5]
            # Share boost keeps pacing from chronically under-serving nodes
            # whose instantaneous load exceeds the network mean.
            new_credits = (
                self._budget_ref * self.credit_share_boost * health * role_multiplier
            )
            new_credits = np.clip(
                new_credits,
                0.25 * self._budget_ref,
                8.0 * self._budget_ref,
            )
        else:
            # Ablation (NoFeedback): no closed-loop feedback -- the quota is reset
            # to the harvest-anchored allocation every cycle (original behaviour).
            # With credit pacing enabled this mis-scaled quota becomes a hard
            # delivery cap and yields a measurable physical penalty.
            new_credits = base_credit * role_multiplier
            # Keep a tiny positive floor so diagnostics stay finite, but stay
            # orders of magnitude below radio cycle bills (~1e-3..1e-2 J).
            new_credits = np.maximum(new_credits, 1e-9)

        self.ecb.update_credits(new_credits)
        self.ecb.bills = np.zeros(self.network.N)

        self.current_cycle += 1
        self.round_in_cycle = 0
        self._cycle_harvest_buffer = []
        self._role_energy = np.zeros(self.network.N)
