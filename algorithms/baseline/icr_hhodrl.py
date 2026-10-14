"""
ICR-HHODRL: Intelligent Clustering and Routing using
Quantum-Inspired HHO + DRL
Reference: Wang et al. (2025) Ad Hoc Networks

Quantum-inspired HHO for clustering (long timescale)
DRL for routing (short timescale)
Unidirectional coupling: HHO → DRL (no feedback)
"""

import numpy as np
from core.d3qn import D3QN_Agent
from core.hho import HarrisHawksOptimizer
from .base_algorithm import BaseAlgorithm


class QuantumInspiredHHO(HarrisHawksOptimizer):
    """
    Quantum-inspired HHO with rotation gate
    """

    def __init__(self, dim: int, pop_size: int = 30, max_iter: int = 100,
                 lb: float = 0.0, ub: float = 1.0):
        super().__init__(dim, pop_size, max_iter, lb, ub)

        # Quantum rotation angle
        self.theta = 0.05 * np.pi

    def quantum_rotation(self, position: np.ndarray) -> np.ndarray:
        """
        Apply quantum rotation gate for exploration
        """
        # Rotation matrix
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)

        # Apply rotation with probability
        if np.random.random() < 0.3:
            # Rotate position
            rotated = position * cos_t + np.random.randn(self.dim) * sin_t
            rotated = np.clip(rotated, self.lb, self.ub)
            return rotated
        else:
            return position

    def optimize(self, fitness_func):
        """Override optimize to add quantum rotation"""
        # Run standard HHO
        best_pos, best_fit = super().optimize(fitness_func)

        # Apply quantum rotation to final solution
        for _ in range(10):  # Quantum refinement
            candidate = self.quantum_rotation(best_pos)
            candidate_fit = fitness_func(candidate)

            if candidate_fit < best_fit:
                best_pos = candidate
                best_fit = candidate_fit

        return best_pos, best_fit


class ICR_HHODRL(BaseAlgorithm):
    """
    Intelligent Clustering and Routing Protocol
    QHHO for clustering + DRL for routing
    """

    def __init__(self, network, config: dict):
        super().__init__(network, config)

        # Quantum-inspired HHO for clustering
        self.qhho = QuantumInspiredHHO(
            dim=network.N,
            pop_size=config.get('hho_pop_size', 30),
            max_iter=config.get('hho_iter', 50),
            lb=0.0,
            ub=1.0
        )

        # DRL router for path selection
        # State: [src_battery, dst_battery, distance, hop_count]
        state_dim = 4
        # Implemented actions: [direct_to_bs, relay_via_ch].
        action_dim = 2

        self.drl_router = D3QN_Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.get('lr', 0.001),
            gamma=config.get('gamma', 0.9),
            epsilon_start=config.get('epsilon_start', 0.3),
            epsilon_end=config.get('epsilon_end', 0.01),
            epsilon_decay=config.get('epsilon_decay', 0.99),
            buffer_capacity=config.get('buffer_capacity', 10000),
            batch_size=config.get('batch_size', 64),
            target_update_freq=config.get('target_update_freq', 100),
        )
        self.exploration_decay_interval = config.get(
            'exploration_decay_interval', 10
        )
        self.aggregation_ratio = config.get('aggregation_ratio', 0.35)

        # Clustering parameters
        self.num_clusters = max(3, network.N // 20)
        self.cluster_heads = []
        self.cluster_assignment = {}

        # Re-cluster period
        self.K_reoptimize = config.get('K_reoptimize', 100)

        # Training mode
        self.training = True

    def initialize(self):
        """Initialize ICR-HHODRL"""
        self._quantum_optimize_clusters()

    def _quantum_optimize_clusters(self):
        """Use Quantum HHO to optimize cluster structure"""

        alive_ids = [node.id for node in self.network.nodes if node.is_alive]
        if not alive_ids:
            self.cluster_heads = []
            self.cluster_assignment = {}
            return
        if len(alive_ids) <= self.num_clusters:
            self.cluster_heads = alive_ids
            self._assign_clusters()
            return

        def fitness_function(solution):
            """
            Multi-objective fitness:
            - Minimize intra-cluster distance
            - Balance cluster sizes
            - Maximize CH residual energy
            """
            ch_indices = sorted(
                alive_ids, key=lambda node_id: solution[node_id]
            )[-self.num_clusters:]

            # Intra-cluster distance
            intra_dist = 0.0
            cluster_sizes = {ch: 0 for ch in ch_indices}

            for i in range(self.network.N):
                node = self.network.nodes[i]

                if not node.is_alive:
                    continue

                if i in ch_indices:
                    dist = node.distance_to(self.network.bs)
                    intra_dist += dist ** 2
                    cluster_sizes[i] = 1
                else:
                    min_dist = float('inf')
                    nearest_ch = ch_indices[0]

                    for ch_id in ch_indices:
                        ch_node = self.network.nodes[ch_id]
                        dist = node.distance_to(ch_node)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_ch = ch_id

                    intra_dist += min_dist ** 2
                    cluster_sizes[nearest_ch] += 1

            # Balance penalty
            sizes = list(cluster_sizes.values())
            balance_penalty = np.std(sizes) * 10

            # Energy penalty (prefer high-energy nodes as CHs)
            energy_penalty = 0.0
            for ch_id in ch_indices:
                ch_node = self.network.nodes[ch_id]
                energy_penalty += (1.0 - ch_node.battery / ch_node.B_max) * 5

            return intra_dist + balance_penalty + energy_penalty

        # Run quantum HHO
        best_solution, _ = self.qhho.optimize(fitness_function)

        # Extract cluster heads
        self.cluster_heads = sorted(
            alive_ids, key=lambda node_id: best_solution[node_id]
        )[-self.num_clusters:]
        self._assign_clusters()

    def _assign_clusters(self):
        """Assign nodes to clusters"""
        self.cluster_assignment = {}

        for node_id in range(self.network.N):
            if not self.network.nodes[node_id].is_alive:
                continue
            if node_id in self.cluster_heads:
                self.cluster_assignment[node_id] = node_id
            else:
                node = self.network.nodes[node_id]
                min_dist = float('inf')
                nearest_ch = self.cluster_heads[0]

                for ch_id in self.cluster_heads:
                    if not self.network.nodes[ch_id].is_alive:
                        continue

                    ch_node = self.network.nodes[ch_id]
                    dist = node.distance_to(ch_node)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ch = ch_id

                self.cluster_assignment[node_id] = nearest_ch

    def get_routing_state(self, src_node_id: int, dst_node_id: int,
                         hop_count: int) -> np.ndarray:
        """Get state for DRL routing decision"""
        src = self.network.nodes[src_node_id]
        dst = self.network.nodes[dst_node_id]

        state = np.array([
            src.battery / src.B_max,
            dst.battery / dst.B_max,
            src.distance_to(dst) / 100.0,  # Normalized
            min(hop_count / 5.0, 1.0)
        ])

        return state

    def schedule_round(self, tasks, current_round: int,
                      solar_harvests: np.ndarray):
        """Schedule using ICR-HHODRL"""
        self.stats['rounds_executed'] += 1

        # Re-optimize clusters periodically (no feedback from DRL)
        if (
            (current_round > 0 and current_round % self.K_reoptimize == 0)
            or any(not self.network.nodes[ch].is_alive for ch in self.cluster_heads)
        ):
            self._quantum_optimize_clusters()

        # Apply harvests
        for i, node in enumerate(self.network.nodes):
            if node.is_alive:
                node.harvest(solar_harvests[i])

        # Group tasks by cluster
        cluster_tasks = {ch_id: [] for ch_id in self.cluster_heads}

        for task in tasks:
            if not self.network.nodes[task.source_node].is_alive:
                continue

            ch_id = self.cluster_assignment.get(task.source_node)
            if ch_id is not None and self.network.nodes[ch_id].is_alive:
                cluster_tasks[ch_id].append(task)

        results = {'completed': 0, 'dropped': 0, 'energy': 0.0}

        # Process each cluster
        for ch_id, tasks_in_cluster in cluster_tasks.items():
            if not tasks_in_cluster:
                continue

            ch_node = self.network.nodes[ch_id]
            if not ch_node.is_alive:
                results['dropped'] += len(tasks_in_cluster)
                continue

            aggregated_data = 0
            cluster_energy = 0.0
            cluster_completed = 0
            cluster_dropped = 0

            # Members use DRL routing to CH
            for task in tasks_in_cluster:
                src_node = self.network.nodes[task.source_node]

                if not src_node.is_alive:
                    cluster_dropped += 1
                    continue

                # DRL routing decision
                state = self.get_routing_state(task.source_node, ch_id, 1)
                action = self.drl_router.select_action(
                    state, eval_mode=not self.training
                )

                # Action 0: direct to BS; action 1: via cluster head.

                if task.source_node == ch_id:
                    aggregated_data += task.data_size
                    cluster_completed += 1
                    continue

                if action == 0:
                    # A direct packet reaches the BS and must not be forwarded
                    # through the CH a second time.
                    dist = src_node.distance_to(self.network.bs)
                    e_tx = self.network.E_tx(task.data_size, dist)
                    if src_node.consume(e_tx):
                        cluster_energy += e_tx
                        results['completed'] += 1
                        reward = 10.0 - e_tx * 1000
                    else:
                        cluster_dropped += 1
                        reward = -1.0
                else:
                    # Via the implemented one-hop member-to-CH path.
                    dist = src_node.distance_to(ch_node)
                    e_tx = self.network.E_tx(task.data_size, dist)
                    e_rx = self.network.E_rx(task.data_size)
                    if src_node.battery >= e_tx and ch_node.battery >= e_rx:
                        src_node.consume(e_tx)
                        ch_node.consume(e_rx)
                        aggregated_data += task.data_size
                        cluster_energy += e_tx + e_rx
                        cluster_completed += 1
                        reward = 10.0 - (e_tx + e_rx) * 1000
                    else:
                        cluster_dropped += 1
                        reward = -1.0

                next_state = self.get_routing_state(task.source_node, ch_id, 2)
                if self.training:
                    self.drl_router.store_transition(
                        state, action, reward, next_state,
                        done=not src_node.is_alive,
                    )

            # CH transmits to BS
            if aggregated_data > 0:
                aggregated_bits = max(
                    1, int(aggregated_data * self.aggregation_ratio)
                )
                E_DA = aggregated_data * self.network.E_DA
                dist_to_bs = ch_node.distance_to(self.network.bs)
                E_tx_bs = self.network.E_tx(aggregated_bits, dist_to_bs)

                if ch_node.consume(E_DA + E_tx_bs):
                    cluster_energy += E_DA + E_tx_bs
                    results['completed'] += cluster_completed
                    results['dropped'] += cluster_dropped
                else:
                    results['dropped'] += cluster_completed + cluster_dropped
            else:
                results['dropped'] += cluster_dropped

            results['energy'] += cluster_energy

        # Train DRL router
        if self.training:
            self.drl_router.train_step()
            if current_round % self.exploration_decay_interval == 0:
                self.drl_router.decay_epsilon()

        # Update statistics
        self.stats['total_tasks'] += len(tasks)
        self.stats['completed_tasks'] += results['completed']
        self.stats['dropped_tasks'] += results['dropped']
        self.stats['total_energy_consumed'] += results['energy']

        return results
