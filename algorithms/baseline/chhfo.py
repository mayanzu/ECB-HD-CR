"""
CHHFO: Collaborative Harris Hawk Optimization + Fuzzy Logic
Reference: Hu et al. (2024) PLOS ONE

Pure metaheuristic approach without DRL
HHO for cluster head selection + fuzzy logic for routing decisions
"""

import numpy as np
from core.hho import HarrisHawksOptimizer
from .base_algorithm import BaseAlgorithm


class FuzzyLogicSystem:
    """Simple fuzzy logic for routing decisions"""

    def __init__(self):
        pass

    def fuzzify_battery(self, battery_level: float) -> dict:
        """Fuzzify battery level into {low, medium, high}"""
        fuzzy = {}
        # Low: [0, 0.3]
        if battery_level <= 0.3:
            fuzzy['low'] = 1.0
        elif battery_level <= 0.5:
            fuzzy['low'] = (0.5 - battery_level) / 0.2
        else:
            fuzzy['low'] = 0.0

        # Medium: [0.3, 0.7]
        if battery_level <= 0.3:
            fuzzy['medium'] = 0.0
        elif battery_level <= 0.5:
            fuzzy['medium'] = (battery_level - 0.3) / 0.2
        elif battery_level <= 0.7:
            fuzzy['medium'] = (0.7 - battery_level) / 0.2
        else:
            fuzzy['medium'] = 0.0

        # High: [0.5, 1.0]
        if battery_level <= 0.5:
            fuzzy['high'] = 0.0
        elif battery_level <= 0.7:
            fuzzy['high'] = (battery_level - 0.5) / 0.2
        else:
            fuzzy['high'] = 1.0

        return fuzzy

    def fuzzify_distance(self, distance: float, max_distance: float) -> dict:
        """Fuzzify distance into {near, medium, far}"""
        norm_dist = distance / max_distance
        fuzzy = {}

        # Near: [0, 0.3]
        fuzzy['near'] = max(0, min(1, (0.3 - norm_dist) / 0.3))

        # Medium: [0.2, 0.6]
        if norm_dist <= 0.2:
            fuzzy['medium'] = 0.0
        elif norm_dist <= 0.4:
            fuzzy['medium'] = (norm_dist - 0.2) / 0.2
        elif norm_dist <= 0.6:
            fuzzy['medium'] = (0.6 - norm_dist) / 0.2
        else:
            fuzzy['medium'] = 0.0

        # Far: [0.5, 1.0]
        fuzzy['far'] = max(0, min(1, (norm_dist - 0.5) / 0.5))

        return fuzzy

    def infer_should_transmit(self, battery_fuzzy: dict,
                             distance_fuzzy: dict) -> float:
        """
        Fuzzy inference: should node transmit?

        Rules:
        1. IF battery=high AND distance=near THEN transmit=high
        2. IF battery=high AND distance=medium THEN transmit=medium
        3. IF battery=medium AND distance=near THEN transmit=medium
        4. IF battery=low OR distance=far THEN transmit=low
        """
        rules = []

        # Rule 1
        rules.append(min(battery_fuzzy.get('high', 0),
                        distance_fuzzy.get('near', 0)) * 1.0)

        # Rule 2
        rules.append(min(battery_fuzzy.get('high', 0),
                        distance_fuzzy.get('medium', 0)) * 0.7)

        # Rule 3
        rules.append(min(battery_fuzzy.get('medium', 0),
                        distance_fuzzy.get('near', 0)) * 0.7)

        # Rule 4
        rules.append(max(battery_fuzzy.get('low', 0),
                        distance_fuzzy.get('far', 0)) * 0.2)

        # Defuzzify using weighted average
        return max(rules)


class CHHFO(BaseAlgorithm):
    """Collaborative HHO + Fuzzy Logic for WSN"""

    def __init__(self, network, config: dict):
        super().__init__(network, config)

        # HHO for cluster head selection
        self.hho = HarrisHawksOptimizer(
            dim=network.N,
            pop_size=config.get('hho_pop_size', 30),
            max_iter=config.get('hho_iter', 50),
            lb=0.0,
            ub=1.0
        )

        # Fuzzy logic system
        self.fuzzy = FuzzyLogicSystem()

        # Clustering parameters
        self.num_clusters = max(3, network.N // 20)  # ~5% as CHs
        self.cluster_heads = []
        self.cluster_assignment = {}

        # Re-cluster every K rounds
        self.K_reoptimize = config.get('K_reoptimize', 50)
        self.transmit_threshold = config.get('fuzzy_transmit_threshold', 0.2)

    def initialize(self):
        """Initialize CHHFO"""
        self._optimize_cluster_heads()

    def _optimize_cluster_heads(self):
        """Use HHO to select optimal cluster heads"""

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
            Fitness: minimize total energy consumption
            solution: array of [0,1] values, top K are CHs
            """
            # Select top K as cluster heads
            ch_indices = sorted(
                alive_ids, key=lambda node_id: solution[node_id]
            )[-self.num_clusters:]

            # Compute total intra-cluster distance
            total_dist = 0.0

            for i in range(self.network.N):
                if not self.network.nodes[i].is_alive:
                    continue
                if i in ch_indices:
                    # CH transmits to BS
                    node = self.network.nodes[i]
                    dist = node.distance_to(self.network.bs)
                    total_dist += dist ** 2
                else:
                    # Member transmits to nearest CH
                    node = self.network.nodes[i]
                    min_dist = float('inf')

                    for ch_id in ch_indices:
                        ch_node = self.network.nodes[ch_id]
                        dist = node.distance_to(ch_node)
                        if dist < min_dist:
                            min_dist = dist

                    total_dist += min_dist ** 2

            return total_dist

        # Run HHO optimization
        best_solution, _ = self.hho.optimize(fitness_function)

        # Extract cluster heads
        self.cluster_heads = sorted(
            alive_ids, key=lambda node_id: best_solution[node_id]
        )[-self.num_clusters:]
        if not self.cluster_heads:
            self.cluster_heads = [
                max(alive_ids, key=lambda i: self.network.nodes[i].battery)
            ]

        # Assign nodes to clusters
        self._assign_clusters()

    def _assign_clusters(self):
        """Assign each node to nearest cluster head"""
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
                    ch_node = self.network.nodes[ch_id]
                    dist = node.distance_to(ch_node)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ch = ch_id

                self.cluster_assignment[node_id] = nearest_ch

    def schedule_round(self, tasks, current_round: int,
                      solar_harvests: np.ndarray):
        """Schedule using CHHFO"""
        self.stats['rounds_executed'] += 1

        # Re-optimize cluster structure periodically
        if (
            (current_round > 0 and current_round % self.K_reoptimize == 0)
            or any(not self.network.nodes[ch].is_alive for ch in self.cluster_heads)
        ):
            self._optimize_cluster_heads()

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
            if ch_id is not None:
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

            # Members transmit to CH using fuzzy logic
            for task in tasks_in_cluster:
                source_node = self.network.nodes[task.source_node]

                if not source_node.is_alive:
                    cluster_dropped += 1
                    continue

                # Fuzzy decision: should transmit?
                battery_fuzzy = self.fuzzy.fuzzify_battery(
                    source_node.battery / source_node.B_max
                )
                distance_fuzzy = self.fuzzy.fuzzify_distance(
                    source_node.distance_to(ch_node),
                    100.0  # Max distance in field
                )

                transmit_score = self.fuzzy.infer_should_transmit(
                    battery_fuzzy, distance_fuzzy
                )

                # Transmit when fuzzy confidence reaches the configured threshold.
                if transmit_score >= self.transmit_threshold:
                    if task.source_node == ch_id:
                        # CH's own task
                        aggregated_data += task.data_size
                        cluster_completed += 1
                    else:
                        # Member transmits to CH
                        dist = source_node.distance_to(ch_node)
                        E_tx = self.network.E_tx(task.data_size, dist)
                        E_rx = self.network.E_rx(task.data_size)

                        # Check both endpoints before mutating either battery;
                        # otherwise a failed CH reception would strand source
                        # energy without a delivered packet.
                        if (source_node.battery >= E_tx and
                                ch_node.battery >= E_rx):
                            source_node.consume(E_tx)
                            ch_node.consume(E_rx)
                            aggregated_data += task.data_size
                            cluster_energy += E_tx + E_rx
                            cluster_completed += 1
                        else:
                            cluster_dropped += 1
                else:
                    cluster_dropped += 1

            # CH aggregates and transmits to BS
            if aggregated_data > 0:
                E_DA = aggregated_data * self.network.E_DA
                dist_to_bs = ch_node.distance_to(self.network.bs)
                E_tx_bs = self.network.E_tx(aggregated_data, dist_to_bs)

                if ch_node.consume(E_DA + E_tx_bs):
                    cluster_energy += E_DA + E_tx_bs
                    results['completed'] += cluster_completed
                    results['dropped'] += cluster_dropped
                else:
                    # CH failed: only this cluster's aggregated tasks are lost.
                    results['dropped'] += cluster_completed + cluster_dropped
            else:
                results['dropped'] += cluster_dropped

            results['energy'] += cluster_energy

        # Update statistics
        self.stats['total_tasks'] += len(tasks)
        self.stats['completed_tasks'] += results['completed']
        self.stats['dropped_tasks'] += results['dropped']
        self.stats['total_energy_consumed'] += results['energy']

        return results
