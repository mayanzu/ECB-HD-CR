"""
D3QN-EH: Single-timescale D3QN baseline
Pure DRL approach without hierarchical structure
"""

import sys

import numpy as np
from core.d3qn import D3QN_Agent
from .base_algorithm import BaseAlgorithm


class D3QN_EH(BaseAlgorithm):
    """Single-timescale D3QN for EH-WSN scheduling"""

    def __init__(self, network, config: dict):
        super().__init__(network, config)

        # D3QN agent (shared across all nodes)
        # State: [battery, predicted_harvest, current_batch_size, time_of_day]
        state_dim = 4
        # Actions: [drop, smallest-packet first, highest-priority first, FIFO].
        # Every non-drop action attempts the complete current-round batch, and
        # every delivered packet pays the full first-order radio cost.  The
        # ordering only matters when residual energy cannot serve every task.
        action_dim = 4

        self.d3qn = D3QN_Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.get('lr', 0.001),
            gamma=config.get('gamma', 0.9),
            epsilon_start=config.get(
                'd3qn_eh_epsilon_start',
                config.get('epsilon_start', 1.0),
            ),
            epsilon_end=config.get(
                'd3qn_eh_epsilon_end',
                config.get('epsilon_end', 0.01),
            ),
            epsilon_decay=config.get(
                'd3qn_eh_epsilon_decay',
                config.get('epsilon_decay', 0.995),
            ),
            buffer_capacity=config.get('buffer_capacity', 10000),
            batch_size=config.get('batch_size', 64),
            target_update_freq=config.get('target_update_freq', 100),
        )
        self.exploration_decay_interval = config.get(
            'exploration_decay_interval', 10
        )
        self.energy_floor = config.get('physical_energy_floor', 0.002)

        # Harvest prediction (simple moving average)
        self.window_size = 20
        self.harvest_history = [[] for _ in range(network.N)]
        self.harvest_predictions = np.zeros(network.N)

        # Reward weights
        self.w_complete = config.get(
            'd3qn_eh_w_complete',
            config.get('w_complete', 10.0),
        )
        self.w_energy = config.get(
            'd3qn_eh_w_energy',
            config.get('w_energy', 2.0),
        )
        self.w_delay = config.get(
            'd3qn_eh_w_delay',
            config.get('w_delay', 1.0),
        )

        # Training mode
        self.training = True

    def initialize(self):
        """Initialize D3QN-EH"""
        self.harvest_predictions = np.ones(self.network.N) * 0.1

    def get_state(self, node_id: int, queue_size: int,
                  current_round: int) -> np.ndarray:
        """Get state representation"""
        node = self.network.nodes[node_id]

        # Diurnal pattern (0-1 representing time of day)
        time_of_day = (current_round % 1440) / 1440.0

        state = np.array([
            node.battery / node.B_max,
            self.harvest_predictions[node_id],
            min(queue_size / 10.0, 1.0),
            time_of_day
        ])

        return state

    def compute_reward(self, completed_count: int, energy_spent: float,
                      dropped_count: int, node_alive: bool) -> float:
        """Compute reward"""
        reward = (self.w_complete * completed_count -
                  self.w_energy * energy_spent -
                  self.w_delay * dropped_count)

        # Large penalty for node death
        if not node_alive:
            reward -= 100.0

        return reward

    def get_valid_actions(self, node_id: int) -> np.ndarray:
        """Allow idling only when the node is physically near depletion."""
        node = self.network.nodes[node_id]
        if not node.is_alive or node.battery < self.energy_floor:
            return np.array([True, False, False, False], dtype=bool)
        return np.array([False, True, True, True], dtype=bool)

    def schedule_round(self, tasks, current_round: int,
                      solar_harvests: np.ndarray):
        """Schedule tasks using single-timescale D3QN"""
        self.stats['rounds_executed'] += 1

        # Update harvest predictions
        for i in range(self.network.N):
            self.harvest_history[i].append(solar_harvests[i])
            if len(self.harvest_history[i]) > self.window_size:
                self.harvest_history[i].pop(0)

            if len(self.harvest_history[i]) > 0:
                self.harvest_predictions[i] = np.mean(
                    self.harvest_history[i]
                )

        # Apply harvests
        for i, node in enumerate(self.network.nodes):
            if node.is_alive:
                node.harvest(solar_harvests[i])

        # Group tasks by node
        node_tasks = {i: [] for i in range(self.network.N)}
        for task in tasks:
            node_tasks[task.source_node].append(task)

        results = {'completed': 0, 'dropped': 0, 'energy': 0.0}

        # Process each node
        for node_id in range(self.network.N):
            node = self.network.nodes[node_id]
            if not node.is_alive:
                results['dropped'] += len(node_tasks[node_id])
                continue

            tasks_at_node = node_tasks[node_id]
            if len(tasks_at_node) == 0:
                continue

            # Get state
            state = self.get_state(node_id, len(tasks_at_node),
                                  current_round)

            # Select action
            action = self.d3qn.select_action(
                state,
                valid_actions=self.get_valid_actions(node_id),
                eval_mode=not self.training,
            )

            if action == 0:
                results['dropped'] += len(tasks_at_node)
                completed_count = 0
                total_energy = 0.0
            else:
                completed_count = 0
                total_energy = 0.0
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
                    energy_cost = self.network.E_tx(
                        task.data_size, dist_to_bs
                    )

                    if node.consume(energy_cost):
                        completed_count += 1
                        total_energy += energy_cost
                    else:
                        # A smaller later task may still be feasible.  Failed
                        # attempts spend no energy under SensorNode.consume().
                        continue

                results['completed'] += completed_count
                results['dropped'] += len(tasks_at_node) - completed_count
                results['energy'] += total_energy
                self.record_energy(node_id, total_energy)

            reward = self.compute_reward(
                completed_count,
                total_energy,
                len(tasks_at_node) - completed_count,
                node.is_alive
            )

            # Get next state
            next_state = self.get_state(node_id, 0, current_round + 1)

            # Store transition
            if self.training:
                self.d3qn.store_transition(
                    state, action, reward, next_state,
                    done=not node.is_alive,
                    next_valid_actions=self.get_valid_actions(node_id),
                )

        # Train D3QN
        if self.training:
            loss = self.d3qn.train_step()
            if current_round % self.exploration_decay_interval == 0:
                self.d3qn.decay_epsilon()

        # Update statistics
        self.stats['total_tasks'] += len(tasks)
        self.stats['completed_tasks'] += results['completed']
        self.stats['dropped_tasks'] += results['dropped']
        self.stats['total_energy_consumed'] += results['energy']

        return results
