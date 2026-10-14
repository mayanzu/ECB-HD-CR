"""
Task Generation and Management
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """Data sensing task"""
    task_id: int
    source_node: int
    generation_time: int
    data_size: int  # bits
    priority: int   # 1-5, higher = more important
    deadline: Optional[int] = None
    completed: bool = False
    completion_time: Optional[int] = None


class TaskGenerator:
    """Generate tasks for sensor nodes"""

    def __init__(self, num_nodes: int, arrival_rate: float = 0.5,
                 data_size_range: tuple = (100, 500), seed: Optional[int] = None):
        """
        Args:
            num_nodes: Number of sensor nodes
            arrival_rate: Poisson arrival rate (tasks per node per round)
            data_size_range: (min, max) data size in bits
        """
        self.num_nodes = num_nodes
        self.arrival_rate = arrival_rate
        self.data_size_min, self.data_size_max = data_size_range
        self.task_counter = 0
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def generate_tasks_for_round(self, current_round: int) -> List[Task]:
        """
        Generate tasks for all nodes in current round

        Returns:
            List of new tasks
        """
        tasks = []

        for node_id in range(self.num_nodes):
            # Poisson arrivals per node
            num_tasks = self.rng.poisson(self.arrival_rate)

            for _ in range(num_tasks):
                task = Task(
                    task_id=self.task_counter,
                    source_node=node_id,
                    generation_time=current_round,
                    data_size=self.rng.randint(
                        self.data_size_min, self.data_size_max + 1
                    ),
                    priority=self.rng.randint(1, 6),  # 1-5
                )
                tasks.append(task)
                self.task_counter += 1

        return tasks


class TaskScheduler:
    """Base class for task scheduling algorithms"""

    def __init__(self, network):
        self.network = network
        self.completed_tasks = []
        self.dropped_tasks = []
        self.total_tasks = 0

    def schedule_round(self, tasks: List[Task], current_round: int):
        """
        Schedule tasks for current round
        Must be implemented by subclasses
        """
        raise NotImplementedError

    def get_statistics(self) -> dict:
        """Get task completion statistics"""
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': len(self.completed_tasks),
            'dropped_tasks': len(self.dropped_tasks),
            'completion_rate': (len(self.completed_tasks) / self.total_tasks
                               if self.total_tasks > 0 else 0.0)
        }


class SimpleDirectScheduler(TaskScheduler):
    """
    Simple baseline scheduler: each node transmits directly to BS
    """

    def schedule_round(self, tasks: List[Task], current_round: int):
        """Schedule tasks using direct transmission"""
        self.total_tasks += len(tasks)

        for task in tasks:
            node = self.network.nodes[task.source_node]

            if not node.is_alive:
                self.dropped_tasks.append(task)
                continue

            # Calculate energy cost for direct transmission to BS
            distance = node.distance_to(self.network.bs)
            E_tx = self.network.E_tx(task.data_size, distance)

            # Try to transmit
            if node.consume(E_tx):
                task.completed = True
                task.completion_time = current_round
                self.completed_tasks.append(task)
            else:
                # Insufficient energy
                self.dropped_tasks.append(task)


class ClusterScheduler(TaskScheduler):
    """
    Cluster-based scheduler with data aggregation
    """

    def __init__(self, network, cluster_heads: List[int]):
        super().__init__(network)
        self.cluster_heads = cluster_heads
        self.cluster_assignment = self._assign_clusters()

    def _assign_clusters(self) -> dict:
        """Assign each node to nearest cluster head"""
        assignment = {}

        for node in self.network.nodes:
            if node.id in self.cluster_heads:
                assignment[node.id] = node.id
            else:
                # Find nearest cluster head
                min_dist = float('inf')
                nearest_ch = self.cluster_heads[0]

                for ch_id in self.cluster_heads:
                    ch_node = self.network.nodes[ch_id]
                    dist = node.distance_to(ch_node)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ch = ch_id

                assignment[node.id] = nearest_ch

        return assignment

    def schedule_round(self, tasks: List[Task], current_round: int):
        """Schedule tasks using clustering"""
        self.total_tasks += len(tasks)

        # Group tasks by cluster
        cluster_tasks = {ch_id: [] for ch_id in self.cluster_heads}

        for task in tasks:
            node = self.network.nodes[task.source_node]
            if not node.is_alive:
                self.dropped_tasks.append(task)
                continue

            ch_id = self.cluster_assignment[task.source_node]
            cluster_tasks[ch_id].append(task)

        # Process each cluster
        for ch_id, tasks_in_cluster in cluster_tasks.items():
            if not tasks_in_cluster:
                continue

            ch_node = self.network.nodes[ch_id]
            if not ch_node.is_alive:
                self.dropped_tasks.extend(tasks_in_cluster)
                continue

            # Member nodes transmit to CH
            aggregated_data = 0
            for task in tasks_in_cluster:
                if task.source_node == ch_id:
                    # CH's own task
                    aggregated_data += task.data_size
                    task.completed = True
                else:
                    # Member node transmits to CH
                    member = self.network.nodes[task.source_node]
                    dist_to_ch = member.distance_to(ch_node)
                    E_tx = self.network.E_tx(task.data_size, dist_to_ch)
                    E_rx = self.network.E_rx(task.data_size)

                    if member.battery >= E_tx and ch_node.battery >= E_rx:
                        member.consume(E_tx)
                        ch_node.consume(E_rx)
                        aggregated_data += task.data_size
                        task.completed = True
                    else:
                        self.dropped_tasks.append(task)
                        continue

            # CH aggregates and transmits to BS
            if aggregated_data > 0:
                E_DA = aggregated_data * self.network.E_DA
                dist_to_bs = ch_node.distance_to(self.network.bs)
                E_tx_bs = self.network.E_tx(aggregated_data, dist_to_bs)

                if ch_node.consume(E_DA + E_tx_bs):
                    # Successfully transmitted
                    for task in tasks_in_cluster:
                        if task.completed:
                            task.completion_time = current_round
                            self.completed_tasks.append(task)
                else:
                    # CH ran out of energy
                    self.dropped_tasks.extend(
                        [t for t in tasks_in_cluster if t.completed]
                    )
