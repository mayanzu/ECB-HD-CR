"""
Base class for all scheduling algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class BaseAlgorithm(ABC):
    """Base class for all scheduling algorithms"""

    def __init__(self, network, config: dict):
        self.network = network
        self.config = config
        self.name = self.__class__.__name__

        # Statistics
        self.stats = {
            'total_energy_consumed': 0.0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'rounds_executed': 0
        }

    @abstractmethod
    def initialize(self):
        """Initialize algorithm-specific components"""
        pass

    @abstractmethod
    def schedule_round(self, tasks: List, current_round: int,
                      solar_harvests: np.ndarray) -> Dict:
        """
        Schedule tasks for current round

        Args:
            tasks: List of Task objects
            current_round: Current round number
            solar_harvests: Energy harvested this round per node

        Returns:
            Dictionary with scheduling results
        """
        pass

    def record_energy(self, node_id: int, energy: float):
        """Record energy consumption"""
        self.stats['total_energy_consumed'] += energy

    def record_task(self, completed: bool):
        """Record task outcome"""
        self.stats['total_tasks'] += 1
        if completed:
            self.stats['completed_tasks'] += 1
        else:
            self.stats['dropped_tasks'] += 1

    def get_statistics(self) -> Dict:
        """Get algorithm statistics"""
        completion_rate = (self.stats['completed_tasks'] /
                          self.stats['total_tasks']
                          if self.stats['total_tasks'] > 0 else 0.0)

        return {
            **self.stats,
            'completion_rate': completion_rate,
            'avg_energy_per_round': (
                self.stats['total_energy_consumed'] /
                max(1, self.stats['rounds_executed'])
            )
        }

    def reset(self):
        """Reset statistics"""
        self.stats = {
            'total_energy_consumed': 0.0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'rounds_executed': 0
        }
