"""
EH-WSN Network Simulator
Energy Harvesting Wireless Sensor Network simulation framework
"""

import numpy as np
from typing import List, Tuple, Dict


class SensorNode:
    """Single sensor node with energy harvesting capability"""

    def __init__(self, node_id: int, position: Tuple[float, float],
                 E_init: float = 0.5, B_max: float = 2.0):
        self.id = node_id
        self.x, self.y = position
        self.E_init = E_init
        self.B_max = B_max

        # Energy state
        self.battery = E_init
        self.is_alive = True
        self.death_round = None

        # Statistics
        self.total_harvest = 0.0
        self.total_consumed = 0.0
        self.rounds_alive = 0

    def harvest(self, amount: float):
        """Harvest energy (with saturation)"""
        if getattr(self, '_skip_next_harvest', False):
            self._skip_next_harvest = False
            return 0.0
        before = self.battery
        self.battery = min(self.battery + amount, self.B_max)
        actual = self.battery - before
        self.total_harvest += actual
        return actual

    def consume(self, amount: float) -> bool:
        """Consume energy, return True if successful"""
        if amount < 0:
            raise ValueError("energy consumption cannot be negative")
        if self.battery >= amount:
            self.battery -= amount
            self.total_consumed += amount
            return True
        # A physically infeasible action is rejected without spending energy.
        # Background consumption may still deplete the node later in the round.
        return False

    def consume_up_to(self, amount: float) -> float:
        """Consume up to ``amount`` and return the energy actually spent."""
        if amount < 0:
            raise ValueError("energy consumption cannot be negative")
        actual = min(self.battery, amount)
        self.battery -= actual
        self.total_consumed += actual
        return actual

    def check_alive(self, current_round: int) -> bool:
        """Check if node is still alive"""
        # Use small threshold to handle floating point precision
        if self.is_alive and self.battery <= 1e-6:  # 0.000001 J threshold
            self.is_alive = False
            self.death_round = current_round
        return self.is_alive

    def distance_to(self, other) -> float:
        """Euclidean distance to another node or BS"""
        if isinstance(other, SensorNode):
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        elif hasattr(other, 'x') and hasattr(other, 'y'):
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        else:  # tuple (x, y)
            return np.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)


class BaseStation:
    """Base station with unlimited energy"""

    def __init__(self, position: Tuple[float, float] = (50, 50)):
        self.x, self.y = position

    def distance_to(self, node: SensorNode) -> float:
        return node.distance_to((self.x, self.y))


class EH_WSN:
    """Energy Harvesting WSN network"""

    def __init__(self, N: int = 100, M: float = 100.0,
                 bs_position: Tuple[float, float] = (50, 50),
                 seed: int = 42):
        self.N = N
        self.M = M
        self.bs = BaseStation(bs_position)

        # Create nodes with random positions. A dedicated topology RNG is used
        # (instead of reseeding the global stream to a fixed constant), so that
        # distinct seeds produce genuinely different deployments. This makes
        # independent runs sample independent node layouts rather than reusing
        # one fixed topology, which is required for meaningful seed variance.
        self.topology_seed = seed
        rng = np.random.RandomState(seed)
        self.nodes = []
        for i in range(N):
            x = rng.uniform(0, M)
            y = rng.uniform(0, M)
            node = SensorNode(i, (x, y))
            self.nodes.append(node)

        # Energy model parameters (First-order radio model)
        self.E_elec = 50e-9  # 50 nJ/bit
        self.E_fs = 10e-12   # 10 pJ/bit/m^2 (free space)
        self.E_mp = 0.0013e-12  # 0.0013 pJ/bit/m^4 (multipath)
        self.E_DA = 5e-9     # 5 nJ/bit (data aggregation)
        self.d0 = np.sqrt(self.E_fs / self.E_mp)  # Threshold distance

        # Statistics
        self.current_round = 0
        self.alive_count = N
        self.fnd = None  # First Node Death
        self.hnd = None  # Half Nodes Death
        self.lnd = None  # Last Node Death

    def E_tx(self, data_bits: int, distance: float) -> float:
        """Energy cost for transmitting data_bits over distance"""
        if distance < self.d0:
            return data_bits * (self.E_elec + self.E_fs * distance**2)
        else:
            return data_bits * (self.E_elec + self.E_mp * distance**4)

    def E_rx(self, data_bits: int) -> float:
        """Energy cost for receiving data_bits"""
        return data_bits * self.E_elec

    def update_alive_count(self):
        """Update network lifetime statistics"""
        alive = sum(1 for node in self.nodes if node.is_alive)

        if alive < self.N and self.fnd is None:
            self.fnd = self.current_round
        if alive < self.N / 2 and self.hnd is None:
            self.hnd = self.current_round
        if alive == 0 and self.lnd is None:
            self.lnd = self.current_round

        self.alive_count = alive
        return alive

    def get_alive_nodes(self) -> List[SensorNode]:
        """Return list of alive nodes"""
        return [node for node in self.nodes if node.is_alive]

    def reset(self):
        """Reset network to initial state"""
        self.current_round = 0
        self.alive_count = self.N
        self.fnd = None
        self.hnd = None
        self.lnd = None

        for node in self.nodes:
            node.battery = node.E_init
            node.is_alive = True
            node.death_round = None
            node.total_harvest = 0.0
            node.total_consumed = 0.0
            node.rounds_alive = 0
