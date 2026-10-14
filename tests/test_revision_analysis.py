import itertools
import unittest
from types import SimpleNamespace
import numpy as np
from core.network import EH_WSN
from core.control_plane import CentralControlPlane
from algorithms.ecb_hd.ecb_mechanism import ECB_Mechanism
from data.task_model import Task
from analysis.optimality_bounds import bound_from_costs, WEIGHTS, ENERGY_REFERENCE_J


class ControlPlaneTests(unittest.TestCase):
    def test_report_command_and_topology_are_paid_and_billed(self):
        network = EH_WSN(N=1)
        ledger = ECB_Mechanism(1)
        algorithm = SimpleNamespace(ecb=ledger, enable_clustering=True)
        plane = CentralControlPlane(network, algorithm, {})
        node = network.nodes[0]
        before = node.battery
        charged, alive = plane.begin_round([Task(0, 0, 0, 300, 1)], 0, np.zeros(1))
        expected = network.E_tx(160, node.distance_to(network.bs)) + network.E_rx(128)
        self.assertAlmostEqual(charged, expected)
        self.assertAlmostEqual(before - node.battery, expected)
        self.assertAlmostEqual(ledger.bills[0], expected)
        self.assertEqual(sum(plane.messages.values()), 3)
        plane.end_round(alive)

    def test_harvest_occurs_once_before_telemetry(self):
        network = EH_WSN(N=1)
        plane = CentralControlPlane(network, SimpleNamespace(), {})
        before = network.nodes[0].battery
        charged, alive = plane.begin_round([], 1, np.array([0.1]))
        network.nodes[0].harvest(0.1)
        plane.end_round(alive)
        self.assertAlmostEqual(network.nodes[0].battery, before + 0.1 - charged)
        self.assertAlmostEqual(network.nodes[0].total_harvest, 0.1)

    def test_unaffordable_report_excludes_round_without_fake_death(self):
        network = EH_WSN(N=1)
        network.nodes[0].battery = 1e-6
        plane = CentralControlPlane(network, SimpleNamespace(), {})
        charged, alive = plane.begin_round([], 1, np.zeros(1))
        self.assertEqual(charged, 0.0)
        self.assertFalse(network.nodes[0].is_alive)
        self.assertEqual(plane.failed_transactions, 1)
        plane.end_round(alive)
        self.assertTrue(network.nodes[0].is_alive)
        self.assertAlmostEqual(network.nodes[0].battery, 1e-6)


class BoundTests(unittest.TestCase):
    def test_bound_dominates_every_delivery_subset(self):
        lower = np.array([1e-5, 3e-5, 8e-5, 1e-4])
        bound = bound_from_costs(lower)
        for n in range(1, len(lower) + 1):
            for subset in itertools.combinations(range(len(lower)), n):
                value = (WEIGHTS['completion'] * n / len(lower) + WEIGHTS['lifetime']
                         - WEIGHTS['energy'] * sum(lower[i] for i in subset) / n / ENERGY_REFERENCE_J)
                self.assertLessEqual(value, bound + 1e-12)
        self.assertLessEqual(WEIGHTS['lifetime'], bound)

    def test_bound_rejects_invalid_costs(self):
        with self.assertRaises(ValueError):
            bound_from_costs([np.nan])
        with self.assertRaises(ValueError):
            bound_from_costs([-1])
