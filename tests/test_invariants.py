import unittest

import numpy as np

from algorithms.baseline.ecb_static import ECB_Static
from algorithms.baseline.chhfo import CHHFO
from algorithms.baseline.d3qn_eh import D3QN_EH
from algorithms.baseline.icr_hhodrl import ICR_HHODRL
from algorithms.baseline.two_timescale_drl import TwoTimescaleDRL
from algorithms.ecb_hd.ecb_hd_cr import ECB_HD_CR
from algorithms.ecb_hd.ecb_hd_cr_ablations import ECB_HD_CR_NoFeedback
from core.d3qn import D3QN_Agent
from core.network import EH_WSN, SensorNode
from data.task_model import Task


class EnergyCausalityTests(unittest.TestCase):
    def test_failed_action_does_not_drain_battery(self):
        node = SensorNode(0, (0.0, 0.0), E_init=0.1, B_max=1.0)

        self.assertFalse(node.consume(0.2))
        self.assertAlmostEqual(node.battery, 0.1)
        self.assertAlmostEqual(node.total_consumed, 0.0)

    def test_background_consumption_can_drain_remaining_energy(self):
        node = SensorNode(0, (0.0, 0.0), E_init=0.1, B_max=1.0)

        spent = node.consume_up_to(0.2)

        self.assertAlmostEqual(spent, 0.1)
        self.assertAlmostEqual(node.battery, 0.0)
        self.assertAlmostEqual(node.total_consumed, 0.1)


class MaskedReplayTests(unittest.TestCase):
    def test_replay_stores_next_action_mask(self):
        agent = D3QN_Agent(2, 3, batch_size=1)
        mask = np.array([False, True, False])
        state = np.zeros(2)

        agent.store_transition(state, 1, 1.0, state, False, mask)
        *_, sampled_masks = agent.memory.sample(1)

        np.testing.assert_array_equal(sampled_masks[0], mask)
        self.assertIsInstance(agent.train_step(), float)


class BaselineConsistencyTests(unittest.TestCase):
    def test_static_credit_disables_periodic_control_messages(self):
        network = EH_WSN(N=3, seed=1)
        baseline = ECB_Static(network, self._config())

        self.assertFalse(baseline.cr.enable_control_overhead)
        self.assertTrue(baseline.cr.enable_bill_state)
        self.assertFalse(baseline.cr.enable_bill_reward)

    def test_icr_direct_action_does_not_charge_cluster_head(self):
        network = EH_WSN(N=3, bs_position=(50.0, 50.0), seed=2)
        method = ICR_HHODRL(network, self._config())
        method.cluster_heads = [0]
        method._assign_clusters()
        method.drl_router.select_action = lambda *args, **kwargs: 0
        source_id = 1
        task = Task(0, source_id, 0, 300, 1)
        ch_before = network.nodes[0].battery
        source = network.nodes[source_id]
        expected = network.E_tx(task.data_size, source.distance_to(network.bs))

        result = method.schedule_round([task], 0, np.zeros(network.N))

        self.assertEqual(result["completed"], 1)
        self.assertAlmostEqual(result["energy"], expected)
        self.assertAlmostEqual(network.nodes[0].battery, ch_before)

    def test_d3qn_batch_ordering_pays_full_radio_cost(self):
        network = EH_WSN(N=1, bs_position=(50.0, 50.0), seed=3)
        method = D3QN_EH(network, self._config())
        method.initialize()
        method.d3qn.select_action = lambda *args, **kwargs: 1
        tasks = [Task(i, 0, 0, bits, 1) for i, bits in enumerate([300, 200, 400])]
        expected = sum(
            network.E_tx(task.data_size, network.nodes[0].distance_to(network.bs))
            for task in tasks
        )

        result = method.schedule_round(tasks, 0, np.zeros(network.N))

        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["dropped"], 0)
        self.assertAlmostEqual(result["energy"], expected)

    def test_two_timescale_batch_ordering_pays_full_radio_cost(self):
        network = EH_WSN(N=1, bs_position=(50.0, 50.0), seed=4)
        method = TwoTimescaleDRL(network, self._config())
        method.initialize()
        method.d3qn.select_action = lambda *args, **kwargs: 1
        tasks = [Task(i, 0, 0, bits, 1) for i, bits in enumerate([300, 200, 400])]
        expected = sum(
            network.E_tx(task.data_size, network.nodes[0].distance_to(network.bs))
            for task in tasks
        )

        result = method.schedule_round(tasks, 0, np.zeros(network.N))

        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["dropped"], 0)
        self.assertAlmostEqual(result["energy"], expected)

    def test_all_hho_decoders_select_only_alive_cluster_heads(self):
        for method_class, optimizer_name, optimize_method in [
            (ECB_HD_CR, "hho", "_optimize_clusters"),
            (CHHFO, "hho", "_optimize_cluster_heads"),
            (ICR_HHODRL, "qhho", "_quantum_optimize_clusters"),
        ]:
            with self.subTest(method=method_class.__name__):
                network = EH_WSN(N=6, seed=5)
                network.nodes[5].is_alive = False
                network.nodes[4].is_alive = False
                method = method_class(network, self._config())
                optimizer = getattr(method, optimizer_name)
                optimizer.optimize = lambda fitness: (np.arange(network.N), 0.0)

                getattr(method, optimize_method)()

                self.assertEqual(len(method.cluster_heads), 3)
                self.assertTrue(
                    all(network.nodes[ch].is_alive for ch in method.cluster_heads)
                )

    def test_ecb_preserves_settled_balance_for_next_hho_call(self):
        network = EH_WSN(N=3, seed=6)
        config = self._config()
        config["enable_control_overhead"] = False
        method = ECB_HD_CR(network, config)
        method.initialize()
        method.ecb.credits[:] = 0.1
        method.ecb.bills[:] = np.array([0.2, 0.05, 0.1])

        method._cycle_update(np.zeros(network.N))

        self.assertLess(method._last_credit_balance_ratio[0], 0.0)
        self.assertGreater(method._last_credit_balance_ratio[1], 0.0)
        self.assertAlmostEqual(method.ecb.bills.sum(), 0.0)

    def test_ecb_control_energy_is_included_in_the_settled_bill(self):
        network = EH_WSN(N=1, seed=9)
        method = ECB_HD_CR(network, self._config())
        method.initialize()

        charged = method._charge_control_overhead()

        self.assertGreater(charged, 0.0)
        self.assertAlmostEqual(method.ecb.bills[0], charged)

    def test_no_feedback_freezes_adaptation_but_exposes_remaining_credit(self):
        network = EH_WSN(N=3, seed=10)
        config = self._config()
        config["enable_control_overhead"] = False
        method = ECB_HD_CR_NoFeedback(network, config)
        method.initialize()
        method.ecb.credits[:] = 0.1
        method.ecb.bills[:] = np.array([0.2, 0.05, 0.1])

        method._cycle_update(np.zeros(network.N))

        np.testing.assert_allclose(
            method.ecb.lambda_rates, method.ecb.lambda_0
        )
        np.testing.assert_allclose(method.ecb.mu_rates, method.ecb.mu_0)
        np.testing.assert_allclose(method._last_credit_balance_ratio, 1.0)
        self.assertTrue(method.enable_bill_state)
        credit = method.ecb.credits[0]
        self.assertAlmostEqual(method.get_state(0, 1)[1], credit / (credit + 1e-6))
        method.ecb.bills[0] = credit
        self.assertAlmostEqual(method.get_state(0, 1)[1], 0.0)
        self.assertEqual(method._apply_bill_reward(0, 7.5), 7.5)
        self.assertFalse(method.enable_control_overhead)

    def test_ch_own_task_does_not_spend_reserved_downstream_energy(self):
        network = EH_WSN(N=3, bs_position=(0.0, 100.0), seed=7)
        network.nodes[0].x, network.nodes[0].y = (0.0, 0.0)
        network.nodes[2].x, network.nodes[2].y = (0.0, 1.0)
        config = self._config()
        config.update(
            {
                "ch_energy_reserve_ratio": 0.0,
                "enable_control_overhead": False,
            }
        )
        method = ECB_HD_CR(network, config)
        method.initialize()
        method.cluster_heads = [2]
        method._assign_clusters()
        method._route_action = lambda node_id, queue: 2 if node_id == 0 else 1

        member_task = Task(0, 0, 0, 300, 1)
        ch_task = Task(1, 2, 0, 300, 1)
        member = network.nodes[0]
        ch = network.nodes[2]
        e_member = network.E_tx(300, member.distance_to(ch))
        e_receive = network.E_rx(300)
        e_downstream = method._cluster_downstream_cost(2, 300)
        e_ch_own = network.E_tx(300, ch.distance_to(network.bs))
        ch.battery = e_receive + e_ch_own + 0.5 * e_downstream

        result = method.schedule_round(
            [member_task, ch_task], 0, np.zeros(network.N)
        )

        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["dropped"], 1)
        self.assertAlmostEqual(
            result["energy"], e_member + e_receive + e_downstream
        )

    def test_proposed_continues_after_an_unaffordable_task(self):
        network = EH_WSN(N=1, bs_position=(50.0, 50.0), seed=8)
        method = ECB_HD_CR(network, self._config())
        method.initialize()
        method._route_action = lambda node_id, queue: 1
        network.nodes[0].battery = 0.002
        tasks = [
            Task(0, 0, 0, 100000, 1),
            Task(1, 0, 0, 100, 1),
        ]

        result = method.schedule_round(tasks, 0, np.zeros(network.N))

        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["dropped"], 1)
        self.assertAlmostEqual(
            result["energy"],
            network.E_tx(100, network.nodes[0].distance_to(network.bs)),
        )

    @staticmethod
    def _config():
        return {
            "lambda_0": 0.05,
            "mu_0": 0.03,
            "eta_c": 0.3,
            "xi": 0.5,
            "T_L": 50,
            "hho_pop_size": 2,
            "hho_iter": 1,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay": 1.0,
            "batch_size": 64,
        }


if __name__ == "__main__":
    unittest.main()
