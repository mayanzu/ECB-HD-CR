"""Ideal slotted BS control protocol with sensor-side radio accounting.

The wrapper applies harvest before telemetry and suppresses the algorithms'
second harvest call. Failed control transactions exclude the node from that
round's decision stage; permanent death is still checked after background draw.
"""
from collections import Counter


class CentralControlPlane:
    def __init__(self, network, algorithm, config):
        self.network = network
        self.algorithm = algorithm
        self.config = config
        self.telemetry_bits = int(config.get("telemetry_base_bits", 128))
        self.descriptor_bits = int(config.get("task_descriptor_bits", 32))
        self.command_bits = int(config.get("route_command_bits", 64))
        self.topology_bits = int(config.get("topology_message_bits", 64))
        self.period = int(config.get("T_L", 50))
        self.energy = {"telemetry": 0.0, "commands": 0.0, "topology": 0.0}
        self.messages = {key: 0 for key in self.energy}
        self.failed_transactions = 0

    def _bill(self, node_id, cost):
        target = getattr(self.algorithm, "cr", self.algorithm)
        if hasattr(target, "ecb"):
            target.ecb.record_bill(node_id, cost)

    def begin_round(self, tasks, current_round, harvests):
        counts = Counter(task.source_node for task in tasks)
        previous_alive = [node.is_alive for node in self.network.nodes]
        target = getattr(self.algorithm, "cr", self.algorithm)
        clustered = bool(getattr(target, "enable_clustering", hasattr(target, "cluster_heads")))
        charged = 0.0
        for node, harvest in zip(self.network.nodes, harvests):
            if not node.is_alive:
                continue
            node.harvest(harvest)
            node._skip_next_harvest = True
            count = counts[node.id]
            sizes = {"telemetry": self.telemetry_bits + count * self.descriptor_bits}
            if count:
                # ICR makes one decision per task; the other schedulers decide
                # once per batch. This conservative frame count is explicit.
                decisions = count if self.algorithm.__class__.__name__ == "ICR_HHODRL" else 1
                sizes["commands"] = decisions * self.command_bits
            if clustered and current_round % self.period == 0:
                sizes["topology"] = self.topology_bits
            available = True
            for kind, bits in sizes.items():
                cost = (self.network.E_tx(bits, node.distance_to(self.network.bs))
                        if kind == "telemetry" else self.network.E_rx(bits))
                if not node.consume(cost):
                    self.failed_transactions += 1
                    available = False
                    break
                self.energy[kind] += cost
                self.messages[kind] += 1
                self._bill(node.id, cost)
                charged += cost
            if not available:
                node.is_alive = False  # temporary control-stage unavailability
        return charged, previous_alive

    def end_round(self, previous_alive):
        for node, was_alive in zip(self.network.nodes, previous_alive):
            node.is_alive = was_alive
            node._skip_next_harvest = False
