"""
Energy scaling wrapper for experiments
Scales down solar energy to create realistic scarcity
"""

import numpy as np

class EnergyScaler:
    """Scale solar energy to match consumption"""

    def __init__(self, scale_factor=0.0001):
        """
        Args:
            scale_factor: Multiply solar energy by this factor
                         0.0001 means 10,000x reduction
        """
        self.scale_factor = scale_factor

    def scale_trace(self, trace: np.ndarray) -> np.ndarray:
        """Scale energy trace"""
        return trace * self.scale_factor

    def scale_spatial_traces(self, traces: np.ndarray) -> np.ndarray:
        """Scale 2D spatial traces"""
        return traces * self.scale_factor


def get_scaled_solar_traces(solar_gen, total_rounds, num_nodes,
                            scale_factor=0.0001):
    """
    Generate solar traces with energy scaling

    Args:
        solar_gen: SolarTraceGenerator instance
        total_rounds: Number of rounds
        num_nodes: Number of nodes
        scale_factor: Energy scaling factor (default: 0.0001)

    Returns:
        Scaled spatial traces (num_nodes × total_rounds)
    """
    scaler = EnergyScaler(scale_factor)

    # Generate base trace
    base_trace = solar_gen.generate_synthetic_trace(total_rounds)

    # Add spatial variation
    spatial_traces = solar_gen.add_spatial_variation(base_trace, num_nodes)

    # Scale down
    scaled_traces = scaler.scale_spatial_traces(spatial_traces)

    return scaled_traces


if __name__ == '__main__':
    print("Energy Scaler Test")
    print("="*80)

    # Example
    original_energy = 100.0  # J
    scale_factor = 0.0001

    scaled = original_energy * scale_factor

    print(f"Original solar energy: {original_energy} J")
    print(f"Scale factor: {scale_factor}")
    print(f"Scaled energy: {scaled} J")
    print(f"\nThis reduces energy by {int(1/scale_factor)}x")
    print("="*80)
