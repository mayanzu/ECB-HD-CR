"""
Solar Energy Harvesting Model
Loads and processes NSRDB solar irradiance data
"""

import numpy as np
from typing import Optional


class SolarHarvester:
    """Solar energy harvesting model"""

    def __init__(self, panel_area: float = 0.01, efficiency: float = 0.20):
        """
        Args:
            panel_area: Solar panel area in m^2 (default: 10 cm^2)
            efficiency: Panel efficiency (default: 20%)
        """
        self.panel_area = panel_area
        self.efficiency = efficiency

    def irradiance_to_power(self, irradiance: float) -> float:
        """
        Convert solar irradiance (W/m^2) to harvested power (W)

        Args:
            irradiance: Solar irradiance in W/m^2

        Returns:
            Harvested power in Watts
        """
        return irradiance * self.panel_area * self.efficiency

    def power_to_energy(self, power: float, duration: float = 60.0) -> float:
        """
        Convert power to energy over duration

        Args:
            power: Power in Watts
            duration: Time duration in seconds (default: 60s = 1 minute)

        Returns:
            Energy in Joules
        """
        return power * duration


class SolarTraceGenerator:
    """Generate synthetic or load real solar traces"""

    def __init__(self, trace_type: str = 'synthetic', seed: Optional[int] = None):
        """
        Args:
            trace_type: 'synthetic' or 'nsrdb'
            seed: Random seed for reproducibility
        """
        self.trace_type = trace_type
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        self.harvester = SolarHarvester()

    def generate_synthetic_trace(self, total_rounds: int) -> np.ndarray:
        """
        Generate synthetic diurnal solar pattern

        Simplified model:
        - Day/night cycle: 12 hours day, 12 hours night
        - Peak at noon: cosine pattern
        - Random weather variation

        Args:
            total_rounds: Number of rounds (1 round = 1 minute)

        Returns:
            Array of harvested energy per round (in Joules)
        """
        energy_trace = np.zeros(total_rounds)

        rounds_per_day = 24 * 60  # 1440 rounds per day
        peak_irradiance = 800  # W/m^2 at noon
        sunrise = 6 * 60  # 6:00 AM
        sunset = 18 * 60  # 6:00 PM

        for t in range(total_rounds):
            time_of_day = t % rounds_per_day
            day = t // rounds_per_day

            if sunrise <= time_of_day < sunset:
                # Daytime: cosine pattern
                progress = (time_of_day - sunrise) / (sunset - sunrise)
                angle = progress * np.pi  # 0 to π
                base_irradiance = peak_irradiance * np.sin(angle)

                # Add weather variation (20% std)
                weather_factor = self.rng.normal(1.0, 0.2)
                weather_factor = max(0.3, min(1.5, weather_factor))

                irradiance = base_irradiance * weather_factor
            else:
                # Nighttime
                irradiance = 0.0

            # Convert to energy
            power = self.harvester.irradiance_to_power(irradiance)
            energy = self.harvester.power_to_energy(power, duration=60.0)
            energy_trace[t] = energy

        return energy_trace

    def load_nsrdb_trace(self, location: str = 'california',
                         year: int = 2023) -> np.ndarray:
        """Reject the obsolete location-only NSRDB API.

        Real NSRDB experiments must name an auditable CSV and use
        :func:`data.nsrdb_loader.nsrdb_base_trace`.  Returning a synthetic
        trace from a method named ``load_nsrdb_trace`` would silently corrupt
        source-comparison claims.
        """
        raise NotImplementedError(
            "Use data.nsrdb_loader.nsrdb_base_trace(csv_path, total_rounds); "
            f"location={location!r}, year={year!r} is not a file-backed trace."
        )

    def add_spatial_variation(self, base_trace: np.ndarray,
                            num_nodes: int) -> np.ndarray:
        """
        Add spatial variation for different node locations

        Nodes in different locations receive different amounts of solar energy
        due to canopy shading, building shadows, etc.

        Args:
            base_trace: Base energy trace
            num_nodes: Number of nodes

        Returns:
            Array of shape (num_nodes, len(base_trace))
        """
        spatial_traces = np.zeros((num_nodes, len(base_trace)))

        for i in range(num_nodes):
            # Each node has a location-dependent scaling factor
            # Uniform distribution [0.5, 1.0] for diversity
            location_factor = self.rng.uniform(0.5, 1.0)

            # Add temporal correlation (same weather for nearby nodes)
            spatial_traces[i] = base_trace * location_factor

        return spatial_traces


def generate_solar_data_for_experiment(num_nodes: int = 100,
                                       total_rounds: int = 30000,
                                       num_runs: int = 30,
                                       output_dir: str = '../data/'):
    """
    Generate solar data for multiple experimental runs

    Args:
        num_nodes: Number of sensor nodes
        total_rounds: Simulation length in rounds
        num_runs: Number of independent runs
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for run in range(num_runs):
        generator = SolarTraceGenerator(seed=1000 + run)
        base_trace = generator.generate_synthetic_trace(total_rounds)
        spatial_traces = generator.add_spatial_variation(base_trace, num_nodes)

        # Save to file
        filename = f'{output_dir}/solar_N{num_nodes}_run{run:02d}.npz'
        np.savez_compressed(filename,
                          spatial_traces=spatial_traces,
                          base_trace=base_trace,
                          num_nodes=num_nodes,
                          total_rounds=total_rounds,
                          run_id=run)

        print(f'Generated run {run}: {filename}')


if __name__ == '__main__':
    # Generate data for experiments
    generate_solar_data_for_experiment(num_nodes=100, num_runs=5)
