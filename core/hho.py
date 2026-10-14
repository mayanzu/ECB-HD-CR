"""
Harris Hawks Optimization (HHO) algorithm
Reference: Heidari et al. (2019) Future Generation Computer Systems
"""

import numpy as np
import math
from typing import Callable, Tuple, List


class HarrisHawksOptimizer:
    """Generic bounded continuous Harris Hawks optimizer."""

    def __init__(self, dim: int, pop_size: int = 30, max_iter: int = 100,
                 lb: float = 0.0, ub: float = 1.0):
        """
        Args:
            dim: Dimension (number of nodes)
            pop_size: Population size
            max_iter: Maximum iterations
            lb: Lower bound
            ub: Upper bound
        """
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

        # Initialize population
        self.positions = np.random.uniform(lb, ub, (pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.rabbit_position = None
        self.rabbit_fitness = np.inf

    def optimize(self, fitness_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Run HHO optimization

        Args:
            fitness_func: Fitness function to minimize

        Returns:
            (best_position, best_fitness)
        """
        self.rabbit_position = None
        self.rabbit_fitness = np.inf

        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = fitness_func(self.positions[i])
            if not np.isfinite(self.fitness[i]):
                self.fitness[i] = np.inf

            if self.fitness[i] < self.rabbit_fitness:
                self.rabbit_fitness = self.fitness[i]
                self.rabbit_position = self.positions[i].copy()

        if self.rabbit_position is None:
            self.rabbit_position = np.clip(self.positions[0].copy(), self.lb, self.ub)

        # Main loop
        for t in range(self.max_iter):
            E0 = 2 * np.random.random() - 1  # [-1, 1]
            E = 2 * E0 * (1 - t / self.max_iter)  # Escaping energy

            for i in range(self.pop_size):
                # Update position
                q = np.random.random()
                r = np.random.random()
                J = 2 * (1 - r)  # Random jump strength

                if abs(E) >= 1:
                    # Exploration phase (|E| >= 1)
                    if q >= 0.5:
                        # Perch randomly
                        rand_idx = np.random.randint(0, self.pop_size)
                        X_rand = self.positions[rand_idx]
                        r1 = np.random.random()
                        r2 = np.random.random()
                        new_pos = X_rand - r1 * np.abs(
                            X_rand - 2 * r2 * self.positions[i]
                        )
                    else:
                        # Perch on random tall trees
                        r3 = np.random.random()
                        r4 = np.random.random()
                        new_pos = (self.rabbit_position -
                                  self.positions.mean(axis=0)) - r3 * (
                                  self.lb + r4 * (self.ub - self.lb))

                else:
                    # Exploitation phase (|E| < 1)
                    if r >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege
                        Delta_X = self.rabbit_position - self.positions[i]
                        new_pos = Delta_X - E * np.abs(
                            J * self.rabbit_position - self.positions[i]
                        )

                    elif r >= 0.5 and abs(E) < 0.5:
                        # Hard besiege
                        new_pos = self.rabbit_position - E * np.abs(
                            Delta_X := self.rabbit_position - self.positions[i]
                        )

                    elif r < 0.5 and abs(E) >= 0.5:
                        # Soft besiege with progressive rapid dives
                        Y = self.rabbit_position - E * np.abs(
                            J * self.rabbit_position - self.positions[i]
                        )
                        Y = np.clip(Y, self.lb, self.ub)
                        fitness_Y = fitness_func(Y)
                        if not np.isfinite(fitness_Y):
                            fitness_Y = np.inf

                        if fitness_Y < self.fitness[i]:
                            new_pos = Y
                        else:
                            S = np.random.random(self.dim)
                            LF = self._levy_flight(self.dim)
                            Z = Y + S * LF
                            Z = np.clip(Z, self.lb, self.ub)
                            fitness_Z = fitness_func(Z)
                            if not np.isfinite(fitness_Z):
                                fitness_Z = np.inf

                            if fitness_Z < self.fitness[i]:
                                new_pos = Z
                            else:
                                new_pos = self.positions[i]

                    else:
                        # Hard besiege with progressive rapid dives
                        Y = self.rabbit_position - E * np.abs(
                            J * self.rabbit_position -
                            self.positions.mean(axis=0)
                        )
                        Y = np.clip(Y, self.lb, self.ub)
                        fitness_Y = fitness_func(Y)
                        if not np.isfinite(fitness_Y):
                            fitness_Y = np.inf

                        if fitness_Y < self.fitness[i]:
                            new_pos = Y
                        else:
                            S = np.random.random(self.dim)
                            LF = self._levy_flight(self.dim)
                            Z = Y + S * LF
                            Z = np.clip(Z, self.lb, self.ub)
                            fitness_Z = fitness_func(Z)
                            if not np.isfinite(fitness_Z):
                                fitness_Z = np.inf

                            if fitness_Z < self.fitness[i]:
                                new_pos = Z
                            else:
                                new_pos = self.positions[i]

                # Apply bounds
                new_pos = np.clip(new_pos, self.lb, self.ub)

                # Evaluate new position
                new_fitness = fitness_func(new_pos)
                if not np.isfinite(new_fitness):
                    new_fitness = np.inf

                # Update position if better
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness

                    # Update rabbit position
                    if new_fitness < self.rabbit_fitness:
                        self.rabbit_fitness = new_fitness
                        self.rabbit_position = new_pos.copy()

        return self.rabbit_position, self.rabbit_fitness

    def _levy_flight(self, dim: int) -> np.ndarray:
        """Generate Levy flight random walk"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta *
                 2**((beta - 1) / 2))) ** (1 / beta)

        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)

        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step


class CreditAllocator:
    """HHO-based credit allocation for ECB-HD-CR"""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.hho = HarrisHawksOptimizer(
            dim=num_nodes,
            pop_size=30,
            max_iter=50,
            lb=0.0,
            ub=2.0  # Credit can be 0 to 2x of average
        )

    def allocate_credits(self, predicted_harvests: np.ndarray,
                        prev_bills: np.ndarray,
                        prev_task_completion: np.ndarray,
                        cycle_length: int = 1) -> np.ndarray:
        """
        Allocate credit quotas using HHO

        Args:
            predicted_harvests: Predicted energy harvest per node per round
            prev_bills: Previous cycle energy bills
            prev_task_completion: Previous task completion rates
            cycle_length: Number of rounds covered by the issued quota

        Returns:
            Credit quotas for each node
        """
        # Normalize inputs
        cycle_harvests = predicted_harvests * max(1, cycle_length)
        mean_harvest = cycle_harvests.mean()
        if mean_harvest == 0:
            mean_harvest = 1.0

        normalized_harvests = cycle_harvests / mean_harvest
        normalized_bills = prev_bills / (prev_bills.mean() + 1e-6)

        # Define fitness function
        def fitness(credits):
            """
            Fitness = balance energy + maximize throughput + fairness
            """
            # Penalty for mismatch between credit and predicted harvest
            energy_mismatch = np.sum((credits - normalized_harvests) ** 2)

            # Penalty for deviating from historical bills
            bill_mismatch = np.sum((credits - normalized_bills) ** 2)

            # Reward for enabling high task completion
            throughput_reward = -np.sum(credits * prev_task_completion)

            # Fairness: penalize large std
            fairness_penalty = np.std(credits)

            return (0.4 * energy_mismatch +
                   0.3 * bill_mismatch +
                   0.2 * throughput_reward +
                   0.1 * fairness_penalty)

        # Run optimization
        best_allocation, _ = self.hho.optimize(fitness)

        # Denormalize and return
        credits = best_allocation * mean_harvest
        return credits
