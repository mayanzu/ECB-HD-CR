"""
Dueling Double Deep Q-Network (D3QN) implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DuelingDQN(nn.Module):
    """Dueling DQN architecture"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)

        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done,
             next_valid_actions):
        self.buffer.append(
            (state, action, reward, next_state, done, next_valid_actions)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                np.array(next_masks, dtype=bool))

    def __len__(self):
        return len(self.buffer)


class D3QN_Agent:
    """Double Dueling DQN Agent"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 0.001, gamma: float = 0.9,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        import os
        forced_device = os.environ.get("ECB_DEVICE")
        self.device = torch.device(forced_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)

        # Training stats
        self.update_count = 0
        self.episode_rewards = []

    def select_action(self, state, valid_actions=None, eval_mode=False):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            valid_actions: Mask of valid actions (optional)
            eval_mode: If True, no exploration

        Returns:
            Selected action index
        """
        if not eval_mode and random.random() < self.epsilon:
            # Exploration
            if valid_actions is not None:
                valid_indices = np.where(valid_actions)[0]
                return np.random.choice(valid_indices)
            else:
                return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]

                # Mask invalid actions
                if valid_actions is not None:
                    q_values[~valid_actions] = -np.inf

                return np.argmax(q_values)

    def store_transition(self, state, action, reward, next_state, done,
                         next_valid_actions=None):
        """Store a transition and the feasible actions at the next state."""
        if next_valid_actions is None:
            next_valid_actions = np.ones(self.action_dim, dtype=bool)
        else:
            next_valid_actions = np.asarray(next_valid_actions, dtype=bool)
            if next_valid_actions.shape != (self.action_dim,):
                raise ValueError(
                    "next_valid_actions must have shape "
                    f"({self.action_dim},), got {next_valid_actions.shape}"
                )
            if not np.any(next_valid_actions):
                raise ValueError("at least one next action must be feasible")
        self.memory.push(
            state, action, reward, next_state, done, next_valid_actions
        )

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_masks = torch.BoolTensor(next_masks).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: select action with policy net, evaluate with target net
        with torch.no_grad():
            next_policy_q = self.policy_net(next_states)
            next_policy_q = next_policy_q.masked_fill(~next_masks, -torch.inf)
            next_actions = next_policy_q.argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            )
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
