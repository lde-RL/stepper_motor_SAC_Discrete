"""
Replay Buffer with Prioritized Experience Replay (PER)
- Efficient sum-tree implementation for O(log n) sampling
- n-step returns support
- Importance sampling weight correction
"""

import numpy as np
import torch
from typing import Tuple, List, Optional


class SumTree:
    """
    Binary sum tree for efficient prioritized sampling

    Structure:
        - Leaf nodes store priorities
        - Internal nodes store sum of children
        - Root stores total priority sum
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Update parent nodes after priority change"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index corresponding to priority value s"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority sum"""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new data with given priority"""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """Update priority of existing entry"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Get data corresponding to priority value s
        Returns: (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Reference: Schaul et al. (2016)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.001,
        epsilon: float = 1e-6,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        """
        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_annealing: Amount to increase beta per sample (reaches 1.0)
            epsilon: Small constant to avoid zero priority
            n_step: n-step return calculation
            gamma: Discount factor for n-step
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = []

        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.n_entries

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add transition to buffer (with n-step processing)"""
        transition = (obs, action, reward, next_obs, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate n-step return
        obs_0, action_0, _, _, _ = self.n_step_buffer[0]
        n_step_reward = 0.0
        n_step_done = False

        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                n_step_done = True
                break

        next_obs_n = self.n_step_buffer[-1][3]

        # Store with maximum priority (will be updated after training)
        data = (obs_0, action_0, n_step_reward, next_obs_n, n_step_done)
        self.tree.add(self.max_priority ** self.alpha, data)

        # Remove oldest transition from n-step buffer
        self.n_step_buffer.pop(0)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized sampling

        Returns:
            obs, actions, rewards, next_obs, dones, weights, indices
        """
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        # Increase beta (importance sampling correction)
        self.beta = min(1.0, self.beta + self.beta_annealing)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        # Unpack batch
        obs = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_obs = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return (
            torch.FloatTensor(obs),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class SimpleReplayBuffer:
    """
    Simple uniform replay buffer (no prioritization)
    Fallback option if PER is too expensive
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.n_step = n_step
        self.gamma = gamma

        self.buffer = []
        self.position = 0
        self.n_step_buffer = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add transition"""
        transition = (obs, action, reward, next_obs, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate n-step return
        obs_0, action_0, _, _, _ = self.n_step_buffer[0]
        n_step_reward = 0.0
        n_step_done = False

        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                n_step_done = True
                break

        next_obs_n = self.n_step_buffer[-1][3]
        data = (obs_0, action_0, n_step_reward, next_obs_n, n_step_done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data

        self.position = (self.position + 1) % self.capacity
        self.n_step_buffer.pop(0)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch uniformly"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        obs = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_obs = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return (
            torch.FloatTensor(obs),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(dones),
            None,  # No weights for uniform sampling
            None   # No indices
        )

    def update_priorities(self, indices: Optional[List[int]], td_errors: Optional[np.ndarray]):
        """No-op for uniform buffer"""
        pass
