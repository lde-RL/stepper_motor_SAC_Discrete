"""
SAC-Discrete Algorithm Implementation for Stepper Motor Control
- Discrete action space (16 actions: 4-bit switch combinations)
- Entropy-regularized policy
- Twin Q-networks with target networks
- Automatic entropy tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """Policy network: π(a|s) - outputs logits over discrete actions"""

    def __init__(self, obs_dim: int, num_actions: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits (not probabilities)"""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.logits_out(x)
        return logits

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy
        Returns: (action_id, log_prob)
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy for given obs-action pairs
        Used in policy update
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class QNetwork(nn.Module):
    """Q-network: Q(s,a) - outputs Q-values for all actions"""

    def __init__(self, obs_dim: int, num_actions: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q(s, ·) for all actions"""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.q_out(x)
        return q_values


class SACDiscrete:
    """
    SAC for discrete action spaces
    Reference: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 16,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions

        # Networks
        self.policy = PolicyNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q1 = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q2 = QNetwork(obs_dim, num_actions, hidden_dim).to(device)

        # Target Q-networks
        self.q1_target = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q2_target = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Freeze target networks
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            # Target entropy: -log(1/|A|) * ratio (typically 0.5-1.0)
            if target_entropy is None:
                self.target_entropy = -np.log(1.0 / num_actions) * 0.7
            else:
                self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha).to(device)

        self.training_step = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action from policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _ = self.policy.get_action(obs_tensor, deterministic)
        return action

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Update networks with a batch of transitions

        Args:
            obs: (batch, obs_dim)
            actions: (batch,) - action indices
            rewards: (batch,)
            next_obs: (batch, obs_dim)
            dones: (batch,)
            weights: (batch,) - importance sampling weights for PER

        Returns:
            Dictionary of training metrics
        """
        if weights is None:
            weights = torch.ones_like(rewards)

        # ============= Update Q-networks =============
        with torch.no_grad():
            # Get next state action probabilities from current policy
            next_logits = self.policy(next_obs)
            next_probs = F.softmax(next_logits, dim=-1)

            # Get Q-values from target networks
            next_q1 = self.q1_target(next_obs)
            next_q2 = self.q2_target(next_obs)
            next_q = torch.min(next_q1, next_q2)

            # Expected Q-value: E_a'~π[Q(s',a') - α*log π(a'|s')]
            next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=-1)

            # TD target
            q_target = rewards + (1 - dones) * self.gamma * next_v

        # Current Q-values for taken actions
        q1_pred = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q losses (weighted for PER)
        q1_loss = (weights * F.mse_loss(q1_pred, q_target, reduction='none')).mean()
        q2_loss = (weights * F.mse_loss(q2_pred, q_target, reduction='none')).mean()

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ============= Update Policy =============
        # Get current policy distribution
        logits = self.policy(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get Q-values from current Q-networks (no gradient through Q)
        with torch.no_grad():
            q1_curr = self.q1(obs)
            q2_curr = self.q2(obs)
            q_curr = torch.min(q1_curr, q2_curr)

        # Policy loss: E_a~π[α*log π(a|s) - Q(s,a)]
        policy_loss = (probs * (self.alpha * log_probs - q_curr)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ============= Update Temperature (α) =============
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            # Current entropy
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # α loss: -α * (entropy - target_entropy)
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # ============= Update Target Networks =============
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.training_step += 1

        # Calculate TD errors for PER priority update
        with torch.no_grad():
            td_errors = torch.abs(q1_pred - q_target)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'entropy': -(probs * log_probs).sum(dim=-1).mean().item(),
            'q_mean': q_curr.mean().item(),
            'td_errors': td_errors.cpu().numpy()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'training_step': self.training_step
        }, path)

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.training_step = checkpoint['training_step']
