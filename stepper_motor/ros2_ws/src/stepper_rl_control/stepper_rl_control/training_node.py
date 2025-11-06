"""
PC Training Node for Stepper Motor RL
- Generates sin wave reference trajectories
- Calculates rewards using encoder feedback
- Manages episode logic (no physical reset)
- Trains SAC-Discrete policy
- Deploys updated policies to Teensy
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import json

# ROS messages (placeholder - will be replaced with custom messages)
from std_msgs.msg import Float32, UInt8, Int32

# RL components
from .sac_discrete import SACDiscrete
from .replay_buffer import PrioritizedReplayBuffer, SimpleReplayBuffer
from .policy_quantization import QuantizedPolicy


class TrajectoryGenerator:
    """Generates variable sin wave trajectories for training"""

    def __init__(
        self,
        min_amplitude: float = 0.5,
        max_amplitude: float = 2.0,
        min_frequency: float = 0.5,
        max_frequency: float = 3.0,
        ramp_duration: float = 0.1
    ):
        self.min_amp = min_amplitude
        self.max_amp = max_amplitude
        self.min_freq = min_frequency
        self.max_freq = max_frequency
        self.ramp_duration = ramp_duration

        self.amplitude = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.start_time = 0.0

    def reset(self):
        """Randomize trajectory parameters for new episode"""
        self.amplitude = np.random.uniform(self.min_amp, self.max_amp)
        self.frequency = np.random.uniform(self.min_freq, self.max_freq)
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.start_time = 0.0

    def get_target(self, t: float, preview_steps: int = 32, dt: float = 0.001) -> tuple:
        """
        Get current target and preview

        Args:
            t: Current time in episode (seconds)
            preview_steps: Number of future steps to preview (K)
            dt: Time step (1 kHz = 0.001 s)

        Returns:
            theta_now, dtheta_now, theta_preview[]
        """
        # Apply ramp-up/ramp-down at episode boundaries
        if t < self.ramp_duration:
            ramp = t / self.ramp_duration
        else:
            ramp = 1.0

        amp = self.amplitude * ramp

        # Current target
        theta_now = amp * np.sin(2 * np.pi * self.frequency * t + self.phase)
        dtheta_now = amp * 2 * np.pi * self.frequency * np.cos(
            2 * np.pi * self.frequency * t + self.phase
        )

        # Future preview
        theta_preview = []
        for i in range(1, preview_steps + 1):
            t_future = t + i * dt
            theta_future = amp * np.sin(2 * np.pi * self.frequency * t_future + self.phase)
            theta_preview.append(theta_future)

        return theta_now, dtheta_now, np.array(theta_preview)


class RewardCalculator:
    """Calculate reward based on encoder feedback (TRAINING ONLY)"""

    def __init__(
        self,
        error_weight: float = 1.0,
        error_derivative_weight: float = 0.5,
        switch_penalty: float = 0.01,
        fault_penalty: float = 10.0
    ):
        self.k_error = error_weight
        self.k_deriv = error_derivative_weight
        self.k_switch = switch_penalty
        self.k_fault = fault_penalty

        self.last_error = 0.0
        self.last_action = 0

    def reset(self):
        """Reset internal state for new episode"""
        self.last_error = 0.0
        self.last_action = 0

    def compute(
        self,
        theta_actual: float,
        theta_ref: float,
        action: int,
        fault_flags: int,
        dt: float = 0.001
    ) -> tuple:
        """
        Compute reward and done flag

        Returns:
            (reward, done)
        """
        # Angle error (wrapped to [-pi, pi])
        error = theta_ref - theta_actual
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi

        # Error derivative
        error_deriv = (error - self.last_error) / dt

        # Reward components
        r_error = -self.k_error * abs(error)
        r_deriv = -self.k_deriv * abs(error_deriv)

        # Switch penalty (discourage rapid switching)
        r_switch = -self.k_switch * (action != self.last_action)

        # Total reward
        reward = r_error + r_deriv + r_switch

        # Check termination
        done = False
        if fault_flags != 0:
            reward -= self.k_fault
            done = True

        # Update state
        self.last_error = error
        self.last_action = action

        return reward, done


class StepperRLTrainer(Node):
    """Main training node"""

    def __init__(self, config: dict):
        super().__init__('stepper_rl_trainer')

        self.config = config

        # QoS profiles
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.ref_pub = self.create_publisher(Float32, 'ref_target', qos)
        self.action_pub = self.create_publisher(UInt8, 'action_cmd', qos)

        # Subscribers
        self.encoder_sub = self.create_subscription(
            Float32, 'encoder_angle', self.encoder_callback, qos
        )
        self.obs_sub = self.create_subscription(
            Float32, 'observation', self.obs_callback, qos
        )

        # State
        self.current_encoder_angle = 0.0
        self.current_obs = None
        self.episode_start_time = self.get_clock().now()
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0

        # Trajectory generation
        self.traj_gen = TrajectoryGenerator(
            min_amplitude=config['traj']['min_amplitude'],
            max_amplitude=config['traj']['max_amplitude'],
            min_frequency=config['traj']['min_frequency'],
            max_frequency=config['traj']['max_frequency']
        )
        self.traj_gen.reset()

        # Reward calculation
        self.reward_calc = RewardCalculator(
            error_weight=config['reward']['error_weight'],
            error_derivative_weight=config['reward']['derivative_weight'],
            switch_penalty=config['reward']['switch_penalty'],
            fault_penalty=config['reward']['fault_penalty']
        )

        # RL components
        obs_dim = self._calculate_obs_dim()
        self.agent = SACDiscrete(
            obs_dim=obs_dim,
            num_actions=config['rl']['num_actions'],
            hidden_dim=config['rl']['hidden_dim'],
            lr=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            tau=config['rl']['tau'],
            alpha=config['rl']['alpha'],
            auto_entropy_tuning=config['rl']['auto_entropy_tuning']
        )

        # Replay buffer
        if config['rl']['use_per']:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config['rl']['buffer_capacity'],
                obs_dim=obs_dim,
                alpha=config['rl']['per_alpha'],
                beta=config['rl']['per_beta'],
                n_step=config['rl']['n_step'],
                gamma=config['rl']['gamma']
            )
        else:
            self.replay_buffer = SimpleReplayBuffer(
                capacity=config['rl']['buffer_capacity'],
                obs_dim=obs_dim,
                n_step=config['rl']['n_step'],
                gamma=config['rl']['gamma']
            )

        # Training state
        self.last_obs = None
        self.last_action = None
        self.episode_rewards = []
        self.episode_return = 0.0

        # Logging
        self.log_dir = Path(config['logging']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(config['logging']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Control timer (1 kHz)
        self.control_dt = 0.001
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

        # Training timer (adjustable frequency)
        self.train_freq = config['rl']['train_frequency']
        self.train_timer = self.create_timer(1.0 / self.train_freq, self.train_loop)

        # Logging timer
        self.log_timer = self.create_timer(10.0, self.log_stats)

        # Episode management
        self.max_episode_steps = config['episode']['max_steps']
        self.min_episode_steps = config['episode']['min_steps']

        self.get_logger().info('StepperRLTrainer initialized')
        self.get_logger().info(f'Observation dimension: {obs_dim}')
        self.get_logger().info(f'Replay buffer capacity: {config["rl"]["buffer_capacity"]}')

    def _calculate_obs_dim(self) -> int:
        """Calculate observation dimension"""
        # theta_ref_now (1) + dtheta_ref_now (1) + theta_ref_preview (K) +
        # last_action_mask (1) + action_hist (N)
        K = self.config['rl']['preview_length']
        N = self.config['rl']['action_hist_length']
        return 1 + 1 + K + 1 + N

    def encoder_callback(self, msg: Float32):
        """Receive encoder angle (rad) - TRAINING ONLY"""
        self.current_encoder_angle = msg.data

    def obs_callback(self, msg: Float32):
        """Receive observation from Teensy (placeholder)"""
        # TODO: Parse actual Observation message when custom msgs are built
        pass

    def control_loop(self):
        """1 kHz control loop"""
        # Calculate episode time
        current_time = self.get_clock().now()
        episode_elapsed = (current_time - self.episode_start_time).nanoseconds / 1e9

        # Get reference target
        theta_ref, dtheta_ref, theta_preview = self.traj_gen.get_target(
            episode_elapsed,
            preview_steps=self.config['rl']['preview_length'],
            dt=self.control_dt
        )

        # Publish reference (placeholder - will use custom RefTarget message)
        ref_msg = Float32()
        ref_msg.data = float(theta_ref)
        self.ref_pub.publish(ref_msg)

        # Construct observation
        obs = self._build_observation(theta_ref, dtheta_ref, theta_preview)

        # Select action
        if len(self.replay_buffer) < self.config['rl']['warmup_steps']:
            # Random exploration during warmup
            action = np.random.randint(0, self.config['rl']['num_actions'])
        else:
            action = self.agent.select_action(obs, deterministic=False)

        # Publish action
        action_msg = UInt8()
        action_msg.data = int(action)
        self.action_pub.publish(action_msg)

        # Calculate reward (using encoder)
        reward, done = self.reward_calc.compute(
            self.current_encoder_angle,
            theta_ref,
            action,
            fault_flags=0  # TODO: Get from Teensy
        )

        self.episode_return += reward

        # Store transition
        if self.last_obs is not None and self.last_action is not None:
            self.replay_buffer.add(
                self.last_obs,
                self.last_action,
                reward,
                obs,
                done
            )

        # Check episode termination
        self.episode_step += 1
        self.total_steps += 1

        if done or self.episode_step >= self.max_episode_steps:
            self._end_episode()

        # Update state
        self.last_obs = obs
        self.last_action = action

    def _build_observation(
        self,
        theta_ref: float,
        dtheta_ref: float,
        theta_preview: np.ndarray
    ) -> np.ndarray:
        """
        Build observation vector (NO encoder data!)

        Format: [theta_ref_now, dtheta_ref_now, theta_preview..., last_action, action_hist...]
        """
        obs = [theta_ref, dtheta_ref]
        obs.extend(theta_preview.tolist())

        # Last action (placeholder - should come from Teensy observation)
        if self.last_action is not None:
            obs.append(float(self.last_action))
        else:
            obs.append(0.0)

        # Action history (placeholder)
        N = self.config['rl']['action_hist_length']
        obs.extend([0.0] * N)

        return np.array(obs, dtype=np.float32)

    def train_loop(self):
        """Training update loop"""
        if len(self.replay_buffer) < self.config['rl']['warmup_steps']:
            return

        if len(self.replay_buffer) < self.config['rl']['batch_size']:
            return

        # Sample batch
        batch = self.replay_buffer.sample(self.config['rl']['batch_size'])
        obs, actions, rewards, next_obs, dones, weights, indices = batch

        # Move to device
        obs = obs.to(self.agent.device)
        actions = actions.to(self.agent.device)
        rewards = rewards.to(self.agent.device)
        next_obs = next_obs.to(self.agent.device)
        dones = dones.to(self.agent.device)
        if weights is not None:
            weights = weights.to(self.agent.device)

        # Update agent
        metrics = self.agent.update(obs, actions, rewards, next_obs, dones, weights)

        # Update priorities (for PER)
        if self.config['rl']['use_per'] and indices is not None:
            self.replay_buffer.update_priorities(indices, metrics['td_errors'])

        # Log metrics
        if self.total_steps % 1000 == 0:
            self.get_logger().info(
                f"Step {self.total_steps} | "
                f"Q-loss: {metrics['q1_loss']:.4f} | "
                f"Policy-loss: {metrics['policy_loss']:.4f} | "
                f"Alpha: {metrics['alpha']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

    def _end_episode(self):
        """Handle episode termination"""
        self.episode_rewards.append(self.episode_return)
        self.episode_count += 1

        self.get_logger().info(
            f"Episode {self.episode_count} ended | "
            f"Steps: {self.episode_step} | "
            f"Return: {self.episode_return:.4f} | "
            f"Avg Return (100): {np.mean(self.episode_rewards[-100:]):.4f}"
        )

        # Save model periodically
        if self.episode_count % self.config['logging']['save_interval'] == 0:
            self._save_model()

        # Reset episode
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_start_time = self.get_clock().now()
        self.traj_gen.reset()
        self.reward_calc.reset()

    def _save_model(self):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.model_dir / f'model_ep{self.episode_count}_{timestamp}.pt'
        self.agent.save(str(model_path))
        self.get_logger().info(f'Saved model to {model_path}')

        # Save quantized version for Teensy deployment
        quant_policy = QuantizedPolicy(self.agent.policy)
        quant_policy.quantize()
        quant_path = self.model_dir / f'model_ep{self.episode_count}_{timestamp}_quant.bin'
        quant_policy.save(str(quant_path))

    def log_stats(self):
        """Periodic statistics logging"""
        if len(self.episode_rewards) == 0:
            return

        stats = {
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'avg_return_10': float(np.mean(self.episode_rewards[-10:])),
            'avg_return_100': float(np.mean(self.episode_rewards[-100:])),
            'max_return': float(np.max(self.episode_rewards)),
        }

        self.get_logger().info(f"Stats: {stats}")

        # Save to file
        stats_file = self.log_dir / 'training_stats.jsonl'
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    # Load configuration
    config = {
        'traj': {
            'min_amplitude': 0.5,
            'max_amplitude': 2.0,
            'min_frequency': 0.5,
            'max_frequency': 3.0
        },
        'reward': {
            'error_weight': 1.0,
            'derivative_weight': 0.5,
            'switch_penalty': 0.01,
            'fault_penalty': 10.0
        },
        'rl': {
            'num_actions': 16,
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'auto_entropy_tuning': True,
            'buffer_capacity': 100000,
            'use_per': True,
            'per_alpha': 0.6,
            'per_beta': 0.4,
            'n_step': 5,
            'batch_size': 256,
            'warmup_steps': 1000,
            'train_frequency': 200,
            'preview_length': 32,
            'action_hist_length': 16
        },
        'episode': {
            'min_steps': 2000,
            'max_steps': 5000
        },
        'logging': {
            'log_dir': 'logs',
            'model_dir': 'models',
            'save_interval': 10
        }
    }

    node = StepperRLTrainer(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
