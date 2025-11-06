"""
Monitoring and Visualization Utilities
- Real-time training metrics
- Episode trajectory plotting
- Performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict
from collections import deque


class MetricsLogger:
    """Log training metrics to file and tensorboard"""

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("Tensorboard not available, logging to file only")
                self.use_tensorboard = False

        self.metrics_file = self.log_dir / 'metrics.jsonl'

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar metric"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, value_dict: Dict[str, float], step: int):
        """Log multiple scalars"""
        if self.writer:
            self.writer.add_scalars(tag, value_dict, step)

    def log_episode(self, episode: int, metrics: dict):
        """Log episode metrics to file"""
        metrics['episode'] = episode
        metrics['timestamp'] = datetime.now().isoformat()

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def load_metrics(self) -> List[dict]:
        """Load all logged metrics"""
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        return metrics

    def close(self):
        """Close logger"""
        if self.writer:
            self.writer.close()


class TrajectoryVisualizer:
    """Visualize reference and actual trajectories"""

    def __init__(self, max_points: int = 1000):
        self.max_points = max_points

        # Data buffers
        self.time = deque(maxlen=max_points)
        self.ref_angle = deque(maxlen=max_points)
        self.actual_angle = deque(maxlen=max_points)
        self.error = deque(maxlen=max_points)
        self.action = deque(maxlen=max_points)
        self.reward = deque(maxlen=max_points)

        # Setup plot
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 10))
        self.fig.suptitle('Real-time Training Monitor')

        self.lines = []
        self._setup_plots()

    def _setup_plots(self):
        """Setup subplot configurations"""
        # Plot 1: Angle tracking
        ax1 = self.axes[0]
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title('Reference vs Actual Angle')
        ax1.grid(True)
        line_ref, = ax1.plot([], [], 'b-', label='Reference')
        line_act, = ax1.plot([], [], 'r-', label='Actual')
        ax1.legend()
        self.lines.append((line_ref, line_act))

        # Plot 2: Tracking error
        ax2 = self.axes[1]
        ax2.set_ylabel('Error (rad)')
        ax2.set_title('Tracking Error')
        ax2.grid(True)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        line_err, = ax2.plot([], [], 'r-')
        self.lines.append((line_err,))

        # Plot 3: Action
        ax3 = self.axes[2]
        ax3.set_ylabel('Action ID')
        ax3.set_title('Selected Actions (4-bit combinations)')
        ax3.grid(True)
        ax3.set_ylim(-0.5, 15.5)
        line_act, = ax3.plot([], [], 'g-', marker='.', markersize=2)
        self.lines.append((line_act,))

        # Plot 4: Reward
        ax4 = self.axes[3]
        ax4.set_ylabel('Reward')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Instantaneous Reward')
        ax4.grid(True)
        line_rew, = ax4.plot([], [], 'purple')
        self.lines.append((line_rew,))

        plt.tight_layout()

    def update(self, t: float, ref: float, actual: float, action: int, reward: float):
        """Update visualization with new data point"""
        self.time.append(t)
        self.ref_angle.append(ref)
        self.actual_angle.append(actual)
        self.error.append(ref - actual)
        self.action.append(action)
        self.reward.append(reward)

    def render(self):
        """Render current frame"""
        if len(self.time) == 0:
            return

        time_arr = np.array(self.time)

        # Update angle plot
        self.lines[0][0].set_data(time_arr, self.ref_angle)
        self.lines[0][1].set_data(time_arr, self.actual_angle)
        self.axes[0].relim()
        self.axes[0].autoscale_view()

        # Update error plot
        self.lines[1][0].set_data(time_arr, self.error)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

        # Update action plot
        self.lines[2][0].set_data(time_arr, self.action)
        self.axes[2].relim()
        self.axes[2].autoscale_view()

        # Update reward plot
        self.lines[3][0].set_data(time_arr, self.reward)
        self.axes[3].relim()
        self.axes[3].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, path: str):
        """Save current plot"""
        self.fig.savefig(path, dpi=150, bbox_inches='tight')

    def clear(self):
        """Clear all data buffers"""
        self.time.clear()
        self.ref_angle.clear()
        self.actual_angle.clear()
        self.error.clear()
        self.action.clear()
        self.reward.clear()


class PerformanceAnalyzer:
    """Analyze training performance from logged metrics"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.logger = MetricsLogger(log_dir, use_tensorboard=False)

    def plot_learning_curve(self, window: int = 100, save_path: str = None):
        """Plot episode returns over training"""
        metrics = self.logger.load_metrics()

        if len(metrics) == 0:
            print("No metrics found")
            return

        episodes = [m['episode'] for m in metrics]
        returns = [m.get('episode_return', 0) for m in metrics]

        # Moving average
        if len(returns) >= window:
            returns_smooth = np.convolve(returns, np.ones(window)/window, mode='valid')
            episodes_smooth = episodes[window-1:]
        else:
            returns_smooth = returns
            episodes_smooth = episodes

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episodes, returns, alpha=0.3, label='Raw')
        ax.plot(episodes_smooth, returns_smooth, linewidth=2, label=f'MA-{window}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def plot_training_metrics(self, save_path: str = None):
        """Plot Q-loss, policy-loss, alpha over training"""
        metrics = self.logger.load_metrics()

        if len(metrics) == 0:
            print("No metrics found")
            return

        steps = [m.get('step', i) for i, m in enumerate(metrics)]
        q_loss = [m.get('q_loss', 0) for m in metrics]
        policy_loss = [m.get('policy_loss', 0) for m in metrics]
        alpha = [m.get('alpha', 0) for m in metrics]

        fig, axes = plt.subplots(3, 1, figsize=(10, 9))

        axes[0].plot(steps, q_loss)
        axes[0].set_ylabel('Q Loss')
        axes[0].set_title('Q-Network Loss')
        axes[0].grid(True)

        axes[1].plot(steps, policy_loss)
        axes[1].set_ylabel('Policy Loss')
        axes[1].set_title('Policy Loss')
        axes[1].grid(True)

        axes[2].plot(steps, alpha)
        axes[2].set_ylabel('Alpha')
        axes[2].set_xlabel('Training Step')
        axes[2].set_title('Temperature (α)')
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def print_summary(self):
        """Print training summary statistics"""
        metrics = self.logger.load_metrics()

        if len(metrics) == 0:
            print("No metrics found")
            return

        returns = [m.get('episode_return', 0) for m in metrics]

        print("="*50)
        print("Training Summary")
        print("="*50)
        print(f"Total episodes: {len(metrics)}")
        print(f"Mean return: {np.mean(returns):.4f}")
        print(f"Std return: {np.std(returns):.4f}")
        print(f"Max return: {np.max(returns):.4f}")
        print(f"Min return: {np.min(returns):.4f}")
        print(f"Final 100 avg: {np.mean(returns[-100:]):.4f}")
        print("="*50)


def plot_action_distribution(actions: List[int], save_path: str = None):
    """Plot distribution of selected actions"""
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = np.bincount(actions, minlength=16)
    action_ids = np.arange(16)

    ax.bar(action_ids, counts, color='steelblue')
    ax.set_xlabel('Action ID (4-bit combination)')
    ax.set_ylabel('Count')
    ax.set_title('Action Selection Distribution')
    ax.set_xticks(action_ids)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate percentages
    total = len(actions)
    for i, count in enumerate(counts):
        if count > 0:
            pct = 100.0 * count / total
            ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # Test visualization
    import time

    viz = TrajectoryVisualizer()
    plt.ion()
    plt.show()

    # Simulate data
    for i in range(500):
        t = i * 0.001
        ref = np.sin(2 * np.pi * 1.0 * t)
        actual = ref + np.random.normal(0, 0.1)
        action = np.random.randint(0, 16)
        reward = -abs(ref - actual)

        viz.update(t, ref, actual, action, reward)

        if i % 10 == 0:
            viz.render()
            plt.pause(0.01)

    viz.save('test_trajectory.png')
    print("Saved test trajectory plot")
