from collections import deque

import numpy as np


class MovingAverageFilter:
    """Sliding-window moving average filter applied to published actions.

    `window == 1` is a passthrough (no filtering). Larger windows smooth more
    but add lag (lag ≈ window/2 steps).
    """

    def __init__(self, window: int):
        self.window = max(1, window)
        self.buf = None

    def reset(self):
        self.buf = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        if self.window <= 1:
            return action
        if self.buf is None:
            self.buf = deque([action.astype(np.float64).copy()], maxlen=self.window)
        else:
            self.buf.append(action.astype(np.float64).copy())
        return np.mean(self.buf, axis=0)


class ActionFilter:
    """Exponential moving average filter applied to published actions.

    `alpha == 1.0` is a passthrough (no filtering). Smaller `alpha` smooths
    more but adds lag. The same `alpha` is applied to every dimension
    (arm joints and gripper).
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        if self.alpha >= 1.0:
            return action
        if self.prev is None:
            self.prev = action.astype(np.float64).copy()
        else:
            self.prev = self.alpha * action + (1 - self.alpha) * self.prev
        return self.prev.copy()


def interpolate_chunk_transition(prev_action, raw_action, step_idx, total_steps):
    """Linearly ramp from `prev_action` toward `raw_action`.

    `step_idx` is 0-indexed within the transition window; at
    `step_idx == total_steps - 1` the result equals `raw_action`.
    """
    weight = (step_idx + 1) / total_steps
    return (1 - weight) * prev_action + weight * raw_action


DIM_LABELS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]


def save_filter_trajectories(raw, filt, alpha, save_path):
    """Save raw/filtered action trajectories (steps, 8) to an `.npz` file."""
    np.savez(save_path, raw=raw, filt=filt, alpha=np.array(alpha))


def plot_filter_comparison(raw, filt, alpha, save_path):
    """Save a raw-vs-filtered action comparison plot. `raw`/`filt`: (steps, 8)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(DIM_LABELS)
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3), squeeze=False)
    steps = np.arange(raw.shape[0])
    for col in range(ncols):
        ax = axes[0][col]
        ax.plot(steps, raw[:, col], "k--", label="raw")
        ax.plot(steps, filt[:, col], "b-", label="filtered")
        ax.set_title(DIM_LABELS[col])
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Action filter comparison (alpha={alpha})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
