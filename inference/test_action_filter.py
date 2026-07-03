import numpy as np

from action_filter import (
    ActionFilter,
    MovingAverageFilter,
    interpolate_chunk_transition,
    plot_filter_comparison,
    save_filter_trajectories,
)


def test_ema_passthrough_when_alpha_one():
    f = ActionFilter(alpha=1.0)
    action = np.arange(8, dtype=np.float64)
    out = f.apply(action)
    np.testing.assert_array_equal(out, action)


def test_ema_smooths_second_sample():
    f = ActionFilter(alpha=0.5)
    first = np.zeros(8)
    second = np.ones(8)
    f.apply(first)
    out = f.apply(second)
    np.testing.assert_allclose(out, np.full(8, 0.5))


def test_ema_reset_clears_state():
    f = ActionFilter(alpha=0.5)
    f.apply(np.zeros(8))
    f.reset()
    out = f.apply(np.ones(8))
    np.testing.assert_allclose(out, np.ones(8))


def test_moving_average_passthrough_when_window_one():
    f = MovingAverageFilter(window=1)
    action = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(f.apply(action), action)


def test_moving_average_window_math():
    f = MovingAverageFilter(window=3)
    f.apply(np.zeros(8))
    f.apply(np.ones(8))
    out = f.apply(np.full(8, 2.0))
    np.testing.assert_allclose(out, np.ones(8))  # mean(0,1,2)


def test_interpolate_chunk_transition_first_step():
    prev = np.zeros(8)
    raw = np.ones(8)
    out = interpolate_chunk_transition(prev, raw, step_idx=0, total_steps=5)
    np.testing.assert_allclose(out, np.full(8, 0.2))


def test_interpolate_chunk_transition_last_step_equals_raw():
    prev = np.zeros(8)
    raw = np.ones(8)
    out = interpolate_chunk_transition(prev, raw, step_idx=4, total_steps=5)
    np.testing.assert_allclose(out, raw)


def test_plot_and_save_trajectories(tmp_path):
    steps = 5
    raw = np.random.rand(steps, 8)
    filt = np.random.rand(steps, 8)
    png = tmp_path / "cmp.png"
    npz = tmp_path / "cmp.npz"

    plot_filter_comparison(raw, filt, alpha=0.5, save_path=str(png))
    save_filter_trajectories(raw, filt, alpha=0.5, save_path=str(npz))

    assert png.exists() and png.stat().st_size > 0
    data = np.load(npz)
    np.testing.assert_array_equal(data["raw"], raw)
    np.testing.assert_array_equal(data["filt"], filt)
    assert float(data["alpha"]) == 0.5
