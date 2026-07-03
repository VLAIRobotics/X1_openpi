import argparse

import numpy as np
import pytest

from utils import get_config, process_action


def _args(task="example_task"):
    return argparse.Namespace(task=task, max_publish_step=1000, chunk_size=50)


def test_get_config_returns_required_fields():
    config = get_config(_args())

    assert config["state_dim"] == 8
    assert len(config["right0"]) == 8
    assert config["language_instruction"] == "pick up the object"
    assert config["chunk_size"] == 50
    assert config["episode_len"] == 1000
    assert config["action_filter_alpha"] == 1.0
    assert config["action_filter_type"] == "ema"
    assert config["chunk_transition_steps"] == 10


def test_get_config_unknown_task_raises():
    with pytest.raises(ValueError):
        get_config(_args(task="no_such_task"))


def test_process_action_gripper_rule_below_set(monkeypatch):
    import utils

    monkeypatch.setitem(
        utils.TASK_CONFIGS,
        "gripper_rule_task",
        {
            "language_instruction": "x",
            "right0": [0.0] * 8,
            "action_postprocess": {
                "gripper": [{"when": "below", "threshold": -0.5, "set": -1.0}]
            },
        },
    )
    action = np.zeros(8)
    action[7] = -0.6

    out = process_action("gripper_rule_task", action)

    assert out[7] == -1.0
    assert np.all(out[:7] == 0.0)


def test_process_action_no_rules_is_identity():
    action = np.arange(8, dtype=np.float64)
    out = process_action("example_task", action)
    np.testing.assert_array_equal(out, action)
