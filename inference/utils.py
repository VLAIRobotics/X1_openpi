import select
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TASK_CONFIG_PATH = Path(__file__).with_name("task_configs.yaml")


def _load_task_configs():
    with TASK_CONFIG_PATH.open("r", encoding="utf-8") as file:
        task_configs = yaml.safe_load(file)
    if not isinstance(task_configs, dict):
        raise ValueError(f"Invalid task config format in {TASK_CONFIG_PATH}")
    return task_configs


TASK_CONFIGS = _load_task_configs()


def get_config(args):
    task_config = TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {args.task}")

    language_instruction = task_config.get("language_instruction")
    right0 = task_config.get("right0")
    if language_instruction is None or right0 is None:
        raise ValueError(f"Task config for {args.task} is missing required fields")
    if len(right0) != 8:
        raise ValueError(f"right0 for {args.task} must have 8 dims, got {len(right0)}")

    return {
        "episode_len": args.max_publish_step,
        "state_dim": 8,
        "right0": right0,
        "action_postprocess": task_config.get("action_postprocess", {}),
        "task": args.task,
        "language_instruction": language_instruction,
        "chunk_size": args.chunk_size,
        "action_filter_alpha": task_config.get("action_filter_alpha", 1.0),
        "action_filter_type": task_config.get("action_filter_type", "ema"),
        "action_filter_window": task_config.get("action_filter_window", 5),
        "chunk_transition_steps": task_config.get("chunk_transition_steps", 0),
    }


def _apply_gripper_rules(gripper_value, rules):
    for rule in rules:
        condition = rule.get("when")
        threshold = rule.get("threshold")
        if condition == "below" and not gripper_value < threshold:
            continue
        if condition == "above" and not gripper_value > threshold:
            continue

        if "set" in rule:
            gripper_value = rule["set"]
        if "add" in rule:
            gripper_value += rule["add"]
    return gripper_value


def process_action(task, action):
    action = action.copy()
    task_config = TASK_CONFIGS.get(task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {task}")
    rules = task_config.get("action_postprocess", {}).get("gripper", [])
    action[7] = _apply_gripper_rules(action[7], rules)
    return action


def check_keyboard_input():
    """Check if a key was pressed without blocking.

    Returns None when stdin is not a tty (e.g. under pytest), where
    select() on the replaced stdin object would raise.
    """
    try:
        if not sys.stdin.isatty():
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    except (OSError, ValueError):
        return None
    return None


def handle_interactive_mode(task_time):
    """Handle interactive mode when space is pressed.

    Returns: 'continue' to resume, 'reset' to restart, 'quit' to stop.
    """
    print("\n" + "=" * 50)
    print(f"Task time: {time.time() - task_time:.1f} s")
    print("INTERACTIVE MODE")
    print("  'c' - Continue running")
    print("  'r' - Reset to starting point and restart")
    print("  'q' - Quit/Stop")
    print("=" * 50)

    while True:
        key = sys.stdin.read(1).lower()
        if key == "c":
            print("Continuing...")
            return "continue"
        elif key == "r":
            print("Restarting...")
            return "reset"
        elif key == "q":
            print("Stopping...")
            return "quit"
