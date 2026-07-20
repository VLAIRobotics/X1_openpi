import argparse
import os
import signal
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from action_filter import (
    ActionFilter,
    MovingAverageFilter,
    interpolate_chunk_transition,
    plot_filter_comparison,
    save_filter_trajectories,
)
from clients import OpenpiClient
from utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from xarm_operator import CameraStreamer, DualXarmOperator, FakeCameraStreamer, XarmOperator

shutdown_event = threading.Event()


def _on_sigint(signum, frame):
    shutdown_event.set()


def get_observation(operator, cameras, verbose=True):
    try:
        images = cameras.get_images(verbose=verbose)
    except TypeError:
        images = cameras.get_images()
    if images is None:
        return None
    return {"images": images, "state": operator.read_state()}


def wait_for_observation(operator, cameras, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    next_report = 0.0

    while not shutdown_event.is_set():
        observation = get_observation(operator, cameras, verbose=False)
        if observation is not None:
            print("Camera frames ready")
            return observation

        now = time.monotonic()
        if now >= deadline:
            return None
        if now >= next_report:
            print(f"Waiting for camera frames... ({deadline - now:.1f}s left)")
            next_report = now + 1.0
        time.sleep(0.1)

    return None


def inference_fn(policy, operator, cameras, config):
    observation = get_observation(operator, cameras)
    if observation is None:
        return None

    payload = {
        "state": observation["state"],
        "instruction": config["language_instruction"],
    }
    for name in config["image_keys"]:
        payload[name] = observation["images"][name]
    start_time = time.perf_counter()
    actions = policy.predict_action(payload)
    print(f"Model inference time: {(time.perf_counter() - start_time) * 1000:.1f} ms")
    return actions


def maybe_save_filter_plot(args, config, raw_hist, filt_hist):
    if not args.plot_filter or not raw_hist:
        return
    save_dir = args.save_dir if args.save_dir else "/tmp"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(save_dir, f"action_filter_{config['task']}_{timestamp}")
    plot_filter_comparison(
        np.array(raw_hist), np.array(filt_hist), config["action_filter_alpha"], base + ".png"
    )
    save_filter_trajectories(
        np.array(raw_hist), np.array(filt_hist), config["action_filter_alpha"], base + ".npz"
    )
    print(f"Saved action filter comparison to {base}.png / .npz")


def model_inference(args, config, operator, cameras, policy=None):
    if policy is None:
        policy = OpenpiClient(
            host=args.host,
            port=args.port,
            state_dim=config["state_dim"],
            image_keys=config["image_keys"],
        )

    chunk_size = config["chunk_size"]
    home = np.asarray(config["home"])

    if config["action_filter_type"] == "moving_average":
        action_filter = MovingAverageFilter(config["action_filter_window"])
    else:
        action_filter = ActionFilter(config["action_filter_alpha"])

    sub_steps = max(1, round(args.mit_rate / args.publish_rate))
    step_period = 1.0 / args.publish_rate
    episodes_done = 0
    moved_home = False

    try:
        print("Warmup the server...")
        policy.warmup()
        print("Server warmed up")

        if not args.dry_run and args.camera_wait_s > 0:
            observation = wait_for_observation(operator, cameras, args.camera_wait_s)
            if observation is None:
                raise RuntimeError(
                    "Camera frames unavailable; check ROS2 camera topics before moving home"
                )

        if not args.auto_start:
            input("Press enter to move home and start")
        operator.go_home(home)
        moved_home = True
        task_time = time.time()

        while not shutdown_event.is_set():
            t = 0
            action_filter.reset()
            action_buffer = None
            raw_hist, filt_hist = [], []
            chunk_transition_steps = config["chunk_transition_steps"]
            transition_counter = chunk_transition_steps
            prev_action = None
            prev_published = None
            episode_closed = False

            while t < config["episode_len"] and not shutdown_event.is_set():
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        maybe_save_filter_plot(args, config, raw_hist, filt_hist)
                        episode_closed = True
                        operator.go_home(home)
                        if not args.auto_start:
                            input("Press enter to continue")
                        task_time = time.time()
                        break
                    elif result == "quit":
                        maybe_save_filter_plot(args, config, raw_hist, filt_hist)
                        return

                if t % chunk_size == 0:
                    action_buffer = inference_fn(policy, operator, cameras, config)
                    if action_buffer is None:
                        print("Observation unavailable, ending episode")
                        break
                    assert action_buffer.shape[0] >= chunk_size, (
                        f"Action chunk length {action_buffer.shape[0]} < {chunk_size}"
                    )
                    if t > 0:
                        transition_counter = 0

                step_start = time.perf_counter()
                action = process_action(config["task"], action_buffer[t % chunk_size].copy())
                raw_hist.append(action.copy())

                if (
                    chunk_transition_steps > 0
                    and transition_counter < chunk_transition_steps
                    and prev_action is not None
                ):
                    action = interpolate_chunk_transition(
                        prev_action, action, transition_counter, chunk_transition_steps
                    )
                    transition_counter += 1

                action = action_filter.apply(action)
                filt_hist.append(action.copy())
                prev_action = action.copy()

                start = prev_published if prev_published is not None else action
                for s in range(sub_steps):
                    alpha = (s + 1) / sub_steps
                    operator.send_action((1 - alpha) * start + alpha * action)
                    target_time = step_start + (s + 1) * step_period / sub_steps
                    sleep_s = target_time - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                prev_published = action.copy()

                t += 1
                print("Published Step", t)

            if not episode_closed:
                maybe_save_filter_plot(args, config, raw_hist, filt_hist)
            episodes_done += 1
            if args.max_episodes and episodes_done >= args.max_episodes:
                return
            if shutdown_event.is_set():
                return
    finally:
        if moved_home:
            operator.go_home(home)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name in task_configs.yaml")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="openpi server host")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port")
    parser.add_argument("--max_publish_step", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--chunk_size", type=int, default=50, help="Action chunk size")
    parser.add_argument("--publish_rate", type=int, default=30, help="Action step rate (Hz)")
    parser.add_argument("--mit_rate", type=int, default=500, help="MIT command rate (Hz)")
    parser.add_argument("--can_interface", type=str, default="can1", help="CAN interface name")
    parser.add_argument("--left_can_interface", type=str, default="can0", help="Left arm CAN interface")
    parser.add_argument(
        "--right_can_interface",
        type=str,
        default=None,
        help="Right arm CAN interface; defaults to --can_interface",
    )
    parser.add_argument("--max_episodes", type=int, default=0, help="Stop after N episodes (0 = infinite)")
    parser.add_argument("--auto_start", action="store_true", default=False, help="Skip enter-to-continue prompts")
    parser.add_argument("--plot_filter", action="store_true", default=False, help="Save raw-vs-filtered plot per episode")
    parser.add_argument("--save_dir", type=str, default="", help="Directory for --plot_filter outputs")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Mock hardware and cameras")
    parser.add_argument(
        "--camera_wait_s",
        type=float,
        default=15.0,
        help="Wait this many seconds for all camera frames before moving home",
    )
    parser.add_argument(
        "--cam_high_topic", type=str, default="/cam_chest/cam_chest/color/image_raw"
    )
    parser.add_argument(
        "--cam_right_wrist_topic",
        type=str,
        default="/cam_wrist_right/cam_wrist_right/color/image_raw",
    )
    parser.add_argument(
        "--cam_left_wrist_topic",
        type=str,
        default="/cam_wrist_left/cam_wrist_left/color/image_raw",
    )
    parser.add_argument("--cam_low_topic", type=str, default="", help="Optional low/base camera topic")
    return parser


def _topic_map_from_args(args, image_keys):
    topics = {
        "cam_high": args.cam_high_topic,
        "cam_left_wrist": args.cam_left_wrist_topic,
        "cam_right_wrist": args.cam_right_wrist_topic,
    }
    if args.cam_low_topic:
        topics["cam_low"] = args.cam_low_topic
    missing = [name for name in image_keys if name not in topics]
    if missing:
        raise ValueError(f"No ROS topic argument is configured for image keys: {missing}")
    return {name: topics[name] for name in image_keys}


def main():
    operator = None
    cameras = None
    old_settings = None

    try:
        args = build_arg_parser().parse_args()
        config = get_config(args)

        signal.signal(signal.SIGINT, _on_sigint)

        right_can_interface = args.right_can_interface or args.can_interface
        if config["robot_mode"] == "dual":
            operator = DualXarmOperator(
                left_can_interface=args.left_can_interface,
                right_can_interface=right_can_interface,
                dry_run=args.dry_run,
                left_home=config["left0"],
                right_home=config["right0"],
            )
        else:
            operator = XarmOperator(
                can_interface=right_can_interface,
                dry_run=args.dry_run,
                home_position=config["right0"],
            )
        if args.dry_run:
            cameras = FakeCameraStreamer(config["image_keys"])
        else:
            cameras = CameraStreamer(_topic_map_from_args(args, config["image_keys"]))

        import termios
        import tty

        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

        model_inference(args, config, operator, cameras)
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if operator is not None:
            operator.shutdown()
        if cameras is not None:
            cameras.stop()


if __name__ == "__main__":
    main()
