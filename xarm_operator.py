"""xarm hardware layer: xarm-python-control Arm class + ROS2 camera streaming.

The hardware control class and `rclpy` are imported lazily so the dry-run path
and unit tests work without hardware or a ROS2 environment.
"""

import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

# xarm-python-control is kept outside this repo. Override with
# XARM_PYTHON_CONTROL_DIR if the checkout moves.
DEFAULT_XARM_CONTROL_DIR = Path("/home/lft-vlai2/Documents/csy/xarm-python-control")

SINGLE_ARM_DIM = 8
DUAL_ARM_DIM = 16

HOME_POSITION = np.array([
    -0.14133669435977936, 0.444007009267807, -0.05349050089716911,
    1.600000023841858, 0.012397955171763897, 0.12722209095954895,
    0.0061951628886163235, -1.100000023841858,
])

SINGLE_CAMERA_TOPICS = {
    "cam_high": "/cam_chest/cam_chest/color/image_raw",
    "cam_right_wrist": "/cam_wrist_right/cam_wrist_right/color/image_raw",
}
DUAL_CAMERA_TOPICS = {
    "cam_high": "/cam_chest/cam_chest/color/image_raw",
    "cam_left_wrist": "/cam_wrist_left/cam_wrist_left/color/image_raw",
    "cam_right_wrist": "/cam_wrist_right/cam_wrist_right/color/image_raw",
}
CAMERA_TOPICS = SINGLE_CAMERA_TOPICS


def _load_arm_control():
    control_dir = Path(os.environ.get("XARM_PYTHON_CONTROL_DIR", DEFAULT_XARM_CONTROL_DIR)).expanduser()
    if not control_dir.exists():
        raise FileNotFoundError(
            f"xarm-python-control not found at {control_dir}; set XARM_PYTHON_CONTROL_DIR"
        )
    control_dir_str = str(control_dir)
    if control_dir_str not in sys.path:
        sys.path.insert(0, control_dir_str)

    try:
        from arm_control import Arm, CONTROL_HZ
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import xarm-python-control. Make sure its dependencies "
            "(python-can, pinocchio, mujoco, scipy, mink) are installed in the "
            f"active Python environment. Source: {control_dir}"
        ) from exc

    return Arm, CONTROL_HZ


def _as_single_arm_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (SINGLE_ARM_DIM,):
        raise ValueError(f"Expected single-arm action shape ({SINGLE_ARM_DIM},), got {action.shape}")
    return action


def _as_dual_arm_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (DUAL_ARM_DIM,):
        raise ValueError(f"Expected dual-arm action shape ({DUAL_ARM_DIM},), got {action.shape}")
    return action


class XarmOperator:
    def __init__(
        self,
        can_interface: str = "can1",
        dry_run: bool = False,
        home_position=None,
        arm_prefix: str = "xarm_right_",
        use_gravity: bool = True,
    ):
        self.can_interface = can_interface
        self.dry_run = dry_run
        raw_home = HOME_POSITION if home_position is None else np.asarray(home_position)
        if dry_run:
            self.home_position = _as_single_arm_action(raw_home)
            self._positions = self.home_position.copy()
            return

        self.home_position = _as_single_arm_action(raw_home)
        Arm, control_hz = _load_arm_control()
        self._control_hz = control_hz
        self.arm = Arm(can_interface, use_gravity=use_gravity, arm_prefix=arm_prefix)
        print(f"xarm-python-control Arm initialized on {can_interface}, prefix={arm_prefix}")

    def read_state(self) -> np.ndarray:
        if self.dry_run:
            return self._positions.copy()

        return self.arm.read_q8(recv_timeout=0.002)

    def send_action(self, action, duration: float | None = None):
        action = _as_single_arm_action(action)
        if self.dry_run:
            self._positions = action
            return

        duration = 1.0 / self._control_hz if duration is None else duration
        self.arm.move_joints(action[:7], duration=duration)
        self.arm.move_gripper(float(action[7]), duration=0.0, n_steps=1)

    def go_home(self, target=None, nstep: int = 220, step_dt: float = 0.01):
        """Move to `target` using the Arm control class."""
        target = self.home_position if target is None else np.asarray(target)
        target = _as_single_arm_action(target)
        if not self.dry_run:
            duration = max(nstep * step_dt, 1.0 / self._control_hz)
            self.arm.move_joints(target[:7], duration=duration)
            self.arm.move_gripper(float(target[7]), duration=min(max(duration, 0.2), 1.0))
            return

        self.send_action(target)

    def shutdown(self):
        if self.dry_run:
            return
        print("Closing xarm-python-control Arm...")
        self.arm.close()


class DualXarmOperator:
    """Controls two xArm instances as one 16-dim dual-arm robot.

    State/action order follows the X1 dual-arm convention:
    [left joint1..joint7, left gripper, right joint1..joint7, right gripper].
    """

    def __init__(
        self,
        left_can_interface: str = "can0",
        right_can_interface: str = "can1",
        dry_run: bool = False,
        left_home=None,
        right_home=None,
    ):
        self.left = XarmOperator(
            left_can_interface,
            dry_run=dry_run,
            home_position=left_home,
            arm_prefix="xarm_left_",
        )
        try:
            self.right = XarmOperator(
                right_can_interface,
                dry_run=dry_run,
                home_position=right_home,
                arm_prefix="xarm_right_",
            )
        except Exception:
            self.left.shutdown()
            raise

    def read_state(self) -> np.ndarray:
        return np.concatenate([self.left.read_state(), self.right.read_state()])

    def send_action(self, action, duration: float | None = None):
        action = _as_dual_arm_action(action)
        errors = []

        def send(operator, single_action):
            try:
                operator.send_action(single_action, duration=duration)
            except Exception as exc:
                errors.append(exc)

        left_thread = threading.Thread(target=send, args=(self.left, action[:SINGLE_ARM_DIM]))
        right_thread = threading.Thread(target=send, args=(self.right, action[SINGLE_ARM_DIM:]))
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        if errors:
            raise errors[0]

    def go_home(self, target=None, nstep: int = 220, step_dt: float = 0.01):
        target = _as_dual_arm_action(
            np.concatenate([self.left.home_position, self.right.home_position])
            if target is None
            else target
        )
        if not self.left.dry_run and not self.right.dry_run:
            errors = []

            def move(operator, single_target):
                try:
                    operator.go_home(single_target, nstep=nstep, step_dt=step_dt)
                except Exception as exc:
                    errors.append(exc)

            left_thread = threading.Thread(target=move, args=(self.left, target[:SINGLE_ARM_DIM]))
            right_thread = threading.Thread(target=move, args=(self.right, target[SINGLE_ARM_DIM:]))
            left_thread.start()
            right_thread.start()
            left_thread.join()
            right_thread.join()
            if errors:
                raise errors[0]
            return

        self.send_action(target)

    def shutdown(self):
        self.left.shutdown()
        self.right.shutdown()


def _ros_image_to_numpy(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image into an HWC uint8 RGB array."""
    height, width = msg.height, msg.width
    encoding = msg.encoding
    img_array = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return img_array.reshape((height, width, 3))
    if encoding == "bgr8":
        return img_array.reshape((height, width, 3))[:, :, ::-1]
    if encoding == "rgba8":
        return img_array.reshape((height, width, 4))[:, :, :3]
    if encoding == "bgra8":
        return img_array.reshape((height, width, 4))[:, :, [2, 1, 0]]
    if encoding == "mono8":
        return np.repeat(img_array.reshape((height, width, 1)), 3, axis=2)
    raise ValueError(f"Unsupported image encoding: {encoding}")


class CameraStreamer:
    """Subscribes to camera topics on a background rclpy thread and caches
    the latest frame per camera."""

    def __init__(self, topic_map: dict | None = None):
        import rclpy
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        topic_map = topic_map or CAMERA_TOPICS
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("x1_openpi_cameras")
        self._lock = threading.Lock()
        self._frames = {}
        self._stamps = {}

        def make_callback(name):
            def callback(msg):
                img = _ros_image_to_numpy(msg)
                with self._lock:
                    self._frames[name] = img
                    self._stamps[name] = time.monotonic()
            return callback

        self._names = list(topic_map.keys())
        for name, topic in topic_map.items():
            self._node.create_subscription(Image, topic, make_callback(name), 10)

        self._thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._thread.start()

    def get_images(self, max_age_s: float = 0.5, verbose: bool = True):
        """Return the latest frames, or None if any camera is missing/stale."""
        now = time.monotonic()
        with self._lock:
            for name in self._names:
                if name not in self._frames:
                    if verbose:
                        print(f"Camera {name} has no frame yet")
                    return None
                if now - self._stamps[name] > max_age_s:
                    if verbose:
                        print(f"Camera {name} frame is stale ({now - self._stamps[name]:.2f}s)")
                    return None
            return {name: self._frames[name].copy() for name in self._names}

    def stop(self):
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        if self._rclpy.ok():
            self._rclpy.shutdown()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass


class FakeCameraStreamer:
    """Dry-run camera source returning random images."""

    def __init__(self, image_keys=None):
        self._image_keys = tuple(image_keys or CAMERA_TOPICS)

    def get_images(self, max_age_s: float = 0.5, verbose: bool = True):
        return {
            name: np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
            for name in self._image_keys
        }

    def stop(self):
        pass
