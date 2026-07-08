"""xarm hardware layer: direct CAN control (xarm_can) + ROS2 camera streaming.

`xarm_can` and `rclpy` are imported lazily so the dry-run path (and unit
tests) work without hardware or a ROS2 environment.
"""

import threading
import time

import numpy as np

# 关节限位 (rad)，来自 x_air URDF
JOINT_LIMITS_LOWER = np.array([-1.3, -0.1, -1.5, 0.0, -1.5, -0.7, -1.5])
JOINT_LIMITS_UPPER = np.array([3.4, 1.7, 1.5, 2.4, 1.5, 0.7, 1.5])
# 夹爪限位 (rad)：-1.1 全开，0.0 闭合；与 LeRobot 数据集最后一维对齐
GRIPPER_LOWER, GRIPPER_UPPER = -1.1, 0.0

# MIT 控制增益（x_air xarm_deploy_direct.py 现值）
DEFAULT_KP = [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0]
DEFAULT_KD = [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2]
GRIPPER_KP, GRIPPER_KD = 16.0, 0.3
HOME_GRIPPER_KP, HOME_GRIPPER_KD = 10.0, 0.5

HOME_POSITION = np.array([
    -0.14133669435977936, 0.444007009267807, -0.05349050089716911,
    1.600000023841858, 0.012397955171763897, 0.12722209095954895,
    0.0061951628886163235, -1.100000023841858,
])

CAMERA_TOPICS = {
    "cam_high": "/cam_chest/cam_chest/color/image_raw",
    "cam_right_wrist": "/cam_wrist_right/cam_wrist_right/color/image_raw",
}


def clip_action(action: np.ndarray) -> np.ndarray:
    """Clip an 8-dim action to joint and gripper limits."""
    action = np.asarray(action, dtype=np.float64).copy()
    action[:7] = np.clip(action[:7], JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    action[7] = np.clip(action[7], GRIPPER_LOWER, GRIPPER_UPPER)
    return action


class _MockArm:
    """Dry-run stand-in for the CAN hardware: positions track commands."""

    def __init__(self):
        self.positions = HOME_POSITION.copy()


class XarmOperator:
    def __init__(self, can_interface: str = "can1", dry_run: bool = False):
        self.dry_run = dry_run
        if dry_run:
            self._mock = _MockArm()
            return

        import xarm_can as oa

        self._oa = oa
        self.arm = oa.XArm(can_interface, True)  # True = CAN-FD
        motor_types = [
            oa.MotorType.DM8009, oa.MotorType.DM8009,
            oa.MotorType.DM4340, oa.MotorType.DM4340,
            oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
        ]
        send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
        recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
        self.arm.init_arm_motors(motor_types, send_ids, recv_ids)
        self.arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
        self.arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.arm.enable_all()
        time.sleep(0.1)
        self.arm.recv_all()
        self.arm.refresh_all()
        self.arm.recv_all()
        n_motors = len(self.arm.get_arm().get_motors())
        print(f"CAN hardware initialized on {can_interface}: {n_motors} arm motors")

    def read_state(self) -> np.ndarray:
        if self.dry_run:
            return self._mock.positions.copy()

        self.arm.refresh_all()
        self.arm.recv_all()
        joints = [m.get_position() for m in self.arm.get_arm().get_motors()]
        grippers = self.arm.get_gripper().get_motors()
        gripper = grippers[0].get_position() if grippers else 0.0
        return np.array(joints + [gripper], dtype=np.float64)

    def send_action(self, action, kp=None, kd=None, gripper_kp=GRIPPER_KP, gripper_kd=GRIPPER_KD):
        action = clip_action(action)
        if self.dry_run:
            self._mock.positions = action
            return

        kp = kp if kp is not None else DEFAULT_KP
        kd = kd if kd is not None else DEFAULT_KD
        oa = self._oa
        arm_params = [
            oa.MITParam(kp[i], kd[i], float(action[i]), 0.0, 0.0) for i in range(7)
        ]
        self.arm.get_arm().mit_control_all(arm_params)
        self.arm.get_gripper().mit_control_all(
            [oa.MITParam(gripper_kp, gripper_kd, float(action[7]), 0.0, 0.0)]
        )
        # 等待 CAN 总线处理后接收反馈（参考 x_air 遥操作脚本）
        time.sleep(0.0002)
        self.arm.recv_all()

    def go_home(self, target=None, nstep: int = 220, step_dt: float = 0.01):
        """Smoothly interpolate from the current position to `target`."""
        target = clip_action(HOME_POSITION if target is None else np.asarray(target))
        current = self.read_state()
        for step in range(nstep):
            alpha = (step + 1) / nstep
            self.send_action(
                (1 - alpha) * current + alpha * target,
                gripper_kp=HOME_GRIPPER_KP,
                gripper_kd=HOME_GRIPPER_KD,
            )
            if step_dt > 0:
                time.sleep(step_dt)

    def shutdown(self):
        if self.dry_run:
            return
        print("Disabling motors...")
        self.arm.disable_all()
        self.arm.recv_all()


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
        self._node = rclpy.create_node("xarm_aio_cameras")
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

    def get_images(self, max_age_s: float = 0.5, verbose: bool = True):
        return {
            name: np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
            for name in CAMERA_TOPICS
        }

    def stop(self):
        pass
