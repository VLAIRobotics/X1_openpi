import numpy as np

from xarm_operator import (
    HOME_POSITION,
    FakeCameraStreamer,
    XarmOperator,
    clip_action,
)


def test_clip_action_bounds():
    action = np.array([-5.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5])
    out = clip_action(action)

    assert out[0] == -1.3   # joint1 lower
    assert out[1] == 1.7    # joint2 upper
    assert out[3] == 1.0    # 未越界不变
    assert out[7] == 0.0    # gripper upper


def test_clip_action_gripper_lower():
    action = np.zeros(8)
    action[7] = -2.0
    assert clip_action(action)[7] == -1.0


def test_dry_run_send_then_read_roundtrip():
    op = XarmOperator(dry_run=True)
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, -0.5])

    op.send_action(target)
    state = op.read_state()

    assert state.shape == (8,)
    np.testing.assert_allclose(state, target)


def test_dry_run_send_clips_before_apply():
    op = XarmOperator(dry_run=True)
    action = np.zeros(8)
    action[0] = -99.0

    op.send_action(action)

    assert op.read_state()[0] == -1.3


def test_go_home_dry_run_reaches_target():
    op = XarmOperator(dry_run=True)
    op.send_action(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5]))

    op.go_home(nstep=5, step_dt=0.0)

    np.testing.assert_allclose(op.read_state(), HOME_POSITION, atol=1e-9)


def test_fake_camera_streamer_returns_images():
    cams = FakeCameraStreamer()
    images = cams.get_images()

    assert set(images.keys()) == {"cam_high", "cam_right_wrist"}
    for img in images.values():
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8
    cams.stop()
