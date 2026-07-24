import numpy as np

from xarm_operator import (
    DUAL_CAMERA_TOPICS,
    DualXarmOperator,
    HOME_POSITION,
    FakeCameraStreamer,
    XarmOperator,
)


def test_dry_run_send_then_read_roundtrip():
    op = XarmOperator(dry_run=True)
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, -0.5])

    op.send_action(target)
    state = op.read_state()

    assert state.shape == (8,)
    np.testing.assert_allclose(state, target)


def test_dual_dry_run_send_then_read_roundtrip():
    op = DualXarmOperator(dry_run=True)
    target = np.concatenate(
        [
            np.array([0.1, 0.1, 0.3, 0.4, 0.1, 0.1, 0.1, -0.5]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.2, 0.2, 0.2, -0.4]),
        ]
    )

    op.send_action(target)

    np.testing.assert_allclose(op.read_state(), target)


def test_dry_run_send_records_action_directly():
    op = XarmOperator(dry_run=True)
    action = np.zeros(8)
    action[0] = -99.0

    op.send_action(action)

    assert op.read_state()[0] == -99.0


def test_send_action_rejects_wrong_shape():
    op = XarmOperator(dry_run=True)

    try:
        op.send_action(np.zeros(7))
    except ValueError as exc:
        assert "Expected single-arm action shape" in str(exc)
    else:
        raise AssertionError("Expected wrong-shape action to raise ValueError")


def test_go_home_dry_run_reaches_target():
    op = XarmOperator(dry_run=True)
    op.send_action(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5]))

    op.go_home(nstep=5, step_dt=0.0)

    np.testing.assert_allclose(op.read_state(), HOME_POSITION, atol=1e-9)


def test_dual_go_home_dry_run_reaches_target():
    left_home = np.array([0.1, 0.1, 0.0, 0.4, 0.0, 0.0, 0.0, -0.8])
    right_home = np.array([-0.1, 0.4, 0.0, 0.5, 0.0, 0.0, 0.0, -0.7])
    op = DualXarmOperator(dry_run=True, left_home=left_home, right_home=right_home)
    op.send_action(np.concatenate([left_home + 0.05, right_home + 0.05]))

    op.go_home(nstep=5, step_dt=0.0)

    np.testing.assert_allclose(op.read_state(), np.concatenate([left_home, right_home]), atol=1e-9)


def test_fake_camera_streamer_returns_images():
    cams = FakeCameraStreamer()
    images = cams.get_images()

    assert set(images.keys()) == {"cam_high", "cam_right_wrist"}
    for img in images.values():
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8
    cams.stop()


def test_fake_camera_streamer_can_return_dual_images():
    cams = FakeCameraStreamer(DUAL_CAMERA_TOPICS)
    images = cams.get_images()

    assert set(images.keys()) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    cams.stop()
