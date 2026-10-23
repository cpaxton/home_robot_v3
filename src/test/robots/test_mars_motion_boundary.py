import math

import pytest

from emet.robots.innate_mars.motion_boundary import MotionBoundary


def test_start_frame_wrap_and_latched_overrun():
    guard = MotionBoundary((1, 2, math.radians(179)), translation_limit_m=0.1)
    assert guard.check((1, 2, math.radians(-137)), state_age_s=0) is None
    fault = guard.check((1, 2, math.radians(-133)), state_age_s=0)
    assert "yaw boundary" in fault
    assert guard.check((1, 2, math.radians(179)), state_age_s=0) == fault


@pytest.mark.parametrize("pose,age", [((0.121, 0, 0), 0), ((0, 0, 0), .201),
                                      ((0, 0, float('nan')), 0), ((0, 0), 0)])
def test_violation_stops_before_exit(pose, age, capsys):
    guard = MotionBoundary((0, 0, 0), translation_limit_m=.1)
    calls = []
    with pytest.raises(SystemExit) as exc:
        guard.enforce(pose, state_age_s=age, stop=lambda: calls.append('stop') or True)
    assert exc.value.code == 2
    assert calls == ['stop']
    assert 'ERROR:' in capsys.readouterr().err


def test_stop_failure_still_reports_and_exits(capsys):
    guard = MotionBoundary((0, 0, 0), translation_limit_m=0)
    def fail():
        raise RuntimeError('disconnected')
    with pytest.raises(SystemExit):
        guard.enforce((1, 0, 0), state_age_s=0, stop=fail)
    assert 'stop_confirmed=False' in capsys.readouterr().err
