import os
import tempfile

from timdr_core import detect_jump, load_last_state, save_last_state, clear_state


def test_detect_jump_flags_only_over_threshold():
    prev = {"temp": 10.0, "pressure": 1010.0}
    new = {"temp": 13.5, "pressure": 1011.0}
    flags = detect_jump(prev, new, thresholds={"temp": 2.0, "pressure": 5.0})
    assert flags == {"temp_jump": True}


def test_detect_jump_missing_keys_ignored():
    prev = {"temp": 10.0}
    new = {"pressure": 1011.0}
    flags = detect_jump(prev, new, thresholds={"temp": 2.0, "pressure": 5.0})
    assert flags == {}


def test_state_persists_to_disk_across_reload():
    path = tempfile.mktemp(suffix=".json")
    try:
        state = {("stacja_a", "2026-08-19"): {"temp": 10.0}}
        save_last_state(path, state)
        assert os.path.exists(path)
        loaded = load_last_state(path)
        assert loaded[("stacja_a", "2026-08-19")]["temp"] == 10.0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_load_last_state_missing_file_returns_empty_not_crash():
    assert load_last_state("/tmp/does_not_exist_" + os.urandom(4).hex() + ".json") == {}


def test_clear_state_removes_file():
    path = tempfile.mktemp(suffix=".json")
    save_last_state(path, {("x", "y"): {"a": 1}})
    assert os.path.exists(path)
    assert clear_state(path) is True
    assert not os.path.exists(path)
