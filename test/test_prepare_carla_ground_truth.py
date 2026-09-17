import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_carla_ground_truth.py"
SPEC = importlib.util.spec_from_file_location("prepare_carla_ground_truth", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_zero_rotation_maps_position_to_optical_basis():
    pose = MODULE.convert_row(
        {"x": "1", "y": "2", "z": "3", "roll": "0", "pitch": "0", "yaw": "0"}
    )
    assert np.allclose(pose[:, :3], np.eye(3))
    assert np.allclose(pose[:, 3], [2, -3, 1])


def test_converted_rotation_remains_proper():
    pose = MODULE.convert_row(
        {"x": "0", "y": "0", "z": "0", "roll": "7", "pitch": "-4", "yaw": "35"}
    )
    rotation = pose[:, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)
