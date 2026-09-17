import numpy as np

from slam.dense_object_descriptor import pool_mask_features


def test_all_pooling_modes_return_unit_descriptors():
    features = np.arange(8 * 6 * 6, dtype=np.float32).reshape(8, 6, 6)
    mask = np.zeros((60, 60), np.uint8)
    mask[10:50, 15:45] = 1
    for mode in ("centre_3x3", "centre_5x5", "mask", "mask_central"):
        descriptor = pool_mask_features(features, mask, mode)
        assert descriptor.shape == (8,)
        assert np.isclose(np.linalg.norm(descriptor), 1.0)


def test_empty_mask_has_no_descriptor():
    assert pool_mask_features(
        np.ones((4, 3, 3), np.float32), np.zeros((20, 20), np.uint8)
    ) is None
