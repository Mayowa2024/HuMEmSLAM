from slam.ocr_policy import should_refresh_cached_ocr, should_run_ocr_for_object
import numpy as np

from slam.human_slam_node import ocr_crop_is_usable, useful_ocr_text


OCR_CLASSES = {"building", "traffic_sign", "information_sign"}
ALWAYS = {"traffic_sign", "information_sign"}


def test_sign_triggers_ocr_outside_periodic_interval():
    assert should_run_ocr_for_object(
        False, "traffic_sign", OCR_CLASSES, ALWAYS, 0, 3
    )


def test_building_remains_interval_controlled():
    assert not should_run_ocr_for_object(
        False, "building", OCR_CLASSES, ALWAYS, 0, 3
    )
    assert should_run_ocr_for_object(
        True, "building", OCR_CLASSES, ALWAYS, 0, 3
    )


def test_non_ocr_class_is_never_processed():
    assert not should_run_ocr_for_object(
        True, "bridge", OCR_CLASSES, ALWAYS, 0, 3
    )


def test_per_keyframe_budget_is_enforced():
    assert not should_run_ocr_for_object(
        True, "traffic_sign", OCR_CLASSES, ALWAYS, 3, 3
    )


def test_ocr_quality_gate_rejects_punctuation_and_single_characters():
    assert not useful_ocr_text("*")
    assert not useful_ocr_text("@")
    assert not useful_ocr_text("0")
    assert useful_ocr_text("12%")
    assert useful_ocr_text("A4")


def test_ocr_crop_quality_gate_rejects_small_or_uniform_regions():
    assert not ocr_crop_is_usable(np.zeros((8, 40, 3), dtype=np.uint8))
    assert not ocr_crop_is_usable(np.zeros((20, 20, 3), dtype=np.uint8))


def test_ocr_crop_quality_gate_accepts_sharp_text_like_region():
    crop = np.zeros((40, 80, 3), dtype=np.uint8)
    crop[:, 40:] = 255
    assert ocr_crop_is_usable(crop)


def test_cached_text_is_reused_until_refresh_age_or_better_crop():
    arguments = dict(
        has_text=True,
        current_frame=12,
        last_ocr_frame=10,
        current_quality=110.0,
        best_quality=100.0,
        refresh_interval=20,
        retry_interval=5,
        quality_gain=1.35,
    )
    assert not should_refresh_cached_ocr(**arguments)
    arguments["current_quality"] = 140.0
    assert should_refresh_cached_ocr(**arguments)
    arguments["current_quality"] = 100.0
    arguments["current_frame"] = 30
    assert should_refresh_cached_ocr(**arguments)


def test_failed_cached_read_retries_sooner_than_successful_text():
    assert not should_refresh_cached_ocr(
        False, 13, 10, 100.0, 100.0, 20, 5, 1.35
    )
    assert should_refresh_cached_ocr(
        False, 15, 10, 100.0, 100.0, 20, 5, 1.35
    )
