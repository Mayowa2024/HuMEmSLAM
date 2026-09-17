def should_run_ocr_for_object(
    periodic_ocr_enabled,
    class_name,
    ocr_classes,
    always_ocr_classes,
    processed_count,
    maximum_objects,
):
    """Apply the bounded event-driven OCR policy for one detected object."""
    return (
        class_name in ocr_classes
        and processed_count < maximum_objects
        and (periodic_ocr_enabled or class_name in always_ocr_classes)
    )


def should_refresh_cached_ocr(
    has_text,
    current_frame,
    last_ocr_frame,
    current_quality,
    best_quality,
    refresh_interval,
    retry_interval,
    quality_gain,
):
    """Decide whether a tracked landmark needs another expensive OCR call."""
    age = max(0, int(current_frame) - int(last_ocr_frame))
    if not has_text:
        return age >= max(1, int(retry_interval))
    if age >= max(1, int(refresh_interval)):
        return True
    reference = max(1e-6, float(best_quality))
    return float(current_quality) >= reference * max(1.0, float(quality_gain))
