import pytest

from motiongate.detector import UltralyticsPersonDetector


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_size": 0}, "image_size"),
        ({"image_size": 640.5}, "image_size"),
        ({"person_class_id": -1}, "person_class_id"),
        ({"person_class_id": True}, "person_class_id"),
        ({"confidence": 1.1}, "confidence"),
    ],
)
def test_detector_rejects_invalid_configuration_before_loading_model(kwargs, message):
    with pytest.raises(ValueError, match=message):
        UltralyticsPersonDetector(**kwargs)
