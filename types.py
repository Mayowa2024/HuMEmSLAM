from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SceneRecord:
    """
    Scene/topology descriptor for one keyframe.

    embedding:
        Scene feature vector from the scene classifier.
    confidence:
        Scene classifier confidence in range [0, 1].
    label:
        human-readable scene label.
    """
    embedding: Optional[np.ndarray]
    confidence: float = 0.0
    label: str = ""
    category_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class TextAnchor:
    """
    OCR text detected on a segmented static object.
    """
    text: str
    conf: float


@dataclass
class StaticObject:
    """
    Static segmented object used for object and text matching.

    class_name:
        YOLO class name.
    seg_conf:
        YOLO segmentation confidence.
    x_centroid, y_centroid:
        Normalised mask centroid in range [0, 1].
    area:
        Normalised mask area: mask_pixels / image_pixels.
    texts:
        OCR text anchors attached to this object.
    """
    class_name: str
    seg_conf: float
    x_centroid: float
    y_centroid: float
    area: float
    texts: List[TextAnchor] = field(default_factory=list)
    observation_count: int = 1


    appearance_embedding: Optional[np.ndarray] = None
    appearance_model: str = ""
    landmark_track_id: Optional[int] = None


@dataclass
class KeyframeRecord:
    """
    Semantic representation of one keyframe.

    """
    keyframe_id: int
    timestamp: float
    scene: SceneRecord
    static_objects: List[StaticObject] = field(default_factory=list)


    pose: Optional[np.ndarray] = None
    orb_keyframe_id: Optional[int] = None
    orb_map_id: Optional[int] = None
    tracking_inliers: Optional[int] = None


    source_frame_id: Optional[int] = None
