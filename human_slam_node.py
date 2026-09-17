import csv
import cv2
import numpy as np
import rclpy
import tensorrt as trt
import threading
import time

from collections import deque
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from human_slam_interfaces.msg import OrbSlamFrame, SemanticCandidates
from pathlib import Path
from queue import Empty, Full, Queue
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO

from slam.cognitive_math_model import CognitiveMathModel
from slam.global_place_descriptor import (
    DescriptorSpec,
    TensorRTGlobalDescriptor,
)
from slam.dense_object_descriptor import (
    DenseDescriptorSpec,
    TensorRTDenseDescriptor,
    pool_mask_features,
)
from slam.ocr_policy import should_refresh_cached_ocr, should_run_ocr_for_object
from slam.ocr_onnx_runtime import TensorRTPPOCRv5
from slam.scene_categories import group_places365_probabilities
from slam.types import KeyframeRecord, SceneRecord, StaticObject, TextAnchor


def useful_ocr_text(value: str) -> bool:
    """Reject OCR fragments that cannot identify a landmark."""
    compact = "".join(character for character in str(value) if character.isalnum())
    return len(compact) >= 2


def ocr_crop_is_usable(
    crop,
    minimum_side=12,
    minimum_pixels=256,
    minimum_sharpness=8.0,
):
    """Reject crops that are too small or blurred to produce useful OCR."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return False
    height, width = crop.shape[:2]
    if min(height, width) < max(1, int(minimum_side)):
        return False
    if height * width < max(1, int(minimum_pixels)):
        return False
    threshold = max(0.0, float(minimum_sharpness))
    if threshold <= 0.0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= threshold


def should_run_candidate_retrieval(
    is_keyframe: bool,
    keyframe_id: int,
    tracking_state: int,
    periodic_loop_keyframe_interval: int,
) -> bool:
    """Schedule periodic loop search while preserving immediate recovery."""
    if tracking_state >= 3:
        return True
    if not is_keyframe:
        return False
    interval = max(1, int(periodic_loop_keyframe_interval))
    return keyframe_id % interval == 0


def candidates_above_threshold(ranked_candidates, threshold, limit):
    """Return only candidates strictly above the ORB submission threshold."""
    return [
        item for item in ranked_candidates
        if float(item[1]) > float(
            item[2].get("submission_threshold", threshold)
        )
    ][:max(0, int(limit))]


def preselect_for_expensive_rerank(
    candidate_keyframes,
    candidate_scene_sequences,
    limit,
    enabled=True,
):
    """Keep the EigenPlaces-ordered prefix for full tri-layer scoring."""
    if not enabled:
        return candidate_keyframes, candidate_scene_sequences
    count = max(1, int(limit))
    return candidate_keyframes[:count], candidate_scene_sequences[:count]


def apply_weak_scene_consensus_policy(
    ranked_candidates,
    semantic_threshold,
    effective_score,
    enabled=True,
    raw_threshold=0.55,
    top_k=5,
    minimum_support=3,
    cluster_width_frames=30,
):
    """Promote one weak scene-only historical cluster to ORB geometry.

    Normal above-threshold results always take precedence. Temporal exclusion
    is performed before ranking by ``build_candidate_inputs``. This function
    only supplies a conservative fallback when several top retrievals vote for
    the same older frame neighbourhood.
    """
    if not enabled or not ranked_candidates:
        return ranked_candidates
    if candidates_above_threshold(ranked_candidates, semantic_threshold, 1):
        return ranked_candidates

    candidate, score, breakdown = ranked_candidates[0]
    raw_score = float(breakdown.get("raw_unified_score", score))
    if (
        breakdown.get("evidence_label") != "weak_scene_only"
        or raw_score < float(raw_threshold)
        or candidate.source_frame_id is None
    ):
        return ranked_candidates

    centre = int(candidate.source_frame_id)
    support = 0
    for item, _, item_breakdown in ranked_candidates[:max(1, int(top_k))]:
        if (
            item_breakdown.get("evidence_label") == "weak_scene_only"
            and item.source_frame_id is not None
            and abs(int(item.source_frame_id) - centre)
            <= max(0, int(cluster_width_frames))
        ):
            support += 1
    if support < max(1, int(minimum_support)):
        return ranked_candidates

    promoted_breakdown = dict(breakdown)
    promoted_breakdown["effective_score"] = float(effective_score)
    promoted_breakdown["evidence_label"] = "weak_scene_consensus"
    promoted_breakdown["scene_consensus_support"] = support
    return [
        (candidate, float(effective_score), promoted_breakdown),
        *ranked_candidates[1:],
    ]


def apply_scene_only_score_policy(
    breakdown,
    raw_score,
    semantic_threshold,
    scene_only_effective_score,
):
    """Demote a qualifying scene-only match without blocking geometry.

    The raw model score remains available in ``breakdown`` for diagnostics and
    tie-breaking. Only candidates that already clear the semantic threshold
    are assigned the common low-confidence effective score.
    """
    scene_only = (
        breakdown.get("scene_score") is not None
        and breakdown.get("object_score") is None
        and breakdown.get("text_score") is None
    )
    if scene_only:
        if float(raw_score) > float(semantic_threshold):
            return float(scene_only_effective_score), "weak_scene_only"
        return float(raw_score), "weak_scene_only"
    return float(raw_score), "strong_multi_layer"


class HuMemSLAMNode(Node):
    """
    ROS 2 wrapper for the HuMemSLAM semantic recovery model.

    This node:
        1. receives keyframe images, currently normal Image messages
        2. runs scene / YOLO segmentation / OCR processing
        3. builds KeyframeRecord objects
        4. calls CognitiveMathModel
        5. selects a semantic candidate keyframe

    Later, this can subscribe to a custom ORB-SLAM3 keyframe message that includes:
        - keyframe id
        - tracking inliers
        - pose
        - BoW / keyframe metadata
    """

    def __init__(self):
        super().__init__("human_slam_node")

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.keyframe_counter = 0

        self.scene_history = deque(maxlen=3)

        self.map_memory = []
        self._landmark_index = {}
        self._next_landmark_track_id = 1
        self._candidate_provenance = {}
        self._ocr_landmark_cache = []
        self._next_ocr_track_id = 1


        self._semantic_neighborhood_cache = {}
        self._last_perception_timings = {
            "yolo_ms": 0.0,
            "object_postprocess_ms": 0.0,
            "object_embedding_ms": 0.0,
            "dino_inference_ms": 0.0,
            "dino_pooling_ms": 0.0,
            "dino_tracking_ms": 0.0,
            "dino_search_ms": 0.0,
            "dino_consensus_ms": 0.0,
            "ocr_ms": 0.0,
            "ocr_crop_count": 0,
            "ocr_cache_hit_count": 0,
            "ocr_cache_refresh_count": 0,
        }

        self.matcher = CognitiveMathModel(
            w_scene=self.w_scene,
            w_object=self.w_object,
            w_text=self.w_text,
            sigma_mask=self.sigma_mask,
            lambda_area=self.lambda_area,
            text_geom_threshold=self.text_geom_threshold,
            semantic_threshold=self.semantic_threshold,
            use_scene=self.use_scene,
            use_object=self.use_object,
            use_text=self.use_text,
            text_conflict_floor=self.text_conflict_floor,
            use_scene_category=self.use_scene_category,
            scene_category_weight=self.scene_category_weight,
            object_class_weights=self.object_class_weights,
            relative_layout_weight=self.relative_layout_weight,
            relative_layout_sigma=self.relative_layout_sigma,
            persistence_gain=self.persistence_gain,
            object_appearance_weight=self.object_appearance_weight,
            fusion_mode=self.fusion_mode,
            object_support_gain=self.object_support_gain,
            text_support_gain=self.text_support_gain,
        )

        self.get_logger().info("HuMemSLAM node initialising")
        self.get_logger().info(f"Camera topic: {self.camera_source}")

        self._load_models()
        self._start_semantic_worker()

        self.semantic_candidates_publisher = self.create_publisher(
            SemanticCandidates,
            self.semantic_candidates_topic,
            10,
        )

        self.image_subscription = None
        self.orb_frame_subscription = None
        if self.use_orb_slam_messages:
            self.orb_frame_subscription = self.create_subscription(
                OrbSlamFrame,
                self.orb_frame_topic,
                self.orb_frame_callback,
                qos_profile_sensor_data,
            )
        else:
            self.image_subscription = self.create_subscription(
                Image,
                self.camera_source,
                self.keyframe_callback,
                1,
            )

        backends = []
        if self.use_scene:
            backends.append("TensorRT scene")
        if self.use_object or self.use_text:
            backends.append("TensorRT YOLO")
        if self.use_text and self.ocr_enabled:
            backends.append(f"{self.ocr_device.upper()} OCR")
        self.get_logger().info(
            f"HuMemSLAM active — {', '.join(backends)} ready"
        )


    def _declare_parameters(self):
        self.declare_parameter("camera_source", "/camera/image_raw")
        self.declare_parameter("use_orb_slam_messages", True)
        self.declare_parameter(
            "orb_frame_topic",
            "/orbslam3/semantic_frame",
        )
        self.declare_parameter(
            "semantic_candidates_topic",
            "/human_slam/semantic_candidates",
        )

        self.declare_parameter("scene_classifier_path", "/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam/weights/resnet50_places365.engine")
        self.declare_parameter("scene_classifier_threshold", 0.5)
        self.declare_parameter("scene_embedding_dim", 512)
        self.declare_parameter("scene_input_height", 224)
        self.declare_parameter("scene_input_width", 224)
        self.declare_parameter("scene_embedding_output", "")
        self.declare_parameter("scene_logits_output", "")
        self.declare_parameter("scene_labels_path", "")



        self.declare_parameter("global_descriptor_enabled", False)
        self.declare_parameter("global_descriptor_name", "places365")
        self.declare_parameter("global_descriptor_engine_path", "")
        self.declare_parameter("global_descriptor_input_height", 320)
        self.declare_parameter("global_descriptor_input_width", 320)
        self.declare_parameter("global_descriptor_input_name", "")
        self.declare_parameter("global_descriptor_output_name", "descriptor")
        self.declare_parameter("use_scene_category", True)
        self.declare_parameter("scene_category_weight", 0.10)
        self.declare_parameter("scene_category_top_k", 3)

        self.declare_parameter("yolo_model_path", "/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam/weights/humanSLAM_YOLO_seg.engine")
        self.declare_parameter("yolo_confidence_threshold", 0.5)
        self.declare_parameter("yolo_box_geometry_fallback", True)

        self.declare_parameter("ocr_confidence_threshold", 0.5)
        self.declare_parameter("ocr_language", "en")
        self.declare_parameter("ocr_use_gpu", True)
        self.declare_parameter("ocr_use_angle_cls", True)
        self.declare_parameter("ocr_show_log", False)
        self.declare_parameter("ocr_enabled", True)
        self.declare_parameter("ocr_backend", "paddle")
        self.declare_parameter("ocr_detector_engine_path", "")
        self.declare_parameter("ocr_recognizer_engine_path", "")
        self.declare_parameter("ocr_recognizer_yaml_path", "")
        self.declare_parameter("ocr_warmup_on_start", True)
        self.declare_parameter("ocr_keyframe_interval", 5)
        self.declare_parameter("ocr_max_objects_per_keyframe", 3)
        self.declare_parameter(
            "ocr_always_classes",
            [
                "advertisement_sign",
                "store_sign",
                "information_sign",
                "traffic_sign",
            ],
        )
        self.declare_parameter(
            "ocr_classes",
            [
                "building",
                "advertisement_sign",
                "store_sign",
                "information_sign",
                "traffic_sign",
            ],
        )


        self.declare_parameter("ocr_enable_mkldnn", False)
        self.declare_parameter("ocr_cpu_threads", 4)
        self.declare_parameter("ocr_text_det_limit_side_len", 640)
        self.declare_parameter("building_ocr_tile_size", 640)
        self.declare_parameter("building_ocr_tile_overlap", 0.20)
        self.declare_parameter("building_ocr_max_tiles", 6)
        self.declare_parameter("ocr_mask_min_foreground_ratio", 0.05)
        self.declare_parameter("ocr_min_crop_side", 12)
        self.declare_parameter("ocr_min_crop_pixels", 256)
        self.declare_parameter("ocr_min_sharpness", 8.0)
        self.declare_parameter("ocr_batch_enabled", True)
        self.declare_parameter("ocr_enable_hpi", False)
        self.declare_parameter("ocr_use_tensorrt", False)
        self.declare_parameter("ocr_tensorrt_precision", "fp16")
        self.declare_parameter("ocr_cache_enabled", True)
        self.declare_parameter("ocr_cache_max_frame_gap", 20)
        self.declare_parameter("ocr_cache_centroid_threshold", 0.20)
        self.declare_parameter("ocr_cache_area_ratio", 3.0)
        self.declare_parameter("ocr_cache_refresh_interval", 20)
        self.declare_parameter("ocr_cache_retry_interval", 5)
        self.declare_parameter("ocr_cache_quality_gain", 1.35)
        self.declare_parameter("ocr_cache_max_entries", 256)

        self.declare_parameter(
            "stable_classes",
            [
                "building",
                "bridge",
                "advertisement_sign",
                "store_sign",
                "information_sign",
                "traffic_sign",
                "traffic_light",
                "wall",
                "fence",
                "guard_rail",
                "tunnel",
                "street_light",
                "pole",
            ],
        )
        self.declare_parameter("crop_margin", 10)
        self.declare_parameter("isolate_mask_for_ocr", True)
        self.declare_parameter("async_processing", True)
        self.declare_parameter("semantic_queue_size", 1)
        self.declare_parameter("candidate_top_k", 25)


        self.declare_parameter("candidate_rerank_top_k", 5)
        self.declare_parameter("candidate_min_keyframe_separation", 20)
        self.declare_parameter("candidate_response_count", 5)
        self.declare_parameter("periodic_loop_keyframe_interval", 5)
        self.declare_parameter("semantic_ambiguity_margin", 0.05)
        self.declare_parameter("weak_scene_consensus_enabled", True)
        self.declare_parameter("weak_scene_consensus_threshold", 0.55)
        self.declare_parameter("weak_scene_consensus_top_k", 5)
        self.declare_parameter("weak_scene_consensus_min_support", 3)
        self.declare_parameter("weak_scene_consensus_cluster_width_frames", 30)
        self.declare_parameter("verbose_detections", False)

        self.declare_parameter("w_scene", 0.3)
        self.declare_parameter("w_object", 0.3)
        self.declare_parameter("w_text", 0.4)
        self.declare_parameter("fusion_mode", "scene_support")
        self.declare_parameter("object_support_gain", 0.15)
        self.declare_parameter("text_support_gain", 0.25)
        self.declare_parameter("use_scene", True)
        self.declare_parameter("use_object", True)
        self.declare_parameter("use_text", True)
        self.declare_parameter("sigma_mask", 0.25)
        self.declare_parameter("lambda_area", 1.0)
        self.declare_parameter("text_geom_threshold", 0.6)
        self.declare_parameter("text_conflict_floor", 0.25)
        self.declare_parameter("semantic_neighborhood_enabled", True)
        self.declare_parameter(
            "semantic_neighborhood_weights", [1.0, 0.6, 0.3]
        )
        self.declare_parameter("semantic_neighborhood_merge_distance", 0.15)
        self.declare_parameter("relative_layout_weight", 0.15)
        self.declare_parameter("relative_layout_sigma", 0.25)
        self.declare_parameter("semantic_persistence_gain", 0.25)
        self.declare_parameter("object_embedding_enabled", False)
        self.declare_parameter("object_embedding_max_objects", 8)
        self.declare_parameter("object_embedding_candidate_top_k", 5)
        self.declare_parameter("object_appearance_weight", 0.35)
        self.declare_parameter("dino_object_enabled", False)
        self.declare_parameter("dino_engine_config", "")
        self.declare_parameter("dino_pooling", "mask_central")
        self.declare_parameter("dino_keyframe_interval", 2)
        self.declare_parameter("dino_skip_on_ocr_keyframe", True)
        self.declare_parameter("dino_history_length", 3)
        self.declare_parameter("dino_temporal_decay", 0.70)
        self.declare_parameter("dino_track_similarity_threshold", 0.75)
        self.declare_parameter("dino_track_centroid_threshold", 0.35)
        self.declare_parameter("dino_similarity_threshold", 0.80)
        self.declare_parameter("dino_similarity_margin", 0.05)
        self.declare_parameter("dino_min_consensus_tracks", 2)
        self.declare_parameter("dino_candidate_top_k", 2)
        self.declare_parameter("dino_candidate_cluster_width", 30)
        self.declare_parameter("dino_search_neighbors_per_track", 5)
        self.declare_parameter("dino_single_landmark_classes", [
            "advertisement_sign", "store_sign", "information_sign",
            "traffic_sign",
        ])
        self.declare_parameter("dino_candidate_classes", [
            "building", "bridge", "advertisement_sign", "store_sign",
            "information_sign", "traffic_sign", "tunnel",
        ])
        self.declare_parameter("object_class_weights", [
            "pole:0.25", "fence:0.35", "wall:0.40",
            "street_light:0.45", "guard_rail:0.45", "building:0.70",
            "bridge:0.80", "tunnel:0.85", "traffic_light:0.90",
            "traffic_sign:1.00", "information_sign:1.00",
            "store_sign:1.00", "advertisement_sign:1.00",
        ])

        self.declare_parameter("semantic_threshold", 0.70)
        self.declare_parameter("object_only_threshold", 0.625)
        self.declare_parameter("object_text_threshold", 0.600)
        self.declare_parameter("scene_only_effective_score", 0.701)
        self.declare_parameter("geom_inlier_threshold", 30)
        self.declare_parameter("latency_output", "")
        self.declare_parameter("candidate_output", "")
        self.declare_parameter("source_image_dir", "")
        self.declare_parameter("debug_output_dir", "")

    def _read_parameters(self):
        self.camera_source = self.get_parameter("camera_source").value
        self.use_orb_slam_messages = bool(
            self.get_parameter("use_orb_slam_messages").value
        )
        self.orb_frame_topic = self.get_parameter("orb_frame_topic").value
        self.semantic_candidates_topic = self.get_parameter(
            "semantic_candidates_topic"
        ).value

        self.scene_classifier_path = self.get_parameter("scene_classifier_path").value
        self.scene_classifier_threshold = self.get_parameter(
            "scene_classifier_threshold"
        ).value
        self.scene_embedding_dim = int(self.get_parameter("scene_embedding_dim").value)
        self.scene_input_height = int(
            self.get_parameter("scene_input_height").value
        )
        self.scene_input_width = int(self.get_parameter("scene_input_width").value)
        self.scene_embedding_output = self.get_parameter(
            "scene_embedding_output"
        ).value
        self.scene_logits_output = self.get_parameter("scene_logits_output").value
        self.scene_labels_path = self.get_parameter("scene_labels_path").value
        self.global_descriptor_enabled = bool(
            self.get_parameter("global_descriptor_enabled").value
        )
        self.global_descriptor_name = str(
            self.get_parameter("global_descriptor_name").value
        )
        self.global_descriptor_engine_path = str(
            self.get_parameter("global_descriptor_engine_path").value
        )
        self.global_descriptor_input_height = int(
            self.get_parameter("global_descriptor_input_height").value
        )
        self.global_descriptor_input_width = int(
            self.get_parameter("global_descriptor_input_width").value
        )
        self.global_descriptor_input_name = str(
            self.get_parameter("global_descriptor_input_name").value
        )
        self.global_descriptor_output_name = str(
            self.get_parameter("global_descriptor_output_name").value
        )
        self.use_scene_category = bool(
            self.get_parameter("use_scene_category").value
        )
        self.scene_category_weight = max(
            0.0, min(1.0, float(
                self.get_parameter("scene_category_weight").value
            ))
        )
        self.scene_category_top_k = max(
            1, int(self.get_parameter("scene_category_top_k").value)
        )

        self.yolo_model_path = self.get_parameter("yolo_model_path").value
        self.yolo_conf = float(self.get_parameter("yolo_confidence_threshold").value)
        self.yolo_box_geometry_fallback = bool(
            self.get_parameter("yolo_box_geometry_fallback").value
        )

        self.ocr_conf = float(self.get_parameter("ocr_confidence_threshold").value)
        self.ocr_language = self.get_parameter("ocr_language").value
        self.ocr_use_gpu = bool(self.get_parameter("ocr_use_gpu").value)
        self.ocr_use_angle_cls = bool(self.get_parameter("ocr_use_angle_cls").value)
        self.ocr_show_log = bool(self.get_parameter("ocr_show_log").value)
        self.ocr_enabled = bool(self.get_parameter("ocr_enabled").value)
        self.ocr_backend = str(self.get_parameter("ocr_backend").value).lower()
        if self.ocr_backend not in {"paddle", "tensorrt"}:
            raise ValueError("ocr_backend must be paddle or tensorrt")
        self.ocr_detector_engine_path = str(
            self.get_parameter("ocr_detector_engine_path").value
        )
        self.ocr_recognizer_engine_path = str(
            self.get_parameter("ocr_recognizer_engine_path").value
        )
        self.ocr_recognizer_yaml_path = str(
            self.get_parameter("ocr_recognizer_yaml_path").value
        )
        self.ocr_warmup_on_start = bool(
            self.get_parameter("ocr_warmup_on_start").value
        )
        self.ocr_keyframe_interval = max(
            1, int(self.get_parameter("ocr_keyframe_interval").value)
        )
        self.ocr_max_objects = max(
            0, int(self.get_parameter("ocr_max_objects_per_keyframe").value)
        )
        self.ocr_classes = set(self.get_parameter("ocr_classes").value)
        self.ocr_always_classes = set(
            self.get_parameter("ocr_always_classes").value
        )
        self.ocr_enable_mkldnn = bool(
            self.get_parameter("ocr_enable_mkldnn").value
        )
        self.ocr_cpu_threads = max(
            1, int(self.get_parameter("ocr_cpu_threads").value)
        )
        self.ocr_text_det_limit_side_len = max(
            64, int(self.get_parameter("ocr_text_det_limit_side_len").value)
        )
        self.building_ocr_tile_size = max(
            128, int(self.get_parameter("building_ocr_tile_size").value)
        )
        self.building_ocr_tile_overlap = float(np.clip(
            self.get_parameter("building_ocr_tile_overlap").value, 0.0, 0.75
        ))
        self.building_ocr_max_tiles = max(
            1, int(self.get_parameter("building_ocr_max_tiles").value)
        )
        self.ocr_mask_min_foreground_ratio = float(np.clip(
            self.get_parameter("ocr_mask_min_foreground_ratio").value,
            0.0, 1.0,
        ))
        self.ocr_min_crop_side = max(
            1, int(self.get_parameter("ocr_min_crop_side").value)
        )
        self.ocr_min_crop_pixels = max(
            1, int(self.get_parameter("ocr_min_crop_pixels").value)
        )
        self.ocr_min_sharpness = max(
            0.0, float(self.get_parameter("ocr_min_sharpness").value)
        )
        self.ocr_batch_enabled = bool(
            self.get_parameter("ocr_batch_enabled").value
        )
        self.ocr_enable_hpi = bool(
            self.get_parameter("ocr_enable_hpi").value
        )
        self.ocr_use_tensorrt = bool(
            self.get_parameter("ocr_use_tensorrt").value
        )
        self.ocr_tensorrt_precision = str(
            self.get_parameter("ocr_tensorrt_precision").value
        ).lower()
        if self.ocr_tensorrt_precision not in {"fp16", "fp32"}:
            raise ValueError("ocr_tensorrt_precision must be fp16 or fp32")
        self.ocr_cache_enabled = bool(
            self.get_parameter("ocr_cache_enabled").value
        )
        self.ocr_cache_max_frame_gap = max(
            1, int(self.get_parameter("ocr_cache_max_frame_gap").value)
        )
        self.ocr_cache_centroid_threshold = max(
            0.0, float(self.get_parameter("ocr_cache_centroid_threshold").value)
        )
        self.ocr_cache_area_ratio = max(
            1.0, float(self.get_parameter("ocr_cache_area_ratio").value)
        )
        self.ocr_cache_refresh_interval = max(
            1, int(self.get_parameter("ocr_cache_refresh_interval").value)
        )
        self.ocr_cache_retry_interval = max(
            1, int(self.get_parameter("ocr_cache_retry_interval").value)
        )
        self.ocr_cache_quality_gain = max(
            1.0, float(self.get_parameter("ocr_cache_quality_gain").value)
        )
        self.ocr_cache_max_entries = max(
            1, int(self.get_parameter("ocr_cache_max_entries").value)
        )

        self.stable_classes = list(self.get_parameter("stable_classes").value)
        self.get_logger().info(f"Loaded stable_classes: {self.stable_classes}")
        self.crop_margin = int(self.get_parameter("crop_margin").value)
        self.isolate_mask_for_ocr = bool(self.get_parameter("isolate_mask_for_ocr").value)
        self.async_processing = bool(self.get_parameter("async_processing").value)
        self.semantic_queue_size = max(
            1, int(self.get_parameter("semantic_queue_size").value)
        )
        self.candidate_top_k = max(
            1, int(self.get_parameter("candidate_top_k").value)
        )
        self.candidate_rerank_top_k = max(
            self.candidate_response_count
            if hasattr(self, "candidate_response_count") else 1,
            int(self.get_parameter("candidate_rerank_top_k").value),
        )
        self.candidate_min_separation = max(
            0,
            int(self.get_parameter("candidate_min_keyframe_separation").value),
        )
        self.candidate_response_count = max(
            1, int(self.get_parameter("candidate_response_count").value)
        )
        self.candidate_rerank_top_k = max(
            self.candidate_response_count, self.candidate_rerank_top_k
        )
        self.periodic_loop_keyframe_interval = max(
            1,
            int(self.get_parameter("periodic_loop_keyframe_interval").value),
        )
        self.semantic_ambiguity_margin = max(
            0.0, float(self.get_parameter("semantic_ambiguity_margin").value)
        )
        self.weak_scene_consensus_enabled = bool(
            self.get_parameter("weak_scene_consensus_enabled").value
        )
        self.weak_scene_consensus_threshold = float(
            self.get_parameter("weak_scene_consensus_threshold").value
        )
        self.weak_scene_consensus_top_k = max(
            1, int(self.get_parameter("weak_scene_consensus_top_k").value)
        )
        self.weak_scene_consensus_min_support = max(
            1, int(self.get_parameter("weak_scene_consensus_min_support").value)
        )
        self.weak_scene_consensus_cluster_width_frames = max(
            0,
            int(self.get_parameter(
                "weak_scene_consensus_cluster_width_frames"
            ).value),
        )
        self.verbose_detections = bool(
            self.get_parameter("verbose_detections").value
        )

        self.w_scene = float(self.get_parameter("w_scene").value)
        self.w_object = float(self.get_parameter("w_object").value)
        self.w_text = float(self.get_parameter("w_text").value)
        self.fusion_mode = str(self.get_parameter("fusion_mode").value)
        self.object_support_gain = float(
            self.get_parameter("object_support_gain").value
        )
        self.text_support_gain = float(
            self.get_parameter("text_support_gain").value
        )
        self.use_scene = bool(self.get_parameter("use_scene").value)
        self.use_object = bool(self.get_parameter("use_object").value)
        self.use_text = bool(self.get_parameter("use_text").value)
        if not any((self.use_scene, self.use_object, self.use_text)):
            raise ValueError("At least one HuMemSLAM layer must be enabled")
        self.sigma_mask = float(self.get_parameter("sigma_mask").value)
        self.lambda_area = float(self.get_parameter("lambda_area").value)
        self.text_geom_threshold = float(self.get_parameter("text_geom_threshold").value)
        self.text_conflict_floor = float(
            self.get_parameter("text_conflict_floor").value
        )
        self.semantic_neighborhood_enabled = bool(
            self.get_parameter("semantic_neighborhood_enabled").value
        )
        self.semantic_neighborhood_weights = [
            float(np.clip(value, 0.0, 1.0))
            for value in self.get_parameter(
                "semantic_neighborhood_weights"
            ).value
        ]
        if not self.semantic_neighborhood_weights:
            self.semantic_neighborhood_weights = [1.0]
        self.semantic_neighborhood_weights[0] = 1.0
        self.semantic_neighborhood_merge_distance = max(
            0.0, float(self.get_parameter(
                "semantic_neighborhood_merge_distance"
            ).value),
        )
        self.relative_layout_weight = float(np.clip(
            self.get_parameter("relative_layout_weight").value, 0.0, 1.0
        ))
        self.relative_layout_sigma = max(
            1e-6, float(self.get_parameter("relative_layout_sigma").value)
        )
        self.persistence_gain = max(
            0.0, float(self.get_parameter("semantic_persistence_gain").value)
        )
        self.object_embedding_enabled = bool(
            self.get_parameter("object_embedding_enabled").value
        )
        self.object_embedding_max_objects = max(
            0, int(self.get_parameter("object_embedding_max_objects").value)
        )
        self.object_embedding_candidate_top_k = max(
            1, int(self.get_parameter("object_embedding_candidate_top_k").value)
        )
        self.object_appearance_weight = float(np.clip(
            self.get_parameter("object_appearance_weight").value, 0.0, 1.0
        ))
        self.dino_object_enabled = bool(
            self.get_parameter("dino_object_enabled").value
        )
        self.dino_engine_config = str(
            self.get_parameter("dino_engine_config").value
        )
        self.dino_pooling = str(self.get_parameter("dino_pooling").value)
        self.dino_keyframe_interval = max(
            1, int(self.get_parameter("dino_keyframe_interval").value)
        )
        self.dino_skip_on_ocr_keyframe = bool(
            self.get_parameter("dino_skip_on_ocr_keyframe").value
        )
        self.dino_history_length = max(
            1, int(self.get_parameter("dino_history_length").value)
        )
        self.dino_temporal_decay = float(np.clip(
            self.get_parameter("dino_temporal_decay").value, 0.0, 1.0
        ))
        self.dino_track_similarity_threshold = float(np.clip(
            self.get_parameter("dino_track_similarity_threshold").value,
            -1.0, 1.0,
        ))
        self.dino_track_centroid_threshold = max(
            0.0, float(self.get_parameter("dino_track_centroid_threshold").value)
        )
        self.dino_similarity_threshold = float(np.clip(
            self.get_parameter("dino_similarity_threshold").value, -1.0, 1.0
        ))
        self.dino_similarity_margin = max(
            0.0, float(self.get_parameter("dino_similarity_margin").value)
        )
        self.dino_min_consensus_tracks = max(
            1, int(self.get_parameter("dino_min_consensus_tracks").value)
        )
        self.dino_candidate_top_k = max(
            0, int(self.get_parameter("dino_candidate_top_k").value)
        )
        self.dino_candidate_cluster_width = max(
            0, int(self.get_parameter("dino_candidate_cluster_width").value)
        )
        self.dino_search_neighbors_per_track = max(
            2, int(self.get_parameter("dino_search_neighbors_per_track").value)
        )
        self.dino_single_landmark_classes = set(
            self.get_parameter("dino_single_landmark_classes").value
        )
        self.dino_candidate_classes = set(
            self.get_parameter("dino_candidate_classes").value
        )
        self.object_class_weights = {}
        for specification in self.get_parameter("object_class_weights").value:
            name, separator, value = str(specification).partition(":")
            if separator and name.strip():
                self.object_class_weights[name.strip()] = float(
                    np.clip(float(value), 0.0, 1.0)
                )

        self.semantic_threshold = float(self.get_parameter("semantic_threshold").value)
        self.object_only_threshold = float(
            self.get_parameter("object_only_threshold").value
        )
        self.object_text_threshold = float(
            self.get_parameter("object_text_threshold").value
        )
        self.scene_only_effective_score = float(
            self.get_parameter("scene_only_effective_score").value
        )
        if self.scene_only_effective_score <= self.semantic_threshold:
            self.get_logger().warn(
                "scene_only_effective_score must be above semantic_threshold "
                "for qualifying scene-only candidates to reach geometry "
                f"({self.scene_only_effective_score:.3f} <= "
                f"{self.semantic_threshold:.3f})"
            )
        self.geom_inlier_threshold = int(self.get_parameter("geom_inlier_threshold").value)
        self.latency_output = str(self.get_parameter("latency_output").value)
        self.candidate_output = str(self.get_parameter("candidate_output").value)
        self.source_image_dir = str(self.get_parameter("source_image_dir").value)
        self.debug_output_dir = str(self.get_parameter("debug_output_dir").value)
        self.get_logger().info(
            "Periodic semantic loop detection: every "
            f"{self.periodic_loop_keyframe_interval} keyframe(s)"
        )


    def _load_models(self):
        self.dino_descriptor = None
        self.global_descriptor = None
        self.scene_runtime = None
        self.scene_engine = None
        self.scene_context = None
        self.scene_tensor_names = []
        self.scene_labels = self._load_scene_labels(self.scene_labels_path)
        self.scene_device_tensors = {}
        self.scene_stream = None

        if self.dino_object_enabled:
            if not self.dino_engine_config:
                raise RuntimeError(
                    "dino_object_enabled is true but dino_engine_config is empty"
                )
            self.dino_descriptor = TensorRTDenseDescriptor(
                DenseDescriptorSpec.from_json(self.dino_engine_config)
            )
            self.get_logger().info(
                f"Dense object descriptor loaded: {self.dino_engine_config}"
            )

        if self.use_scene and self.scene_classifier_path:
            try:
                (
                    self.scene_runtime,
                    self.scene_engine,
                    self.scene_context,
                ) = self.load_scene_engine_once(self.scene_classifier_path)
                self.scene_tensor_names = [
                    self.scene_engine.get_tensor_name(i)
                    for i in range(self.scene_engine.num_io_tensors)
                ]
                self._log_scene_engine_tensors()
                self.get_logger().info("Places365 TensorRT engine loaded")
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to load Places365 TensorRT engine: {exc}"
                )

                self.scene_engine = None
                self.scene_context = None

        if (self.use_scene or self.object_embedding_enabled) and self.global_descriptor_enabled:
            if not self.global_descriptor_engine_path:
                raise RuntimeError(
                    "global_descriptor_enabled is true but engine path is empty"
                )
            spec = DescriptorSpec(
                name=self.global_descriptor_name,
                engine_path=Path(self.global_descriptor_engine_path),
                input_height=self.global_descriptor_input_height,
                input_width=self.global_descriptor_input_width,
                input_name=self.global_descriptor_input_name,
                output_name=self.global_descriptor_output_name,
            )
            self.global_descriptor = TensorRTGlobalDescriptor(spec)
            self.get_logger().info(
                f"Global descriptor loaded: {spec.name} ({spec.engine_path})"
            )

        self.yolo = None
        if self.use_object or self.use_text:
            if not self.yolo_model_path:
                raise RuntimeError("Parameter yolo_model_path is empty")




            self.yolo = YOLO(self.yolo_model_path, task="segment")
            self.get_logger().info(f"YOLO model loaded: {self.yolo_model_path}")
            self.get_logger().debug(f"YOLO class names: {self.yolo.names}")



        self.ocr_model = None
        self.ocr_device = "cpu"
        self.ocr_model_lock = threading.Lock()
        self._ocr_initialization_failed = False

        if self.use_text and self.ocr_enabled and self.ocr_warmup_on_start:
            self._warmup_ocr()

    def load_scene_engine_once(self, engine_path):
        """
        Load TensorRT engine once.
        """
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as file:
            engine_bytes = file.read()

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)

        if engine is None:
            raise RuntimeError("Failed to deserialize scene classifier engine")

        context = engine.create_execution_context()

        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")


        return runtime, engine, context

    def _load_scene_labels(self, labels_path):
        if not labels_path:
            return []

        try:
            with open(labels_path, "r", encoding="utf-8") as labels_file:
                labels = []
                for line in labels_file:
                    label = line.strip()
                    if not label:
                        continue

                    label = label.rsplit(" ", 1)[0]
                    label = label.replace("/a/", "").replace("_", " ")
                    labels.append(label)
                return labels
        except OSError as exc:
            self.get_logger().warn(f"Could not load scene labels: {exc}")
            return []

    def _log_scene_engine_tensors(self):
        for name in self.scene_tensor_names:
            mode = self.scene_engine.get_tensor_mode(name)
            shape = tuple(self.scene_engine.get_tensor_shape(name))
            dtype = self.scene_engine.get_tensor_dtype(name)
            self.get_logger().info(
                f"Scene engine tensor: name={name}, mode={mode}, "
                f"shape={shape}, dtype={dtype}"
            )


    def _start_semantic_worker(self):
        self._shutdown_event = threading.Event()
        self._semantic_queue = Queue(maxsize=self.semantic_queue_size)
        self._semantic_worker = None
        self._latency_lock = threading.Lock()
        self._latency_header_written = False

        if self.async_processing:
            self._semantic_worker = threading.Thread(
                target=self._semantic_worker_loop,
                name="human_slam_semantic_worker",
                daemon=True,
            )
            self._semantic_worker.start()

    def keyframe_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        self.keyframe_counter += 1
        keyframe_id = self.keyframe_counter
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self._submit_semantic_task(
            {
                "cv_image": cv_image,
                "keyframe_id": keyframe_id,
                "timestamp": timestamp,
                "query_frame_id": keyframe_id,
                "orb_keyframe_id": None,
                "orb_map_id": None,
                "pose": None,
                "tracking_inliers": None,
                "store_in_map": True,
                "force_ocr": False,
                "response_header": msg.header,
                "received_monotonic": time.perf_counter(),
            }
        )

    def orb_frame_callback(self, msg: OrbSlamFrame):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg.image,
                desired_encoding="bgr8",
            )
        except CvBridgeError as exc:
            self.get_logger().error(f"ORB frame cv_bridge error: {exc}")
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        keyframe_id = (
            int(msg.reference_keyframe_id)
            if msg.is_keyframe
            else int(msg.frame_id)
        )
        pose = self._pose_message_to_matrix(msg.camera_pose) if msg.pose_valid else None
        run_candidate_retrieval = self._should_run_candidate_retrieval(
            is_keyframe=bool(msg.is_keyframe),
            keyframe_id=keyframe_id,
            tracking_state=int(msg.tracking_state),
        )

        self._submit_semantic_task(
            {
                "cv_image": cv_image,
                "keyframe_id": keyframe_id,
                "timestamp": timestamp,
                "query_frame_id": int(msg.frame_id),
                "orb_keyframe_id": (
                    int(msg.reference_keyframe_id)
                    if msg.has_reference_keyframe
                    else None
                ),
                "orb_map_id": int(msg.map_id),
                "pose": pose,
                "tracking_inliers": int(msg.tracking_inliers),
                "store_in_map": bool(msg.is_keyframe),
                "force_ocr": int(msg.tracking_state) >= 3,
                "run_candidate_retrieval": run_candidate_retrieval,
                "response_header": msg.header,
                "received_monotonic": time.perf_counter(),
            }
        )

    def _submit_semantic_task(self, task):
        if not self.async_processing:
            self._process_keyframe_image(**task)
            return

        try:
            self._semantic_queue.put_nowait(task)
        except Full:


            try:
                self._semantic_queue.get_nowait()
                self._semantic_queue.task_done()
            except Empty:
                pass
            self._semantic_queue.put_nowait(task)
            self.get_logger().debug("Replaced stale queued semantic keyframe")

    def _semantic_worker_loop(self):
        while not self._shutdown_event.is_set():
            try:
                task = self._semantic_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._process_keyframe_image(**task)
            except Exception as exc:
                self.get_logger().error(
                    f"Semantic processing failed for frame "
                    f"{task['query_frame_id']}: {exc}"
                )
            finally:
                self._semantic_queue.task_done()

    def _process_keyframe_image(
        self,
        cv_image,
        keyframe_id: int,
        timestamp: float,
        query_frame_id: int,
        orb_keyframe_id,
        orb_map_id,
        pose,
        tracking_inliers,
        store_in_map: bool,
        force_ocr: bool = False,
        run_candidate_retrieval: bool = True,
        response_header=None,
        received_monotonic=None,
    ):
        started = time.perf_counter()
        received_monotonic = received_monotonic or started
        debug_image = cv_image.copy() if self.debug_output_dir else None
        scene_record = (
            self.run_scene_classifier(cv_image)
            if self.use_scene
            else SceneRecord(embedding=None, confidence=0.0, label="disabled")
        )
        scene_finished = time.perf_counter()
        self.scene_history.appendleft(scene_record)

        run_ocr = self.use_text and self.ocr_enabled and (
            force_ocr or keyframe_id % self.ocr_keyframe_interval == 0
        )
        self._last_perception_timings = {
            "yolo_ms": 0.0,
            "object_postprocess_ms": 0.0,
            "object_embedding_ms": 0.0,
            "dino_inference_ms": 0.0,
            "dino_pooling_ms": 0.0,
            "dino_tracking_ms": 0.0,
            "dino_search_ms": 0.0,
            "dino_consensus_ms": 0.0,
            "ocr_ms": 0.0,
            "ocr_crop_count": 0,
            "ocr_cache_hit_count": 0,
            "ocr_cache_refresh_count": 0,
        }
        static_objects = (
            self.process_yolo_results(
                cv_image,
                keyframe_id=keyframe_id,
                run_ocr=run_ocr,
                run_dino=(
                    self.dino_object_enabled
                    and (force_ocr or keyframe_id % self.dino_keyframe_interval == 0)
                    and not (
                        self.dino_skip_on_ocr_keyframe
                        and run_ocr and not force_ocr
                    )
                ),
                debug_image=debug_image,
            )
            if self.use_object or self.use_text
            else []
        )
        perception_finished = time.perf_counter()

        current_keyframe = KeyframeRecord(
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            scene=scene_record,
            static_objects=static_objects,
            pose=pose,
            orb_keyframe_id=orb_keyframe_id,
            orb_map_id=orb_map_id,
            tracking_inliers=tracking_inliers,
            source_frame_id=int(query_frame_id),
        )

        tracking_started = time.perf_counter()
        self._assign_landmark_tracks(
            current_keyframe,
            self.map_memory[-self.dino_history_length:],
        )
        self._last_perception_timings["dino_tracking_ms"] = (
            time.perf_counter() - tracking_started
        ) * 1000.0

        ranking_query = self._semantic_neighborhood(
            current_keyframe,
            list(reversed(self.map_memory[-2:])),
        )

        query_scene_seq = list(self.scene_history)
        ranked_candidates = []
        if run_candidate_retrieval:
            candidate_keyframes, candidate_scene_sequences = (
                self.build_candidate_inputs(query_scene_seq, ranking_query)
            )
            candidate_keyframes, candidate_scene_sequences = (
                preselect_for_expensive_rerank(
                    candidate_keyframes,
                    candidate_scene_sequences,
                    self.candidate_rerank_top_k,
                    enabled=(
                        self.use_scene
                        and not self.object_embedding_enabled
                        and not self.dino_object_enabled
                    ),
                )
            )
            ranked_candidates = self._rank_candidates(
                ranking_query,
                query_scene_seq,
                candidate_keyframes,
                candidate_scene_sequences,
            )
            ranked_candidates = apply_weak_scene_consensus_policy(
                ranked_candidates,
                self.semantic_threshold,
                self.scene_only_effective_score,
                self.weak_scene_consensus_enabled,
                self.weak_scene_consensus_threshold,
                self.weak_scene_consensus_top_k,
                self.weak_scene_consensus_min_support,
                self.weak_scene_consensus_cluster_width_frames,
            )
        ranking_finished = time.perf_counter()
        self._write_candidate_rows(
            query_frame_id,
            orb_keyframe_id,
            ranking_query,
            query_scene_seq,
            ranked_candidates,
        )
        self._publish_semantic_candidates(
            query_frame_id,
            orb_keyframe_id,
            ranked_candidates,
            response_header,
        )
        published = time.perf_counter()
        self._write_latency_row(
            query_frame_id=query_frame_id,
            is_keyframe=store_in_map,
            queue_ms=(started - received_monotonic) * 1000.0,
            scene_ms=(scene_finished - started) * 1000.0,
            object_ocr_ms=(perception_finished - scene_finished) * 1000.0,
            yolo_ms=self._last_perception_timings.get("yolo_ms", 0.0),
            object_postprocess_ms=self._last_perception_timings.get(
                "object_postprocess_ms", 0.0
            ),
            object_embedding_ms=self._last_perception_timings.get(
                "object_embedding_ms", 0.0
            ),
            dino_inference_ms=self._last_perception_timings.get(
                "dino_inference_ms", 0.0
            ),
            dino_pooling_ms=self._last_perception_timings.get(
                "dino_pooling_ms", 0.0
            ),
            dino_tracking_ms=self._last_perception_timings.get(
                "dino_tracking_ms", 0.0
            ),
            dino_search_ms=self._last_perception_timings.get(
                "dino_search_ms", 0.0
            ),
            dino_consensus_ms=self._last_perception_timings.get(
                "dino_consensus_ms", 0.0
            ),
            ocr_ms=self._last_perception_timings.get("ocr_ms", 0.0),
            ocr_crop_count=self._last_perception_timings.get(
                "ocr_crop_count", 0
            ),
            ocr_cache_hit_count=self._last_perception_timings.get(
                "ocr_cache_hit_count", 0
            ),
            ocr_cache_refresh_count=self._last_perception_timings.get(
                "ocr_cache_refresh_count", 0
            ),
            ranking_ms=(ranking_finished - perception_finished) * 1000.0,
            total_ms=(published - received_monotonic) * 1000.0,
            candidate_count=len(ranked_candidates),
        )
        self._write_debug_frame(
            debug_image,
            query_frame_id,
            scene_record,
            static_objects,
            ranked_candidates,
        )

        if store_in_map:
            self.map_memory.append(current_keyframe)
            self._index_landmark_keyframe(
                current_keyframe, len(self.map_memory) - 1
            )

        self.get_logger().info(
            f"{'Stored keyframe' if store_in_map else 'Processed query frame'} "
            f"{keyframe_id}: "
            f"{len(static_objects)} static objects, "
            f"{sum(len(obj.texts) for obj in static_objects)} OCR texts"
        )

    def _should_run_candidate_retrieval(
        self, is_keyframe: bool, keyframe_id: int, tracking_state: int
    ) -> bool:
        """Run recovery immediately, but same-map search periodically while OK."""
        return should_run_candidate_retrieval(
            is_keyframe,
            keyframe_id,
            tracking_state,
            self.periodic_loop_keyframe_interval,
        )

    def _write_latency_row(
        self,
        query_frame_id,
        is_keyframe,
        queue_ms,
        scene_ms,
        object_ocr_ms,
        yolo_ms,
        object_postprocess_ms,
        object_embedding_ms,
        dino_inference_ms,
        dino_pooling_ms,
        dino_tracking_ms,
        dino_search_ms,
        dino_consensus_ms,
        ocr_ms,
        ocr_crop_count,
        ocr_cache_hit_count,
        ocr_cache_refresh_count,
        ranking_ms,
        total_ms,
        candidate_count,
    ):
        if not self.latency_output:
            return
        output = Path(self.latency_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        row = [
            time.time_ns(),
            int(query_frame_id),
            int(bool(is_keyframe)),
            f"{queue_ms:.3f}",
            f"{scene_ms:.3f}",
            f"{object_ocr_ms:.3f}",
            f"{yolo_ms:.3f}",
            f"{object_postprocess_ms:.3f}",
            f"{object_embedding_ms:.3f}",
            f"{dino_inference_ms:.3f}",
            f"{dino_pooling_ms:.3f}",
            f"{dino_tracking_ms:.3f}",
            f"{dino_search_ms:.3f}",
            f"{dino_consensus_ms:.3f}",
            f"{ocr_ms:.3f}",
            int(ocr_crop_count),
            int(ocr_cache_hit_count),
            int(ocr_cache_refresh_count),
            f"{ranking_ms:.3f}",
            f"{total_ms:.3f}",
            int(candidate_count),
        ]
        with self._latency_lock:
            write_header = not output.exists() or output.stat().st_size == 0
            with output.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if write_header:
                    writer.writerow(
                        [
                            "wall_time_ns",
                            "query_frame_id",
                            "is_keyframe",
                            "queue_ms",
                            "scene_ms",
                            "object_ocr_ms",
                            "yolo_ms",
                            "object_postprocess_ms",
                            "object_embedding_ms",
                            "dino_inference_ms",
                            "dino_pooling_ms",
                            "dino_tracking_ms",
                            "dino_search_ms",
                            "dino_consensus_ms",
                            "ocr_ms",
                            "ocr_crop_count",
                            "ocr_cache_hit_count",
                            "ocr_cache_refresh_count",
                            "ranking_ms",
                            "total_ms",
                            "candidate_count",
                        ]
                    )
                writer.writerow(row)

    def _frame_image_path(self, frame_id):
        if not self.source_image_dir or frame_id is None:
            return ""
        base = Path(self.source_image_dir).expanduser()
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = base / f"{int(frame_id):06d}{suffix}"
            if candidate.exists():
                return str(candidate.resolve())
        return str((base / f"{int(frame_id):06d}.png").resolve())

    def _write_debug_frame(
        self, debug_image, query_frame_id, scene_record, static_objects,
        ranked_candidates
    ):
        if debug_image is None or not self.debug_output_dir:
            return
        lines = [
            f"Frame {int(query_frame_id)}",
            f"Scene: {scene_record.label} ({scene_record.confidence:.3f})",
        ]
        if scene_record.category_distribution:
            categories = sorted(
                scene_record.category_distribution.items(),
                key=lambda item: item[1], reverse=True,
            )[:3]
            lines.append("Categories: " + ", ".join(
                f"{name}={score:.2f}" for name, score in categories
            ))
        if ranked_candidates:
            candidate, score, _ = ranked_candidates[0]
            lines.append(
                "Top candidate: "
                f"KF={candidate.orb_keyframe_id} "
                f"frame={candidate.source_frame_id} score={score:.3f}"
            )
        if static_objects:
            object_labels = [
                f"{item.class_name}={item.seg_conf:.2f}"
                for item in static_objects[:5]
            ]
            lines.append("Objects: " + ", ".join(object_labels))
            ocr_values = [
                text.text
                for item in static_objects
                for text in item.texts
                if text.text
            ]
            lines.append(
                "OCR: " + (" | ".join(ocr_values[:4]) if ocr_values else "none")
            )
        else:
            lines.extend(["Objects: none", "OCR: none"])


        overlay_h = 190
        annotated = np.zeros(
            (debug_image.shape[0] + overlay_h, debug_image.shape[1], 3),
            dtype=np.uint8,
        )
        annotated[:debug_image.shape[0], :] = debug_image
        text_y = debug_image.shape[0]
        for index, line in enumerate(lines):
            cv2.putText(
                annotated, line, (10, text_y + 25 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
        output = Path(self.debug_output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / f"{int(query_frame_id):06d}.png"), annotated)

    @staticmethod
    def _csv_score(value):
        return "" if value is None else f"{float(value):.6f}"

    def _write_candidate_rows(
        self,
        query_frame_id,
        query_reference_keyframe_id,
        query_keyframe,
        query_scene_seq,
        ranked_candidates,
    ):
        if not self.candidate_output or not ranked_candidates:
            return

        selected_count = len(candidates_above_threshold(
            ranked_candidates,
            self.semantic_threshold,
            self.candidate_response_count,
        ))
        best_score = ranked_candidates[0][1]
        second_score = ranked_candidates[1][1] if len(ranked_candidates) > 1 else None
        ambiguous = (
            second_score is not None
            and best_score - second_score < self.semantic_ambiguity_margin
        )
        accepted = bool(candidates_above_threshold(
            ranked_candidates[:1], self.semantic_threshold, 1
        ))
        output = Path(self.candidate_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "wall_time_ns", "query_frame_id", "query_image_path",
            "query_reference_keyframe_id", "rank", "published",
            "candidate_keyframe_id", "candidate_map_id",
            "candidate_source_frame_id", "candidate_image_path",
            "scene_score", "object_score", "text_score", "text_evidence",
            "raw_unified_score", "effective_score", "evidence_label",
            "candidate_source",
            "submission_threshold",
            "best_accepted", "best_ambiguous",
            "geometric_verification_result", "caused_recovery",
        ]
        rows = []
        for rank, (candidate, score, breakdown) in enumerate(
            ranked_candidates, start=1
        ):
            candidate_kf_id = (
                candidate.orb_keyframe_id
                if candidate.orb_keyframe_id is not None
                else candidate.keyframe_id
            )
            rows.append([
                time.time_ns(), int(query_frame_id),
                self._frame_image_path(query_frame_id),
                int(query_reference_keyframe_id or 0), rank,
                int(rank <= selected_count and score > float(
                    breakdown.get("submission_threshold", self.semantic_threshold)
                )),
                int(candidate_kf_id),
                int(candidate.orb_map_id or 0), candidate.source_frame_id,
                self._frame_image_path(candidate.source_frame_id),
                self._csv_score(breakdown["scene_score"]),
                self._csv_score(breakdown["object_score"]),
                self._csv_score(breakdown["text_score"]),
                self._csv_score(breakdown["text_evidence"]),
                self._csv_score(breakdown.get("raw_unified_score", score)),
                self._csv_score(score),
                str(breakdown.get("evidence_label", "strong_multi_layer")),
                str(breakdown.get("candidate_source", "scene")),
                self._csv_score(breakdown.get(
                    "submission_threshold", self.semantic_threshold
                )),
                int(rank == 1 and accepted),
                int(rank == 1 and ambiguous), "not_reported", "not_reported",
            ])
        with self._latency_lock:
            write_header = not output.exists() or output.stat().st_size == 0
            with output.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if write_header:
                    writer.writerow(header)
                writer.writerows(rows)

    def _pose_message_to_matrix(self, pose_message):
        x = float(pose_message.orientation.x)
        y = float(pose_message.orientation.y)
        z = float(pose_message.orientation.z)
        w = float(pose_message.orientation.w)

        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = [
            pose_message.position.x,
            pose_message.position.y,
            pose_message.position.z,
        ]
        return transform

    def destroy_node(self):
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        if (
            getattr(self, "_semantic_worker", None) is not None
            and self._semantic_worker.is_alive()
        ):
            self._semantic_worker.join(timeout=2.0)
        return super().destroy_node()


    def run_scene_classifier(self, cv_image) -> SceneRecord:
        """
        Run Places365 inference and return a descriptor, confidence, and label.

        If the engine exposes a configured embedding output, that tensor is used
        as the descriptor. Otherwise the class-probability vector is used. A
        classifier-only engine therefore works, although a penultimate-layer
        embedding is usually more discriminative for loop-candidate retrieval.
        """
        if (
            (self.scene_engine is None or self.scene_context is None)
            and self.global_descriptor is None
        ):
            return SceneRecord(
                embedding=None,
                confidence=0.0,
                label="unknown",
            )

        try:
            if self.scene_engine is not None and self.scene_context is not None:
                outputs = self._execute_scene_engine(cv_image)
                embedding, logits = self._select_scene_outputs(outputs)
            else:
                embedding, logits = None, None

            if logits is not None:
                probabilities = self._softmax(logits)
                class_id = int(np.argmax(probabilities))
                confidence = float(probabilities[class_id])
                label = (
                    self.scene_labels[class_id]
                    if class_id < len(self.scene_labels)
                    else f"places365_{class_id}"
                )
                category_distribution = group_places365_probabilities(
                    probabilities, self.scene_category_top_k
                ) if self.use_scene_category else {}
            elif embedding is not None:
                probabilities = None
                confidence = 1.0
                label = "embedding"
                category_distribution = {}
            else:
                probabilities = None
                confidence = 1.0
                label = self.global_descriptor_name
                category_distribution = {}

            descriptor = embedding if embedding is not None else probabilities
            if self.global_descriptor is not None:
                descriptor = self.global_descriptor.describe(cv_image)
            if descriptor is None or descriptor.size == 0:
                raise RuntimeError("Scene engine produced no usable output")

            return SceneRecord(
                embedding=descriptor.astype(np.float32, copy=False).reshape(-1),
                confidence=confidence,
                label=label,
                category_distribution=category_distribution,
            )
        except Exception as exc:
            self.get_logger().error(f"Scene inference failed: {exc}")
            return SceneRecord(



                embedding=None,
                confidence=0.0,
                label="inference_error",
            )

    def _execute_scene_engine(self, cv_image):
        """
        Execute a TensorRT 10+ engine using CUDA tensors owned by PyTorch.

        TensorRT receives raw CUDA addresses; no PyCUDA dependency is required.
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for CUDA buffer allocation") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")

        input_names = [
            name
            for name in self.scene_tensor_names
            if self.scene_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 1:
            raise RuntimeError(
                f"Expected one scene-engine input, found {len(input_names)}"
            )

        input_name = input_names[0]
        input_array = self._preprocess_scene_image(cv_image)
        input_shape = tuple(input_array.shape)

        engine_input_shape = tuple(self.scene_engine.get_tensor_shape(input_name))
        if any(dim < 0 for dim in engine_input_shape):
            if not self.scene_context.set_input_shape(input_name, input_shape):
                raise RuntimeError(
                    f"TensorRT rejected input shape {input_shape} for {input_name}"
                )
        elif engine_input_shape != input_shape:
            raise RuntimeError(
                f"Engine expects {engine_input_shape}, preprocessor produced {input_shape}"
            )

        if self.scene_stream is None:
            self.scene_stream = torch.cuda.Stream()
        stream = self.scene_stream

        input_dtype = self._torch_dtype(
            self.scene_engine.get_tensor_dtype(input_name), torch
        )
        input_tensor = self._get_scene_device_tensor(
            input_name, input_shape, input_dtype, torch
        )
        with torch.cuda.stream(stream):
            input_tensor.copy_(torch.from_numpy(input_array), non_blocking=True)
        self.scene_context.set_tensor_address(input_name, input_tensor.data_ptr())

        output_names = [
            name
            for name in self.scene_tensor_names
            if self.scene_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        for name in output_names:
            shape = tuple(self.scene_context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved output shape for {name}: {shape}")
            dtype = self._torch_dtype(self.scene_engine.get_tensor_dtype(name), torch)
            tensor = self._get_scene_device_tensor(name, shape, dtype, torch)
            self.scene_context.set_tensor_address(name, tensor.data_ptr())

        if not self.scene_context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

        stream.synchronize()
        return {
            name: self.scene_device_tensors[name]
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
            for name in output_names
        }

    def _get_scene_device_tensor(self, name, shape, dtype, torch):
        tensor = self.scene_device_tensors.get(name)
        if (
            tensor is None
            or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype
        ):
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
            self.scene_device_tensors[name] = tensor
        return tensor

    def _preprocess_scene_image(self, cv_image):
        if cv_image is None or cv_image.size == 0:
            raise ValueError("Scene image is empty")

        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self.scene_input_width, self.scene_input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])

    def _select_scene_outputs(self, outputs):
        embedding = None
        logits = None

        if self.scene_embedding_output:
            if self.scene_embedding_output not in outputs:
                raise RuntimeError(
                    f"Embedding output '{self.scene_embedding_output}' not in "
                    f"{list(outputs)}"
                )
            embedding = outputs[self.scene_embedding_output]

        if self.scene_logits_output:
            if self.scene_logits_output not in outputs:
                raise RuntimeError(
                    f"Logits output '{self.scene_logits_output}' not in {list(outputs)}"
                )
            logits = outputs[self.scene_logits_output]

        if logits is None:
            logits = next((value for value in outputs.values() if value.size == 365), None)

        if embedding is None:
            non_logits = [
                value
                for value in outputs.values()
                if logits is None or value is not logits
            ]
            if len(non_logits) == 1:
                embedding = non_logits[0]

        return embedding, logits

    def _softmax(self, logits):
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        shifted = logits - float(np.max(logits))
        exp_values = np.exp(shifted)
        denominator = float(np.sum(exp_values))
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError("Invalid scene logits")
        return exp_values / denominator

    def _torch_dtype(self, tensor_rt_dtype, torch):
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        if tensor_rt_dtype not in dtype_map:
            raise RuntimeError(f"Unsupported TensorRT dtype: {tensor_rt_dtype}")
        return dtype_map[tensor_rt_dtype]


    def process_yolo_results(
        self, cv_image, keyframe_id=0, run_ocr: bool = True,
        run_dino: bool = False,
        debug_image=None,
    ):
        perception_started = time.perf_counter()
        h, w, _ = cv_image.shape
        static_objects = []
        pending_ocr = []
        ocr_objects_processed = 0
        object_embedding_ms = 0.0
        dino_requests = []
        dino_inference_ms = 0.0
        dino_pooling_ms = 0.0
        ocr_cache_hits = 0
        ocr_cache_refreshes = 0
        used_ocr_tracks = set()

        yolo_results = self.yolo(
            source=cv_image,
            conf=self.yolo_conf,
            task="segment",
            verbose=False,
        )
        yolo_finished = time.perf_counter()

        for result in yolo_results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None:
                continue



            ordered_indices = sorted(
                range(len(boxes)),
                key=lambda index: (
                    self.yolo.names[int(boxes[index].cls[0])]
                    not in self.ocr_always_classes,
                    -float(boxes[index].conf[0]),
                ),
            )

            for i in ordered_indices:
                box = boxes[i]
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.yolo.names[class_id]

                if confidence < self.yolo_conf:
                    continue

                if self.stable_classes and class_name not in self.stable_classes:
                    continue

                mask = None
                mask_tensor = None
                if masks is not None:



                    mask_tensor = masks.data[i]
                    points = (mask_tensor > 0.5).nonzero(as_tuple=False)
                    if points.numel():
                        mask_h, mask_w = mask_tensor.shape[-2:]
                        geometry = (
                            float(points[:, 1].float().mean().item() / mask_w),
                            float(points[:, 0].float().mean().item() / mask_h),
                            float(points.shape[0] / (mask_w * mask_h)),
                        )
                    else:
                        geometry = None
                elif self.yolo_box_geometry_fallback:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_width = max(0.0, float(x2 - x1))
                    box_height = max(0.0, float(y2 - y1))
                    geometry = (
                        float((x1 + x2) * 0.5 / w),
                        float((y1 + y2) * 0.5 / h),
                        float(box_width * box_height / (w * h)),
                    )
                else:
                    geometry = None

                if geometry is None:
                    continue

                x_centroid, y_centroid, area = geometry
                texts = []
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                should_run_ocr = should_run_ocr_for_object(
                    run_ocr,
                    class_name,
                    self.ocr_classes,
                    self.ocr_always_classes,
                    ocr_objects_processed,
                    self.ocr_max_objects,
                )

                if should_run_ocr:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    x1 = max(0, x1 - self.crop_margin)
                    y1 = max(0, y1 - self.crop_margin)
                    x2 = min(w, x2 + self.crop_margin)
                    y2 = min(h, y2 + self.crop_margin)

                    crop = cv_image[y1:y2, x1:x2]

                    if crop.size > 0:
                        is_sign = class_name in self.ocr_always_classes
                        if (self.isolate_mask_for_ocr and not is_sign
                                and class_name != "building"
                                and mask_tensor is not None):
                            mask = mask_tensor.detach().cpu().numpy()
                            mask = cv2.resize(
                                mask, (w, h), interpolation=cv2.INTER_NEAREST
                            )
                            mask = (mask > 0.5).astype(np.uint8)
                            mask_crop = mask[y1:y2, x1:x2]
                            ocr_crop = crop.copy()
                            ocr_crop[mask_crop == 0] = 255
                            foreground_ratio = float(np.count_nonzero(mask_crop)) / max(
                                1, mask_crop.size
                            )
                            if foreground_ratio < self.ocr_mask_min_foreground_ratio:
                                ocr_crop = crop
                        else:
                            ocr_crop = crop

                        if ocr_crop_is_usable(
                            ocr_crop,
                            self.ocr_min_crop_side,
                            self.ocr_min_crop_pixels,
                            self.ocr_min_sharpness,
                        ):
                            ocr_objects_processed += 1

                static_object = StaticObject(
                    class_name=class_name,
                    seg_conf=confidence,
                    x_centroid=x_centroid,
                    y_centroid=y_centroid,
                    area=area,
                    texts=texts,
                )
                ocr_cache_entry = self._match_ocr_cache(
                    static_object, keyframe_id, used_ocr_tracks
                )
                if ocr_cache_entry is not None:
                    used_ocr_tracks.add(ocr_cache_entry["track_id"])
                    self._touch_ocr_cache(
                        ocr_cache_entry, static_object, keyframe_id
                    )
                    if ocr_cache_entry["texts"]:
                        static_object.texts = [
                            TextAnchor(text=item.text, conf=item.conf)
                            for item in ocr_cache_entry["texts"]
                        ]
                        ocr_cache_hits += 1
                if (
                    self.object_embedding_enabled
                    and self.global_descriptor is not None
                    and len(static_objects) < self.object_embedding_max_objects
                ):
                    embed_x1 = max(0, int(x1))
                    embed_y1 = max(0, int(y1))
                    embed_x2 = min(w, int(x2))
                    embed_y2 = min(h, int(y2))
                    embedding_crop = cv_image[
                        embed_y1:embed_y2, embed_x1:embed_x2
                    ]
                    if embedding_crop.size:
                        embedding_started = time.perf_counter()
                        try:
                            static_object.appearance_embedding = (
                                self.global_descriptor.describe(embedding_crop)
                            )
                        except Exception as exc:
                            self.get_logger().warn(
                                f"Object embedding failed for {class_name}: {exc}"
                            )
                        object_embedding_ms += (
                            time.perf_counter() - embedding_started
                        ) * 1000.0
                static_objects.append(static_object)
                if (
                    run_dino and self.dino_descriptor is not None
                    and class_name in self.dino_candidate_classes
                    and mask_tensor is not None
                ):
                    dino_mask = mask_tensor.detach().cpu().numpy()
                    dino_mask = cv2.resize(
                        dino_mask, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                    dino_requests.append((static_object, dino_mask))
                if should_run_ocr and crop.size > 0 and ocr_crop_is_usable(
                    ocr_crop,
                    self.ocr_min_crop_side,
                    self.ocr_min_crop_pixels,
                    self.ocr_min_sharpness,
                ):
                    crop_quality = self._ocr_crop_quality(ocr_crop)
                    if (
                        ocr_cache_entry is None
                        or should_refresh_cached_ocr(
                            bool(ocr_cache_entry["texts"]),
                            keyframe_id,
                            ocr_cache_entry["last_ocr_frame"],
                            crop_quality,
                            ocr_cache_entry["best_quality"],
                            self.ocr_cache_refresh_interval,
                            self.ocr_cache_retry_interval,
                            self.ocr_cache_quality_gain,
                        )
                    ):
                        pending_ocr.append((
                            static_object, ocr_crop, class_name,
                            ocr_cache_entry, crop_quality,
                        ))
                        if ocr_cache_entry is not None:
                            ocr_cache_refreshes += 1

                if debug_image is not None:
                    color = (60, 210, 255)
                    if mask is None and mask_tensor is not None:
                        mask = mask_tensor.detach().cpu().numpy()
                        mask = cv2.resize(
                            mask, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        mask = (mask > 0.5).astype(np.uint8)
                    if mask is not None:
                        tint = np.zeros_like(debug_image)
                        tint[mask > 0] = color
                        cv2.addWeighted(tint, 0.35, debug_image, 1.0, 0,
                                        dst=debug_image)
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)



                if self.verbose_detections:
                    self.get_logger().info(
                        f"Static object: {class_name}, conf={confidence:.2f}, "
                        f"x={x_centroid:.2f}, y={y_centroid:.2f}, "
                        f"area={area:.5f}, texts={len(texts)}"
                    )

        if dino_requests:
            dino_started = time.perf_counter()
            try:
                dense_features = self.dino_descriptor.describe(cv_image)
            except Exception as exc:
                dense_features = None
                self.get_logger().warn(f"DINOv2 dense inference failed: {exc}")
            dino_inference_ms = (time.perf_counter() - dino_started) * 1000.0
            if dense_features is not None:
                pooling_started = time.perf_counter()
                for static_object, object_mask in dino_requests:
                    static_object.appearance_embedding = pool_mask_features(
                        dense_features, object_mask, mode=self.dino_pooling
                    )
                    if static_object.appearance_embedding is not None:
                        static_object.appearance_model = "dinov2_vits14_dense"
                dino_pooling_ms = (
                    time.perf_counter() - pooling_started
                ) * 1000.0

        postprocess_finished = time.perf_counter()
        if pending_ocr:
            crops = [item[1] for item in pending_ocr]
            classes = [item[2] for item in pending_ocr]
            if self.ocr_batch_enabled:
                batches = self.run_ocr_on_object_crops(crops, classes)
            else:
                batches = [
                    self.run_ocr_on_object_crop(crop, class_name)
                    for crop, class_name in zip(crops, classes)
                ]
            for item, texts in zip(pending_ocr, batches):
                static_object, _, _, cache_entry, crop_quality = item
                if texts:
                    static_object.texts = texts
                elif cache_entry is not None and cache_entry["texts"]:
                    static_object.texts = [
                        TextAnchor(text=value.text, conf=value.conf)
                        for value in cache_entry["texts"]
                    ]
                self._commit_ocr_cache(
                    cache_entry, static_object, texts, crop_quality, keyframe_id
                )

        ocr_finished = time.perf_counter()
        self._last_perception_timings = {
            "yolo_ms": (yolo_finished - perception_started) * 1000.0,
            "object_postprocess_ms": (
                postprocess_finished - yolo_finished
            ) * 1000.0 - object_embedding_ms - dino_inference_ms - dino_pooling_ms,
            "object_embedding_ms": object_embedding_ms,
            "dino_inference_ms": dino_inference_ms,
            "dino_pooling_ms": dino_pooling_ms,
            "dino_tracking_ms": 0.0,
            "dino_search_ms": 0.0,
            "dino_consensus_ms": 0.0,
            "ocr_ms": (ocr_finished - postprocess_finished) * 1000.0,
            "ocr_crop_count": len(pending_ocr),
            "ocr_cache_hit_count": ocr_cache_hits,
            "ocr_cache_refresh_count": ocr_cache_refreshes,
        }

        return static_objects

    @staticmethod
    def _ocr_crop_quality(crop):
        """Prefer larger, sharper landmark views for OCR refreshes."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        height, width = gray.shape[:2]
        return sharpness * float(np.sqrt(max(1, height * width)))

    def _match_ocr_cache(self, obj, frame_id, used_track_ids):
        """Associate a detection with a recent same-class OCR landmark."""
        if not self.ocr_cache_enabled:
            return None
        current_frame = int(frame_id)
        self._ocr_landmark_cache = [
            entry for entry in self._ocr_landmark_cache
            if current_frame - entry["last_seen_frame"]
            <= self.ocr_cache_max_frame_gap
        ]
        best = None
        best_distance = float("inf")
        for entry in self._ocr_landmark_cache:
            if (
                entry["track_id"] in used_track_ids
                or entry["class_name"] != obj.class_name
            ):
                continue
            smaller = max(1e-9, min(float(obj.area), entry["area"]))
            larger = max(float(obj.area), entry["area"])
            if larger / smaller > self.ocr_cache_area_ratio:
                continue
            distance = float(np.hypot(
                obj.x_centroid - entry["x_centroid"],
                obj.y_centroid - entry["y_centroid"],
            ))
            if (
                distance <= self.ocr_cache_centroid_threshold
                and distance < best_distance
            ):
                best, best_distance = entry, distance
        return best

    @staticmethod
    def _touch_ocr_cache(entry, obj, frame_id):
        entry["x_centroid"] = float(obj.x_centroid)
        entry["y_centroid"] = float(obj.y_centroid)
        entry["area"] = float(obj.area)
        entry["last_seen_frame"] = int(frame_id)

    def _commit_ocr_cache(
        self, entry, obj, texts, crop_quality, frame_id,
    ):
        if not self.ocr_cache_enabled:
            return
        copied_texts = [TextAnchor(text=item.text, conf=item.conf) for item in texts]
        if entry is None:
            entry = {
                "track_id": self._next_ocr_track_id,
                "class_name": obj.class_name,
                "x_centroid": float(obj.x_centroid),
                "y_centroid": float(obj.y_centroid),
                "area": float(obj.area),
                "texts": copied_texts,
                "best_quality": float(crop_quality),
                "last_ocr_frame": int(frame_id),
                "last_seen_frame": int(frame_id),
            }
            self._next_ocr_track_id += 1
            self._ocr_landmark_cache.append(entry)
        else:
            self._touch_ocr_cache(entry, obj, frame_id)
            entry["last_ocr_frame"] = int(frame_id)
            if copied_texts:

                merged = {
                    " ".join(item.text.casefold().split()): item
                    for item in entry["texts"]
                }
                for item in copied_texts:
                    key = " ".join(item.text.casefold().split())
                    if key and (key not in merged or item.conf > merged[key].conf):
                        merged[key] = item
                entry["texts"] = list(merged.values())
            entry["best_quality"] = max(
                float(entry["best_quality"]), float(crop_quality)
            )
        if len(self._ocr_landmark_cache) > self.ocr_cache_max_entries:
            self._ocr_landmark_cache.sort(
                key=lambda item: item["last_seen_frame"], reverse=True
            )
            del self._ocr_landmark_cache[self.ocr_cache_max_entries:]

    def compute_mask_geometry(self, mask, image_width: int, image_height: int):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x_centroid = float(np.mean(xs) / image_width)
        y_centroid = float(np.mean(ys) / image_height)
        area = float(len(xs) / (image_width * image_height))

        return x_centroid, y_centroid, area

    def run_ocr_on_crop(self, crop, class_name: str):
        if self._ocr_initialization_failed:
            return []
        try:
            self._ensure_ocr_model()
            ocr_results = self.ocr_model.predict(crop)
        except Exception as exc:
            self.get_logger().warn(f"OCR failed on {class_name}: {exc}")
            return []

        return self._parse_ocr_results(ocr_results, class_name)

    def _parse_ocr_results(self, ocr_results, class_name: str):
        detected_texts = []

        if ocr_results is None:
            return detected_texts

        for result in ocr_results:

            if isinstance(result, TextAnchor):
                if result.conf >= self.ocr_conf and useful_ocr_text(result.text):
                    detected_texts.append(result)
                continue
            if isinstance(result, (list, tuple)):
                detected_texts.extend(
                    self._parse_ocr_results(result, class_name)
                )
                continue


            if isinstance(result, dict):
                result_dict = result


            else:
                result_dict = None

                if hasattr(result, "json"):
                    result_dict = result.json
                    if callable(result_dict):
                        result_dict = result_dict()

                elif hasattr(result, "to_dict"):
                    result_dict = result.to_dict()
                    if callable(result_dict):
                        result_dict = result_dict()

            if result_dict is None:
                self.get_logger().warn(f"Could not parse OCR result: {result}")
                continue




            if isinstance(result_dict.get("res"), dict):
                result_dict = result_dict["res"]

            rec_texts = result_dict.get("rec_texts", [])
            rec_scores = result_dict.get("rec_scores", [])

            for text, confidence in zip(rec_texts, rec_scores):
                confidence = float(confidence)

                if confidence >= self.ocr_conf and useful_ocr_text(text):
                    detected_texts.append(
                        TextAnchor(
                            text=str(text),
                            conf=confidence,
                        )
                    )

                    if self.verbose_detections:
                        self.get_logger().info(
                            f"OCR on {class_name}: '{text}', "
                            f"conf={confidence:.2f}"
                        )

        return detected_texts

    def _ocr_crops_for_object(self, crop, class_name: str):
        """Return bounded OCR tiles for one detected landmark."""
        crops = [crop]
        if class_name == "building":
            height, width = crop.shape[:2]
            tile = self.building_ocr_tile_size
            if height > tile or width > tile:
                step = max(
                    1,
                    int(round(tile * (1.0 - self.building_ocr_tile_overlap))),
                )

                def starts(length):
                    if length <= tile:
                        return [0]
                    values = list(range(0, max(1, length - tile + 1), step))
                    last = length - tile
                    if values[-1] != last:
                        values.append(last)
                    return values

                positions = [
                    (x, y)
                    for y in reversed(starts(height))
                    for x in starts(width)
                ][:self.building_ocr_max_tiles]
                crops = [crop[y:y + tile, x:x + tile] for x, y in positions]
        return crops

    def run_ocr_on_object_crops(self, crops, class_names):
        """Batch all OCR tiles for a keyframe and regroup by source object."""
        if self._ocr_initialization_failed:
            return [[] for _ in crops]
        flattened = []
        owners = []
        for owner, (crop, class_name) in enumerate(zip(crops, class_names)):
            for tile in self._ocr_crops_for_object(crop, class_name):
                flattened.append(tile)
                owners.append(owner)
        if not flattened:
            return [[] for _ in crops]

        try:
            self._ensure_ocr_model()
            results = self.ocr_model.predict(flattened)
        except Exception as exc:
            self.get_logger().warn(f"Batched OCR failed: {exc}")
            return [[] for _ in crops]

        results = [] if results is None else list(results)
        grouped = [dict() for _ in crops]
        for index, result in enumerate(results):
            if index >= len(owners):
                break
            owner = owners[index]
            class_name = class_names[owner]
            for anchor in self._parse_ocr_results([result], class_name):
                key = " ".join(anchor.text.casefold().split())
                current = grouped[owner].get(key)
                if key and (current is None or anchor.conf > current.conf):
                    grouped[owner][key] = anchor
        return [list(items.values()) for items in grouped]

    def run_ocr_on_object_crop(self, crop, class_name: str):
        """Apply class-specific OCR cropping and merge overlapping-tile text."""
        if crop is None or crop.size == 0:
            return []

        crops = self._ocr_crops_for_object(crop, class_name)

        merged = {}
        for item in crops:
            for anchor in self.run_ocr_on_crop(item, class_name):
                key = " ".join(anchor.text.casefold().split())
                if key and (key not in merged or anchor.conf > merged[key].conf):
                    merged[key] = anchor
        return list(merged.values())

    def _ensure_ocr_model(self):
        if self.ocr_model is not None:
            return
        if self._ocr_initialization_failed:
            raise RuntimeError("OCR backend initialization previously failed")

        with self.ocr_model_lock:
            if self.ocr_model is not None:
                return

            if self.ocr_backend == "tensorrt":
                required = {
                    "ocr_detector_engine_path": self.ocr_detector_engine_path,
                    "ocr_recognizer_engine_path": self.ocr_recognizer_engine_path,
                    "ocr_recognizer_yaml_path": self.ocr_recognizer_yaml_path,
                }
                missing = [name for name, value in required.items() if not value]
                if missing:
                    raise RuntimeError(
                        "TensorRT OCR requires: " + ", ".join(missing)
                    )
                try:
                    self.ocr_model = TensorRTPPOCRv5(
                        detector_path=self.ocr_detector_engine_path,
                        recognizer_path=self.ocr_recognizer_engine_path,
                        recognizer_yaml=self.ocr_recognizer_yaml_path,
                        confidence_threshold=self.ocr_conf,
                        detector_side=self.ocr_text_det_limit_side_len,
                    )
                except Exception:
                    self._ocr_initialization_failed = True
                    raise
                self.ocr_device = "gpu:0"
                self.get_logger().info("PP-OCRv5 TensorRT engines loaded on GPU:0")
                return

            import paddle
            from paddleocr import PaddleOCR

            use_gpu = self.ocr_use_gpu and paddle.device.is_compiled_with_cuda()
            if self.ocr_use_gpu and not use_gpu:
                self.get_logger().warn(
                    "ocr_use_gpu is true, but PaddlePaddle is CPU-only; "
                    "falling back to CPU"
                )

            self.ocr_device = "gpu:0" if use_gpu else "cpu"
            try:
                self.ocr_model = PaddleOCR(
                    lang=self.ocr_language,
                    device=self.ocr_device,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                    text_recognition_batch_size=max(1, self.ocr_max_objects),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_limit_side_len=self.ocr_text_det_limit_side_len,
                    text_det_limit_type="max",
                    enable_hpi=self.ocr_enable_hpi,
                    use_tensorrt=self.ocr_use_tensorrt and use_gpu,
                    precision=self.ocr_tensorrt_precision,
                    enable_mkldnn=self.ocr_enable_mkldnn and not use_gpu,
                    cpu_threads=self.ocr_cpu_threads,
                )
            except Exception:
                self._ocr_initialization_failed = True
                raise
            backend = (
                f"TensorRT {self.ocr_tensorrt_precision}"
                if self.ocr_use_tensorrt and use_gpu else "Paddle"
            )
            self.get_logger().info(
                f"PaddleOCR model loaded on {self.ocr_device} via {backend}"
            )

    def _warmup_ocr(self):
        try:
            self._ensure_ocr_model()
            warmup_image = np.zeros((64, 256, 3), dtype=np.uint8)
            self.ocr_model.predict(warmup_image)
            self.get_logger().info(f"{self.ocr_backend} OCR warm-up complete")
        except Exception as exc:
            self.get_logger().warn(f"{self.ocr_backend} OCR warm-up failed: {exc}")


    def _rank_candidates(
        self,
        query_keyframe,
        query_scene_seq,
        candidate_keyframes,
        candidate_scene_sequences,
    ):
        ranked = []
        for candidate, scene_sequence in zip(
            candidate_keyframes,
            candidate_scene_sequences,
        ):
            breakdown = self.matcher.score_breakdown(
                query_keyframe=query_keyframe,
                candidate_keyframe=candidate,
                query_scene_seq=query_scene_seq,
                candidate_scene_seq=scene_sequence,
            )
            raw_score = float(breakdown["unified_score"])
            score, evidence_label = apply_scene_only_score_policy(
                breakdown,
                raw_score,
                self.semantic_threshold,
                self.scene_only_effective_score,
            )
            breakdown["raw_unified_score"] = raw_score
            breakdown["effective_score"] = score
            breakdown["evidence_label"] = evidence_label
            breakdown["candidate_source"] = self._candidate_provenance.get(
                int(candidate.keyframe_id), "scene"
            )
            breakdown["submission_threshold"] = self._submission_threshold()
            ranked.append((candidate, score, breakdown))

        ranked.sort(
            key=lambda item: (
                item[1], item[2].get("raw_unified_score", item[1])
            ),
            reverse=True,
        )
        return ranked

    def _submission_threshold(self):
        """Return the calibrated threshold for the active layer ablation."""
        if not self.use_scene and self.use_object:
            return (
                self.object_text_threshold
                if self.use_text else self.object_only_threshold
            )
        return self.semantic_threshold

    def _publish_semantic_candidates(
        self,
        query_frame_id,
        query_reference_keyframe_id,
        ranked_candidates,
        response_header,
    ):
        message = SemanticCandidates()
        if response_header is not None:
            message.header = response_header

        message.query_frame_id = int(query_frame_id)
        message.query_reference_keyframe_id = int(
            query_reference_keyframe_id or 0
        )

        selected = candidates_above_threshold(
            ranked_candidates,
            self.semantic_threshold,
            self.candidate_response_count,
        )
        message.candidate_keyframe_ids = [
            int(candidate.orb_keyframe_id)
            if candidate.orb_keyframe_id is not None
            else int(candidate.keyframe_id)
            for candidate, _, _ in selected
        ]
        message.candidate_map_ids = [
            int(candidate.orb_map_id or 0) for candidate, _, _ in selected
        ]
        message.candidate_poses = [
            self._matrix_to_pose(candidate.pose) for candidate, _, _ in selected
        ]
        message.semantic_scores = [float(score) for _, score, _ in selected]
        message.best_score = (
            float(message.semantic_scores[0])
            if message.semantic_scores
            else 0.0
        )

        message.ambiguous = (
            len(message.semantic_scores) > 1
            and message.semantic_scores[0] - message.semantic_scores[1]
            < self.semantic_ambiguity_margin
        )
        message.accepted = bool(message.semantic_scores)
        self.semantic_candidates_publisher.publish(message)

        if message.candidate_keyframe_ids:
            self.get_logger().info(
                f"Semantic candidate for frame {query_frame_id}: "
                f"keyframe={message.candidate_keyframe_ids[0]}, "
                f"map={message.candidate_map_ids[0]}, "
                f"score={message.best_score:.3f}, "
                f"accepted={message.accepted}, "
                f"ambiguous={message.ambiguous}"
            )

    @staticmethod
    def _matrix_to_pose(matrix):
        message = Pose()
        if matrix is None:
            message.orientation.w = 1.0
            return message

        message.position.x = float(matrix[0, 3])
        message.position.y = float(matrix[1, 3])
        message.position.z = float(matrix[2, 3])
        rotation = matrix[:3, :3]
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            message.orientation.w = 0.25 * scale
            message.orientation.x = (rotation[2, 1] - rotation[1, 2]) / scale
            message.orientation.y = (rotation[0, 2] - rotation[2, 0]) / scale
            message.orientation.z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            diagonal = int(np.argmax(np.diag(rotation)))
            if diagonal == 0:
                scale = np.sqrt(
                    1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
                ) * 2.0
                message.orientation.w = (rotation[2, 1] - rotation[1, 2]) / scale
                message.orientation.x = 0.25 * scale
                message.orientation.y = (rotation[0, 1] + rotation[1, 0]) / scale
                message.orientation.z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif diagonal == 1:
                scale = np.sqrt(
                    1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
                ) * 2.0
                message.orientation.w = (rotation[0, 2] - rotation[2, 0]) / scale
                message.orientation.x = (rotation[0, 1] + rotation[1, 0]) / scale
                message.orientation.y = 0.25 * scale
                message.orientation.z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(
                    1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
                ) * 2.0
                message.orientation.w = (rotation[1, 0] - rotation[0, 1]) / scale
                message.orientation.x = (rotation[0, 2] + rotation[2, 0]) / scale
                message.orientation.y = (rotation[1, 2] + rotation[2, 1]) / scale
                message.orientation.z = 0.25 * scale
        return message

    def build_candidate_inputs(self, query_scene_seq=None, query_keyframe=None):
        """
        Returns:
            candidate_keyframes:
                [C_i]

            candidate_scene_sequences:
                [[scene_i, scene_i-1, scene_i-2], ...]
        """
        candidates = []
        newest_eligible_index = (
            len(self.map_memory) - self.candidate_min_separation
        )

        for i, keyframe in enumerate(self.map_memory[:newest_eligible_index]):
            if query_keyframe is not None:
                query_orb_id = (
                    query_keyframe.orb_keyframe_id
                    if query_keyframe.orb_keyframe_id is not None
                    else query_keyframe.keyframe_id
                )
                candidate_orb_id = (
                    keyframe.orb_keyframe_id
                    if keyframe.orb_keyframe_id is not None
                    else keyframe.keyframe_id
                )



                if abs(int(query_orb_id) - int(candidate_orb_id)) < self.candidate_min_separation:
                    continue
                if (
                    query_keyframe.source_frame_id is not None
                    and keyframe.source_frame_id is not None
                    and abs(
                        int(query_keyframe.source_frame_id)
                        - int(keyframe.source_frame_id)
                    ) < self.candidate_min_separation
                ):
                    continue
            scene_seq = []

            for j in range(i, max(-1, i - 3), -1):
                scene = self.map_memory[j].scene

                if scene is not None:
                    scene_seq.append(scene)

            scene_score = (
                self.matcher.scene_similarity(query_scene_seq, scene_seq)
                if query_scene_seq is not None
                else 0.0
            )



            object_retrieval_score = self._object_embedding_retrieval_score(
                query_keyframe, keyframe
            )
            candidates.append((
                scene_score, object_retrieval_score, i, keyframe, scene_seq
            ))

        candidates.sort(key=lambda item: item[0], reverse=True)
        self._candidate_provenance = {}
        if self.use_scene:
            scene_candidates = candidates[:self.candidate_rerank_top_k]
            scene_indices = {item[2] for item in scene_candidates}
            for item in scene_candidates:
                self._candidate_provenance[item[2]] = "scene"
            if getattr(self, "dino_object_enabled", False):
                dino_indices = self._dino_candidate_indices(
                    query_keyframe, {item[2] for item in candidates}
                )
                by_index = {item[2]: item for item in candidates}
                for index in dino_indices[:self.dino_candidate_top_k]:
                    if index in scene_indices:
                        self._candidate_provenance[index] = "scene+dino_object"
                    elif index in by_index:
                        scene_candidates.append(by_index[index])
                        self._candidate_provenance[index] = "dino_object"
                candidates = scene_candidates
            elif getattr(self, "object_embedding_enabled", False):
                object_candidates = sorted(
                    candidates, key=lambda item: item[1], reverse=True
                )[:self.object_embedding_candidate_top_k]
                selected_indices = {
                    item[2] for item in scene_candidates + object_candidates
                    if item[0] > 0.0 or item[1] > 0.0
                }
                candidates = [
                    item for item in candidates if item[2] in selected_indices
                ]
            else:
                candidates = scene_candidates

        selected = []
        for scene_score, _, index, keyframe, scene_seq in candidates:
            cache = getattr(self, "_semantic_neighborhood_cache", None)
            cache_key = (
                int(keyframe.keyframe_id),
                int(keyframe.source_frame_id)
                if keyframe.source_frame_id is not None else -1,
            )
            neighborhood = cache.get(cache_key) if cache is not None else None
            if neighborhood is None:
                prior_records = [
                    self.map_memory[j]
                    for j in range(index - 1, max(-1, index - 3), -1)
                ]
                neighborhood = self._semantic_neighborhood(
                    keyframe, prior_records
                )
                if cache is not None:
                    cache[cache_key] = neighborhood
            selected.append((
                scene_score,
                neighborhood,
                scene_seq,
                self._candidate_provenance.get(index, "scene"),
            ))

        self._candidate_provenance = {
            int(item[1].keyframe_id): item[3] for item in selected
        }

        return (
            [item[1] for item in selected],
            [item[2] for item in selected],
        )

    def _dino_candidate_indices(self, query, eligible_indices):
        """Retrieve landmark tracks and require historical cluster consensus."""
        if not self.dino_object_enabled or query is None or not eligible_indices:
            return []
        search_started = time.perf_counter()
        query_by_track = {}
        for obj in query.static_objects:
            track_id = getattr(obj, "landmark_track_id", None)
            if (
                track_id is None or obj.appearance_embedding is None
                or obj.class_name not in self.dino_candidate_classes
                or not getattr(obj, "appearance_model", "").startswith("dinov2")
            ):
                continue
            existing = query_by_track.get(track_id)
            if existing is None or obj.seg_conf > existing.seg_conf:
                query_by_track[track_id] = obj

        votes = []
        for query_track_id, obj in query_by_track.items():
            entries = [
                item for item in self._landmark_index.get(obj.class_name, [])
                if item["memory_index"] in eligible_indices
            ]
            if not entries:
                continue


            by_stored_track = {}
            for entry in entries:
                similarity = float(np.dot(
                    obj.appearance_embedding, entry["embedding"]
                ))
                previous = by_stored_track.get(entry["track_id"])
                if previous is None or similarity > previous[0]:
                    by_stored_track[entry["track_id"]] = (similarity, entry)
            ranked = sorted(by_stored_track.values(), reverse=True,
                            key=lambda item: item[0])
            if not ranked:
                continue
            best_similarity = ranked[0][0]
            second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
            if (
                best_similarity < self.dino_similarity_threshold
                or best_similarity - second_similarity < self.dino_similarity_margin
            ):
                continue
            for similarity, entry in ranked[:self.dino_search_neighbors_per_track]:
                if similarity < self.dino_similarity_threshold:
                    break
                votes.append({
                    "query_track_id": query_track_id,
                    "class_name": obj.class_name,
                    "has_text": bool(obj.texts),
                    "similarity": similarity,
                    "memory_index": entry["memory_index"],
                })
        self._last_perception_timings["dino_search_ms"] = (
            time.perf_counter() - search_started
        ) * 1000.0

        consensus_started = time.perf_counter()
        clusters = []
        for vote in sorted(votes, key=lambda item: item["memory_index"]):
            frame = self.map_memory[vote["memory_index"]].source_frame_id
            frame = int(frame if frame is not None else vote["memory_index"])
            target = next((cluster for cluster in clusters
                           if abs(frame - cluster["centre"]) <=
                           self.dino_candidate_cluster_width), None)
            if target is None:
                target = {"centre": frame, "votes": []}
                clusters.append(target)
            target["votes"].append(vote)
            target["centre"] = int(np.median([
                int(self.map_memory[item["memory_index"]].source_frame_id
                    if self.map_memory[item["memory_index"]].source_frame_id is not None
                    else item["memory_index"])
                for item in target["votes"]
            ]))

        accepted = []
        for cluster in clusters:
            best_by_query_track = {}
            for vote in cluster["votes"]:
                previous = best_by_query_track.get(vote["query_track_id"])
                if previous is None or vote["similarity"] > previous["similarity"]:
                    best_by_query_track[vote["query_track_id"]] = vote
            unique_votes = list(best_by_query_track.values())
            distinctive_single = (
                len(unique_votes) == 1
                and unique_votes[0]["class_name"] in self.dino_single_landmark_classes
                and unique_votes[0]["has_text"]
            )
            if (
                len(unique_votes) < self.dino_min_consensus_tracks
                and not distinctive_single
            ):
                continue
            representative = max(
                unique_votes, key=lambda item: item["similarity"]
            )["memory_index"]
            score = float(np.mean([
                item["similarity"] for item in unique_votes
            ]))
            accepted.append((score, len(unique_votes), representative))
        accepted.sort(reverse=True)
        self._last_perception_timings["dino_consensus_ms"] = (
            time.perf_counter() - consensus_started
        ) * 1000.0
        return [item[2] for item in accepted]

    def _object_embedding_retrieval_score(self, query, candidate):
        """Cheap same-class vector search used only to widen the shortlist."""
        if not getattr(self, "object_embedding_enabled", False) or query is None:
            return 0.0
        query_objects = [
            obj for obj in query.static_objects
            if getattr(obj, "appearance_embedding", None) is not None
        ]
        candidate_objects = [
            obj for obj in candidate.static_objects
            if getattr(obj, "appearance_embedding", None) is not None
        ]
        if not query_objects or not candidate_objects:
            return 0.0
        weighted = 0.0
        denominator = 0.0
        for query_object in query_objects:
            weight = (
                float(query_object.seg_conf)
                * self.object_class_weights.get(query_object.class_name, 1.0)
            )
            denominator += weight
            peers = [
                item for item in candidate_objects
                if item.class_name == query_object.class_name
            ]
            if not peers:
                continue
            similarities = [
                float(np.dot(
                    query_object.appearance_embedding,
                    item.appearance_embedding,
                ))
                for item in peers
            ]
            weighted += weight * float(np.clip(
                (max(similarities) + 1.0) * 0.5, 0.0, 1.0
            ))
        return weighted / denominator if denominator > 0.0 else 0.0

    def _semantic_neighborhood(self, centre, prior_records):
        """Return a non-recursive local semantic representation.

        The centre retains its ORB identity and pose. Missing objects/text may
        be supplied by two earlier local keyframes with decayed confidence.
        Nearby duplicates are merged so a frequently detected pole cannot
        inflate the object count merely by appearing in several frames.
        """
        if not self.semantic_neighborhood_enabled:
            return centre

        objects = []
        records = [centre, *list(prior_records)]
        weights = self.semantic_neighborhood_weights
        for offset, record in enumerate(records[:len(weights)]):
            decay = weights[offset]
            if decay <= 0.0:
                continue
            for source in record.static_objects:
                confidence = float(source.seg_conf)
                if offset:



                    confidence *= decay * decay
                copied = StaticObject(
                    class_name=source.class_name,
                    seg_conf=confidence,
                    x_centroid=source.x_centroid,
                    y_centroid=source.y_centroid,
                    area=source.area,
                    texts=list(source.texts),
                    observation_count=max(
                        1, int(getattr(source, "observation_count", 1))
                    ),
                    appearance_embedding=getattr(
                        source, "appearance_embedding", None
                    ),
                    appearance_model=getattr(source, "appearance_model", ""),
                    landmark_track_id=getattr(source, "landmark_track_id", None),
                )
                nearest = None
                nearest_distance = float("inf")
                for existing in objects:
                    if existing.class_name != copied.class_name:
                        continue
                    distance = self.matcher.mask_distance(existing, copied)
                    if distance < nearest_distance:
                        nearest, nearest_distance = existing, distance
                if (
                    nearest is not None
                    and nearest_distance <= self.semantic_neighborhood_merge_distance
                ):
                    nearest.seg_conf = max(nearest.seg_conf, copied.seg_conf)
                    nearest.observation_count += copied.observation_count
                    if (
                        nearest.appearance_embedding is not None
                        and copied.appearance_embedding is not None
                        and nearest.landmark_track_id == copied.landmark_track_id
                    ):
                        combined = (
                            nearest.appearance_embedding
                            + decay * copied.appearance_embedding
                        )
                        norm = float(np.linalg.norm(combined))
                        if norm > 1e-12:
                            nearest.appearance_embedding = combined / norm
                    known = {" ".join(text.text.casefold().split())
                             for text in nearest.texts}
                    nearest.texts.extend(
                        text for text in copied.texts
                        if " ".join(text.text.casefold().split()) not in known
                    )
                else:
                    objects.append(copied)

        return KeyframeRecord(
            keyframe_id=centre.keyframe_id,
            timestamp=centre.timestamp,
            scene=centre.scene,
            static_objects=objects,
            pose=centre.pose,
            orb_keyframe_id=centre.orb_keyframe_id,
            orb_map_id=centre.orb_map_id,
            tracking_inliers=centre.tracking_inliers,
            source_frame_id=centre.source_frame_id,
        )

    def _assign_landmark_tracks(self, current, prior_records):
        """Associate DINO landmarks over a short temporal neighbourhood."""
        if not self.dino_object_enabled:
            return
        previous = [
            obj for record in reversed(prior_records)
            for obj in record.static_objects
            if getattr(obj, "appearance_model", "").startswith("dinov2")
            and getattr(obj, "landmark_track_id", None) is not None
        ]
        used_tracks = set()
        for obj in current.static_objects:
            embedding = getattr(obj, "appearance_embedding", None)
            if embedding is None or obj.class_name not in self.dino_candidate_classes:
                continue
            best, best_similarity = None, -1.0
            for candidate in previous:
                if (
                    candidate.class_name != obj.class_name
                    or candidate.landmark_track_id in used_tracks
                ):
                    continue
                centroid_distance = float(np.hypot(
                    obj.x_centroid - candidate.x_centroid,
                    obj.y_centroid - candidate.y_centroid,
                ))
                if centroid_distance > self.dino_track_centroid_threshold:
                    continue
                similarity = float(np.dot(
                    embedding, candidate.appearance_embedding
                ))
                if similarity > best_similarity:
                    best, best_similarity = candidate, similarity
            if best is not None and best_similarity >= self.dino_track_similarity_threshold:
                obj.landmark_track_id = best.landmark_track_id
                obj.observation_count = best.observation_count + 1
                aggregated = (
                    embedding
                    + self.dino_temporal_decay * best.appearance_embedding
                )
                norm = float(np.linalg.norm(aggregated))
                if norm > 1e-12:
                    obj.appearance_embedding = aggregated / norm
                used_tracks.add(best.landmark_track_id)
            else:
                obj.landmark_track_id = self._next_landmark_track_id
                self._next_landmark_track_id += 1

    def _index_landmark_keyframe(self, keyframe, memory_index):
        if not self.dino_object_enabled:
            return
        for obj in keyframe.static_objects:
            if (
                obj.class_name not in self.dino_candidate_classes
                or getattr(obj, "appearance_model", "") != "dinov2_vits14_dense"
                or obj.appearance_embedding is None
            ):
                continue
            self._landmark_index.setdefault(obj.class_name, []).append({
                "embedding": obj.appearance_embedding,
                "memory_index": int(memory_index),
                "track_id": obj.landmark_track_id,
                "source_frame_id": keyframe.source_frame_id,
            })

    def semantic_pose_recovery(
        self,
        query_keyframe: KeyframeRecord,
        selected_candidate: KeyframeRecord,
        semantic_score: float,
    ):
        """
        Hook for ORB-SLAM3 pose recovery.

        Semantics has selected the keyframe.
        The selected keyframe's geometry/local map should be used for pose adjustment.
        """
        self.get_logger().warn(
            f"Semantic recovery requested: query={query_keyframe.keyframe_id}, "
            f"candidate={selected_candidate.keyframe_id}, score={semantic_score:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HuMemSLAMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
