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
from slam.ocr_policy import should_run_ocr_for_object
from slam.scene_categories import group_places365_probabilities
from slam.types import KeyframeRecord, SceneRecord, StaticObject, TextAnchor


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
        if float(item[1]) > float(threshold)
    ][:max(0, int(limit))]


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


class HumanSLAMNode(Node):
    """
    ROS 2 wrapper for the HumanSLAM semantic recovery model.

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

        # Current query scene sequence only.
        self.scene_history = deque(maxlen=3)

        # Full keyframe memory.
        self.map_memory = []

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
        )

        self.get_logger().info("HumanSLAM node initialising")
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
            f"HumanSLAM active — {', '.join(backends)} ready"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

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
        # Optional VPR descriptor.  When enabled it replaces the Places365
        # embedding for retrieval/scoring while Places365 still supplies the
        # scene label and grouped category evidence.
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
        # PaddlePaddle 3.3.1 currently fails on these OCRv5 models when the
        # oneDNN/MKLDNN executor is enabled.
        self.declare_parameter("ocr_enable_mkldnn", False)
        self.declare_parameter("ocr_cpu_threads", 4)
        self.declare_parameter("ocr_text_det_limit_side_len", 640)

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
        self.declare_parameter("use_scene", True)
        self.declare_parameter("use_object", True)
        self.declare_parameter("use_text", True)
        self.declare_parameter("sigma_mask", 0.25)
        self.declare_parameter("lambda_area", 1.0)
        self.declare_parameter("text_geom_threshold", 0.6)
        self.declare_parameter("text_conflict_floor", 0.25)

        self.declare_parameter("semantic_threshold", 0.70)
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
        self.candidate_min_separation = max(
            0,
            int(self.get_parameter("candidate_min_keyframe_separation").value),
        )
        self.candidate_response_count = max(
            1, int(self.get_parameter("candidate_response_count").value)
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
        self.use_scene = bool(self.get_parameter("use_scene").value)
        self.use_object = bool(self.get_parameter("use_object").value)
        self.use_text = bool(self.get_parameter("use_text").value)
        if not any((self.use_scene, self.use_object, self.use_text)):
            raise ValueError("At least one HumanSLAM layer must be enabled")
        self.sigma_mask = float(self.get_parameter("sigma_mask").value)
        self.lambda_area = float(self.get_parameter("lambda_area").value)
        self.text_geom_threshold = float(self.get_parameter("text_geom_threshold").value)
        self.text_conflict_floor = float(
            self.get_parameter("text_conflict_floor").value
        )

        self.semantic_threshold = float(self.get_parameter("semantic_threshold").value)
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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        self.global_descriptor = None
        self.scene_runtime = None
        self.scene_engine = None
        self.scene_context = None
        self.scene_tensor_names = []
        self.scene_labels = self._load_scene_labels(self.scene_labels_path)
        self.scene_device_tensors = {}
        self.scene_stream = None

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
                # Keep node alive so YOLO/OCR can still be tested.
                self.scene_engine = None
                self.scene_context = None

        if self.use_scene and self.global_descriptor_enabled:
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
            # TensorRT engine files do not carry enough filename metadata for
            # Ultralytics to infer the task reliably. Declare segmentation at
            # construction time or the engine silently returns boxes without
            # masks and HumanSLAM falls back to coarser box geometry.
            self.yolo = YOLO(self.yolo_model_path, task="segment")
            self.get_logger().info(f"YOLO model loaded: {self.yolo_model_path}")
            self.get_logger().debug(f"YOLO class names: {self.yolo.names}")

        # OCR is intentionally lazy: model construction is expensive and OCR is
        # only needed on selected keyframes and text-bearing object classes.
        self.ocr_model = None
        self.ocr_device = "cpu"
        self.ocr_model_lock = threading.Lock()

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

        # Keep the runtime alive for the lifetime of the engine.
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
                    # Official Places365 files use: "/a/airfield 0".
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

    # ------------------------------------------------------------------
    # Keyframe callback
    # ------------------------------------------------------------------

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
            # Recovery needs the newest observation, not a backlog of stale
            # keyframes. Replace the queued item while leaving active work alone.
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
        static_objects = (
            self.process_yolo_results(
                cv_image, run_ocr=run_ocr, debug_image=debug_image
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

        query_scene_seq = list(self.scene_history)
        ranked_candidates = []
        if run_candidate_retrieval:
            candidate_keyframes, candidate_scene_sequences = (
                self.build_candidate_inputs(query_scene_seq, current_keyframe)
            )
            ranked_candidates = self._rank_candidates(
                current_keyframe,
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
            current_keyframe,
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
        # Fixed-height diagnostics outside the camera image prevent the video
        # geometry from changing as detections appear and disappear.
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
        accepted = best_score > self.semantic_threshold
        output = Path(self.candidate_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "wall_time_ns", "query_frame_id", "query_image_path",
            "query_reference_keyframe_id", "rank", "published",
            "candidate_keyframe_id", "candidate_map_id",
            "candidate_source_frame_id", "candidate_image_path",
            "scene_score", "object_score", "text_score", "text_evidence",
            "raw_unified_score", "effective_score", "evidence_label",
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
                int(rank <= selected_count and score > self.semantic_threshold),
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

    # ------------------------------------------------------------------
    # Scene processing
    # ------------------------------------------------------------------

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
                # None prevents a failed frame from being compared with a
                # successful descriptor whose engine-defined size may differ
                # from the configured fallback dimension.
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

        # Safe auto-detection for common Places365 exports.
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

    # ------------------------------------------------------------------
    # YOLO segmentation + OCR processing
    # ------------------------------------------------------------------

    def process_yolo_results(
        self, cv_image, run_ocr: bool = True, debug_image=None
    ):
        h, w, _ = cv_image.shape
        static_objects = []
        ocr_objects_processed = 0

        yolo_results = self.yolo(
            source=cv_image,
            conf=self.yolo_conf,
            task="segment",
            verbose=False,
        )

        for result in yolo_results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None:
                continue

            # Prioritise explicit signs within the bounded OCR budget. Broad
            # regions such as buildings remain interval-controlled.
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
                if masks is not None:
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(
                        mask, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    mask = (mask > 0.5).astype(np.uint8)
                    geometry = self.compute_mask_geometry(mask, w, h)
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
                        if self.isolate_mask_for_ocr and mask is not None:
                            mask_crop = mask[y1:y2, x1:x2]
                            ocr_crop = crop.copy()
                            ocr_crop[mask_crop == 0] = 255
                        else:
                            ocr_crop = crop

                        texts = self.run_ocr_on_crop(ocr_crop, class_name)
                        ocr_objects_processed += 1

                static_objects.append(
                    StaticObject(
                        class_name=class_name,
                        seg_conf=confidence,
                        x_centroid=x_centroid,
                        y_centroid=y_centroid,
                        area=area,
                        texts=texts,
                    )
                )

                if debug_image is not None:
                    color = (60, 210, 255)
                    if mask is not None:
                        tint = np.zeros_like(debug_image)
                        tint[mask > 0] = color
                        cv2.addWeighted(tint, 0.35, debug_image, 1.0, 0,
                                        dst=debug_image)
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                    # Labels and OCR are written in the fixed black footer by
                    # _write_debug_frame, not over moving image content.

                if self.verbose_detections:
                    self.get_logger().info(
                        f"Static object: {class_name}, conf={confidence:.2f}, "
                        f"x={x_centroid:.2f}, y={y_centroid:.2f}, "
                        f"area={area:.5f}, texts={len(texts)}"
                    )

        return static_objects

    def compute_mask_geometry(self, mask, image_width: int, image_height: int):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x_centroid = float(np.mean(xs) / image_width)
        y_centroid = float(np.mean(ys) / image_height)
        area = float(len(xs) / (image_width * image_height))

        return x_centroid, y_centroid, area

    def run_ocr_on_crop(self, crop, class_name: str):
        detected_texts = []

        try:
            self._ensure_ocr_model()
            ocr_results = self.ocr_model.predict(crop)
        except Exception as exc:
            self.get_logger().warn(f"OCR failed on {class_name}: {exc}")
            return detected_texts

        if ocr_results is None:
            return detected_texts

        for result in ocr_results:

            # PaddleOCR v3 may return a dict directly
            if isinstance(result, dict):
                result_dict = result

            # Or it may return an object with json/to_dict
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

            # Your successful output has rec_texts directly at the top level
            rec_texts = result_dict.get("rec_texts", [])
            rec_scores = result_dict.get("rec_scores", [])

            for text, confidence in zip(rec_texts, rec_scores):
                confidence = float(confidence)

                if confidence >= self.ocr_conf and str(text).strip():
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

    def _ensure_ocr_model(self):
        if self.ocr_model is not None:
            return

        with self.ocr_model_lock:
            if self.ocr_model is not None:
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
                enable_mkldnn=self.ocr_enable_mkldnn and not use_gpu,
                cpu_threads=self.ocr_cpu_threads,
            )
            self.get_logger().info(f"PaddleOCR model loaded on {self.ocr_device}")

    def _warmup_ocr(self):
        try:
            self._ensure_ocr_model()
            warmup_image = np.zeros((64, 256, 3), dtype=np.uint8)
            self.ocr_model.predict(warmup_image)
            self.get_logger().info("PaddleOCR warm-up complete")
        except Exception as exc:
            self.get_logger().warn(f"PaddleOCR warm-up failed: {exc}")

    # ------------------------------------------------------------------
    # Candidate building and recovery
    # ------------------------------------------------------------------

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
            ranked.append((candidate, score, breakdown))

        ranked.sort(
            key=lambda item: (
                item[1], item[2].get("raw_unified_score", item[1])
            ),
            reverse=True,
        )
        return ranked

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
                # Async semantic storage means list-index distance alone is not
                # a reliable measure of age. Enforce both ORB-keyframe and
                # source-frame separation explicitly.
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
            candidates.append((scene_score, keyframe, scene_seq))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.use_scene:
            candidates = candidates[: self.candidate_top_k]

        return (
            [item[1] for item in candidates],
            [item[2] for item in candidates],
        )

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
        node = HumanSLAMNode()
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
