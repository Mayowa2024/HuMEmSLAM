#!/usr/bin/env python3
"""Rectify and split a ZED UVC SBS stream into synchronized stereo images.

This is an emergency stereo-only path for when a ZED enumerates at USB 2 speed
and the ZED SDK cannot open it. It uses the camera's factory calibration file.
"""

import configparser

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy._rclpy_pybind11 import RCLError
from sensor_msgs.msg import CameraInfo, Image


def _value(section, name, fallback_name=None, default=0.0):
    if name in section:
        return float(section[name])
    if fallback_name and fallback_name in section:
        return float(section[fallback_name])
    return float(default)


def load_vga_rectification(calibration_file, width=672, height=376):
    """Return remap tables, rectified projections and effective baseline."""
    config = configparser.ConfigParser()
    if not config.read(calibration_file):
        raise FileNotFoundError(f"Cannot read calibration file: {calibration_file}")

    def camera(side):
        section = config[f"{side}_CAM_VGA"]
        matrix = np.array(
            [
                [_value(section, "fx"), 0.0, _value(section, "cx")],
                [0.0, _value(section, "fy"), _value(section, "cy")],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.array(
            [_value(section, key) for key in ("k1", "k2", "p1", "p2", "k3")],
            dtype=np.float64,
        )
        return matrix, distortion

    left_k, left_d = camera("LEFT")
    right_k, right_d = camera("RIGHT")
    stereo = config["STEREO"]
    rotation_vector = np.array(
        [
            _value(stereo, "RX_VGA"),
            _value(stereo, "CV_VGA"),
            _value(stereo, "RZ_VGA"),
        ],
        dtype=np.float64,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)

    translation_mm = np.array(
        [
            -_value(stereo, "Baseline"),
            _value(stereo, "TY_VGA", "TY"),
            _value(stereo, "TZ_VGA", "TZ"),
        ],
        dtype=np.float64,
    )
    r1, r2, p1, p2 = cv2.stereoRectify(
        left_k,
        left_d,
        right_k,
        right_d,
        (width, height),
        rotation,
        translation_mm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=(width, height),
    )[:4]
    left_maps = cv2.initUndistortRectifyMap(
        left_k, left_d, r1, p1, (width, height), cv2.CV_32FC1
    )
    right_maps = cv2.initUndistortRectifyMap(
        right_k, right_d, r2, p2, (width, height), cv2.CV_32FC1
    )
    baseline_m = abs(float(p2[0, 3] / p2[0, 0])) / 1000.0
    return left_maps, right_maps, p1, p2, baseline_m


def make_rectified_camera_info(projection, width, height, frame_id):
    info = CameraInfo()
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = [
        float(projection[0, 0]), 0.0, float(projection[0, 2]),
        0.0, float(projection[1, 1]), float(projection[1, 2]),
        0.0, 0.0, 1.0,
    ]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [float(value) for value in projection.reshape(-1)]
    info.header.frame_id = frame_id
    return info


class ZedSbsSplitter(Node):
    def __init__(self):
        super().__init__("zed_sbs_splitter")
        self.declare_parameter("input_topic", "/image_raw")
        self.declare_parameter(
            "left_topic", "/zed/zed_node/left/color/rect/image"
        )
        self.declare_parameter(
            "right_topic", "/zed/zed_node/right/color/rect/image"
        )
        self.declare_parameter("left_frame_id", "zed_left_camera_frame_optical")
        self.declare_parameter("right_frame_id", "zed_right_camera_frame_optical")
        self.declare_parameter(
            "calibration_file", "/usr/local/zed/settings/SN34213728.conf"
        )
        self.declare_parameter("rectify", True)

        qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(
            Image, str(self.get_parameter("left_topic").value), qos
        )
        self.right_pub = self.create_publisher(
            Image, str(self.get_parameter("right_topic").value), qos
        )
        self.left_info_pub = self.create_publisher(
            CameraInfo, "/zed/zed_node/left/color/rect/camera_info", qos
        )
        self.right_info_pub = self.create_publisher(
            CameraInfo, "/zed/zed_node/right/color/rect/camera_info", qos
        )
        self.rectify = bool(self.get_parameter("rectify").value)
        self.left_maps = self.right_maps = None
        self.left_projection = self.right_projection = None
        if self.rectify:
            (
                self.left_maps,
                self.right_maps,
                self.left_projection,
                self.right_projection,
                baseline_m,
            ) = load_vga_rectification(
                str(self.get_parameter("calibration_file").value)
            )
            self.get_logger().info(
                "Loaded factory VGA calibration; rectified baseline "
                f"{baseline_m:.6f} m"
            )
        self.frames = 0
        self.create_subscription(
            Image,
            str(self.get_parameter("input_topic").value),
            self.on_image,
            qos,
        )
        self.get_logger().warning(
            "ZED UVC fallback active: factory-rectified stereo; no ZED SDK IMU"
        )

    def on_image(self, msg: Image) -> None:
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = image.shape[:2]
        if width % 2:
            self.get_logger().error(f"Side-by-side width must be even, got {width}")
            return

        midpoint = width // 2
        left = image[:, :midpoint]
        right = image[:, midpoint:]
        if self.rectify:
            left = cv2.remap(left, *self.left_maps, interpolation=cv2.INTER_LINEAR)
            right = cv2.remap(right, *self.right_maps, interpolation=cv2.INTER_LINEAR)
        left_msg = self.bridge.cv2_to_imgmsg(left, encoding="bgr8")
        right_msg = self.bridge.cv2_to_imgmsg(right, encoding="bgr8")


        left_msg.header.stamp = msg.header.stamp
        right_msg.header.stamp = msg.header.stamp
        left_msg.header.frame_id = str(self.get_parameter("left_frame_id").value)
        right_msg.header.frame_id = str(self.get_parameter("right_frame_id").value)
        try:
            self.left_pub.publish(left_msg)
            self.right_pub.publish(right_msg)
            if self.rectify:
                left_info = make_rectified_camera_info(
                    self.left_projection, midpoint, height, left_msg.header.frame_id
                )
                right_info = make_rectified_camera_info(
                    self.right_projection, midpoint, height, right_msg.header.frame_id
                )
                left_info.header.stamp = msg.header.stamp
                right_info.header.stamp = msg.header.stamp
                self.left_info_pub.publish(left_info)
                self.right_info_pub.publish(right_info)
        except RCLError:
            if rclpy.ok():
                raise
            return

        self.frames += 1
        if self.frames == 1:
            self.get_logger().info(
                f"Publishing stereo pair {midpoint}x{height} from {width}x{height} SBS"
            )


def main() -> None:
    rclpy.init()
    node = ZedSbsSplitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
