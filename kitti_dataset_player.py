import re
import threading
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


class KittiDatasetPlayer(Node):
    """Publish a KITTI-style stereo sequence without depending on its origin."""

    def __init__(self):
        super().__init__("kitti_dataset_player")
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("left_dir", "image_0")
        self.declare_parameter("right_dir", "image_1")
        self.declare_parameter("times_file", "times.txt")
        self.declare_parameter("left_topic", "/dataset/left/image_raw")
        self.declare_parameter("right_topic", "/dataset/right/image_raw")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("playback_rate", 1.0)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)
        self.declare_parameter("startup_delay", 3.0)
        self.declare_parameter("completion_delay", 2.0)
        self.declare_parameter("stereo_publish_gap", 0.01)
        self.declare_parameter("results_dir", "")
        self.declare_parameter("imu_file", "")
        self.declare_parameter("imu_topic", "/dataset/imu")
        self.declare_parameter("imu_frame_id", "imu")
        self.declare_parameter("imu_decimation", 1)

        root_value = str(self.get_parameter("dataset_path").value)
        if not root_value:
            raise ValueError("dataset_path is required")

        self.root = Path(root_value).expanduser().resolve()
        self.left_dir = self.root / str(self.get_parameter("left_dir").value)
        self.right_dir = self.root / str(self.get_parameter("right_dir").value)
        self.times_path = self.root / str(self.get_parameter("times_file").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.playback_rate = float(self.get_parameter("playback_rate").value)
        self.start_frame = max(0, int(self.get_parameter("start_frame").value))
        self.end_frame = int(self.get_parameter("end_frame").value)
        self.startup_delay = max(
            0.0, float(self.get_parameter("startup_delay").value)
        )
        self.completion_delay = max(
            0.0, float(self.get_parameter("completion_delay").value)
        )
        self.stereo_publish_gap = max(
            0.0, float(self.get_parameter("stereo_publish_gap").value)
        )
        results_dir = str(self.get_parameter("results_dir").value)
        if results_dir:
            Path(results_dir).expanduser().mkdir(parents=True, exist_ok=True)

        self.left_images = self._find_images(self.left_dir)
        self.right_images = self._find_images(self.right_dir)
        self.timestamps = self._read_timestamps(self.times_path)
        imu_file = str(self.get_parameter("imu_file").value)
        self.imu_samples = (
            self._read_imu(self.root / imu_file) if imu_file else []
        )
        self.imu_frame_id = str(self.get_parameter("imu_frame_id").value)
        self.imu_decimation = max(
            1, int(self.get_parameter("imu_decimation").value)
        )
        self.imu_index = 0
        self._validate_sequence()

        final_index = len(self.timestamps)
        if self.end_frame >= 0:
            final_index = min(final_index, self.end_frame + 1)
        self.indices = list(range(self.start_frame, final_index))
        if not self.indices:
            raise ValueError("Selected frame range is empty")

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.left_pub = self.create_publisher(
            Image, str(self.get_parameter("left_topic").value), qos
        )
        self.right_pub = self.create_publisher(
            Image, str(self.get_parameter("right_topic").value), qos
        )
        self.imu_pub = self.create_publisher(
            Imu,
            str(self.get_parameter("imu_topic").value),
            QoSProfile(depth=2000, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.bridge = CvBridge()
        self.current = 0
        self.wall_start = None
        self.sequence_start = self.timestamps[self.indices[0]]
        self.completed = False

        self.get_logger().info(
            f"Offline dataset ready: {self.root} "
            f"({len(self.indices)} stereo pairs)"
        )
        if self.imu_samples:
            self.get_logger().info(
                f"IMU ready: {len(self.imu_samples)} samples "
                f"(decimation={self.imu_decimation})"
            )
        self.get_logger().info(
            "Waiting for ORB-SLAM3/HuMemSLAM subscribers before playback"
        )
        self.timer = self.create_timer(
            max(self.startup_delay, 0.01), self._start_playback
        )

    @staticmethod
    def _find_images(directory):
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted(
            (path for path in directory.iterdir() if path.suffix.lower() in allowed),
            key=_natural_key,
        )
        if not images:
            raise FileNotFoundError(f"No supported images found in: {directory}")
        return images

    @staticmethod
    def _read_timestamps(path):
        if not path.is_file():
            raise FileNotFoundError(f"Timestamp file not found: {path}")
        timestamps = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            value = line.strip()
            if not value:
                continue
            try:
                fields = value.split()


                timestamps.append(
                    float(fields[1]) if len(fields) >= 2 else float(fields[0])
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid timestamp at {path}:{line_number}: {value}"
                ) from exc
        if not timestamps:
            raise ValueError(f"No timestamps found in: {path}")
        return timestamps

    def _validate_sequence(self):
        counts = (
            len(self.left_images),
            len(self.right_images),
            len(self.timestamps),
        )
        if len(set(counts)) != 1:
            raise ValueError(
                "Stereo sequence length mismatch: "
                f"left={counts[0]}, right={counts[1]}, times={counts[2]}"
            )
        previous = self.timestamps[0]
        for timestamp in self.timestamps[1:]:
            if timestamp < previous:
                raise ValueError("timestamps must be monotonically increasing")
            previous = timestamp

    @staticmethod
    def _read_imu(path):
        if not path.is_file():
            raise FileNotFoundError(f"IMU file not found: {path}")
        samples = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(
                    f"Expected 7 IMU fields at {path}:{line_number}"
                )
            values = [float(value) for value in fields]

            samples.append((values[0] * 1e-9, *values[1:]))
        if not samples:
            raise ValueError(f"No IMU samples found in: {path}")
        return samples

    @staticmethod
    def _set_stamp(message, timestamp):
        seconds = int(timestamp)
        nanoseconds = int(round((timestamp - seconds) * 1e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        message.header.stamp.sec = seconds
        message.header.stamp.nanosec = nanoseconds

    def _publish_imu_until(self, timestamp):
        while (self.imu_index < len(self.imu_samples) and
               self.imu_samples[self.imu_index][0] <= timestamp):
            sample_index = self.imu_index
            sample = self.imu_samples[sample_index]
            self.imu_index += 1
            if sample_index % self.imu_decimation:
                continue
            message = Imu()
            self._set_stamp(message, sample[0])
            message.header.frame_id = self.imu_frame_id
            message.angular_velocity.x = sample[1]
            message.angular_velocity.y = sample[2]
            message.angular_velocity.z = sample[3]
            message.linear_acceleration.x = sample[4]
            message.linear_acceleration.y = sample[5]
            message.linear_acceleration.z = sample[6]
            message.orientation_covariance[0] = -1.0
            self.imu_pub.publish(message)

    def _start_playback(self):
        self.timer.cancel()
        self.wall_start = time.monotonic()
        self.get_logger().info(
            f"Playback active at {self.playback_rate:.2f}x"
        )
        self.timer = self.create_timer(0.001, self._publish_when_due)

    def _publish_when_due(self):
        if self.completed:
            return

        index = self.indices[self.current]
        if self.playback_rate > 0.0:
            sequence_elapsed = self.timestamps[index] - self.sequence_start
            due = self.wall_start + sequence_elapsed / self.playback_rate
            if time.monotonic() < due:
                return

        left = cv2.imread(str(self.left_images[index]), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(self.right_images[index]), cv2.IMREAD_UNCHANGED)
        if left is None or right is None:
            raise RuntimeError(
                f"Failed to read stereo pair at dataset frame {index}"
            )

        timestamp = self.timestamps[index]
        self._publish_imu_until(timestamp)

        encoding = self._encoding_for(left)
        left_msg = self.bridge.cv2_to_imgmsg(left, encoding=encoding)
        right_encoding = self._encoding_for(right)
        right_msg = self.bridge.cv2_to_imgmsg(right, encoding=right_encoding)
        for message in (left_msg, right_msg):
            self._set_stamp(message, timestamp)


            message.header.frame_id = f"{self.frame_id}|dataset_frame={index}"




        self.left_pub.publish(left_msg)
        if self.stereo_publish_gap > 0.0:
            time.sleep(self.stereo_publish_gap)
        self.right_pub.publish(right_msg)
        self.current += 1

        if self.current == 1 or self.current % 100 == 0:
            self.get_logger().info(
                f"Sequence progress: {self.current}/{len(self.indices)}"
            )

        if self.current >= len(self.indices):
            self.completed = True
            self.timer.cancel()
            self.get_logger().info(
                "Sequence complete; allowing pending SLAM work to finish"
            )
            self.timer = self.create_timer(
                max(self.completion_delay, 0.01), self._finish
            )

    @staticmethod
    def _encoding_for(image):
        if image.dtype != "uint8":
            return "passthrough"
        if image.ndim == 2:
            return "mono8"
        channels = image.shape[2]
        if channels == 3:
            return "bgr8"
        if channels == 4:
            return "bgra8"
        return "passthrough"

    def _finish(self):
        self.timer.cancel()
        self.get_logger().info("Offline dataset player finished")


        threading.Thread(target=rclpy.shutdown, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KittiDatasetPlayer()
        rclpy.spin(node)
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print(f"KITTI dataset player error: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
