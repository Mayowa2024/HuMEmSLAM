#!/usr/bin/env bash

ZED_SETUP=/home/teleopbike/zed_ws/install/setup.bash
source /opt/ros/humble/setup.bash
if [[ -f "$ZED_SETUP" ]]; then
  source "$ZED_SETUP"
fi
set -u

echo "HuMemSLAM bike readiness check"
echo "Generated: $(date --iso-8601=seconds)"

echo
echo "[USB topology]"
lsusb -t || true

echo
echo "[ZED USB identity]"
lsusb -d 2b03: || true

echo
echo "[Video devices]"
v4l2-ctl --list-devices 2>/dev/null || true

echo
echo "[Disk space]"
df -h /home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam

echo
echo "[GPU]"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw \
  --format=csv,noheader 2>/dev/null || true

echo
echo "[ROS nodes]"
ros2 node list 2>/dev/null || true

echo
echo "[Required ZED topics]"
required_topics=(
  /zed/zed_node/left/color/rect/image
  /zed/zed_node/right/color/rect/image
  /zed/zed_node/left/camera_info
  /zed/zed_node/right/camera_info
  /zed/zed_node/imu/data_raw
  /zed/zed_node/left_cam_imu_transform
  /tf_static
)
topic_list="$(ros2 topic list 2>/dev/null || true)"
missing=0
for topic in "${required_topics[@]}"; do
  if grep -Fxq "$topic" <<<"$topic_list"; then
    echo "PASS $topic"
  else
    echo "MISS $topic"
    missing=$((missing + 1))
  fi
done

echo
echo "[Result]"
if [[ "$missing" -eq 0 ]]; then
  echo "PASS: all required ZED topics are advertised. Run the live rate/timestamp checks."
  exit 0
fi
echo "NOT READY: $missing required topic(s) are missing."
exit 1
