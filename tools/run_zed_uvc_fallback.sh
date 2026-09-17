#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
set -u

ROOT="/home/teleopbike/Documents/Mayowa/ros2_ws/src/slam/slam"
CONFIG="${ROOT}/config/zed2i_uvc_fallback.yaml"
SPLITTER="${ROOT}/tools/zed_sbs_splitter.py"

cleanup() {
  kill "${USB_CAM_PID:-}" "${SPLITTER_PID:-}" 2>/dev/null || true
  wait "${USB_CAM_PID:-}" "${SPLITTER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 run usb_cam usb_cam_node_exe --ros-args --params-file "${CONFIG}" &
USB_CAM_PID=$!

python3 "${SPLITTER}" &
SPLITTER_PID=$!

echo "ZED UVC fallback started. Press Ctrl+C to stop both nodes."
wait -n "${USB_CAM_PID}" "${SPLITTER_PID}"
