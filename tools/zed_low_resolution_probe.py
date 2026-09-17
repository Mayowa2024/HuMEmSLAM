"""Short diagnostic: VGA/15 FPS stereo capture without depth processing."""
import json
import sys
import time
import pyzed.sl as sl

camera = sl.Camera()
params = sl.InitParameters()
params.camera_resolution = sl.RESOLUTION.VGA
params.camera_fps = 15
params.depth_mode = sl.DEPTH_MODE.NONE
params.sdk_verbose = 1
try:
    status = camera.open(params)
    print(json.dumps({'open_status': str(status), 'requested_resolution': 'VGA',
                      'requested_fps': 15}), flush=True)
    if status != sl.ERROR_CODE.SUCCESS:
        sys.exit(1)
    left, right = sl.Mat(), sl.Mat()
    good = 0
    deadline = time.monotonic() + 10
    while good < 15 and time.monotonic() < deadline:
        if camera.grab() == sl.ERROR_CODE.SUCCESS:
            a = camera.retrieve_image(left, sl.VIEW.LEFT)
            b = camera.retrieve_image(right, sl.VIEW.RIGHT)
            if a == sl.ERROR_CODE.SUCCESS and b == sl.ERROR_CODE.SUCCESS:
                good += 1
    print(json.dumps({'stereo_pairs_retrieved': good,
                      'left_size': [left.get_width(), left.get_height()]}), flush=True)
    sys.exit(0 if good == 15 else 2)
finally:
    camera.close()
