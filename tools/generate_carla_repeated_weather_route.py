#!/usr/bin/env python3
"""Record one continuous, repeated-route CARLA stereo-inertial experiment.

The ego vehicle drives the same closed route five times without teleportation.
Weather changes only while stopped at the origin between laps.  Output follows
the KITTI-like convention already used by HuMemSLAM.
"""

import argparse
import csv
import json
import math
import queue
import random
import shutil
import time
from pathlib import Path


WEATHER_NAMES = (
    "ClearNoon",
    "HardRainNoon",
    "ClearSunset",
    "ClearNight",
    "RainNight",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-spawn", type=int, default=0)
    parser.add_argument(
        "--route-anchor-spawns", default="",
        help=("Comma-separated spawn indices. The origin is appended when needed. "
              "Use this after visually selecting a compact, text-rich circuit."))
    parser.add_argument(
        "--route-radius-m", type=float, default=120.0,
        help="Approximate radius used by the automatic compact-route selector.")
    parser.add_argument("--route-sampling-m", type=float, default=2.0)
    parser.add_argument("--target-speed-kmh", type=float, default=22.0)
    parser.add_argument("--arrival-radius-m", type=float, default=4.0)
    parser.add_argument("--weather-hold-seconds", type=float, default=3.0)
    parser.add_argument("--sim-fps", type=int, default=100)
    parser.add_argument("--camera-fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--baseline-m", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--max-lap-seconds", type=float, default=600.0)
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--disk-check-frames", type=int, default=100)
    return parser.parse_args()


def angle_wrap(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def speed_mps(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def distance_2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def import_carla_api():
    try:
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError as exc:
        raise SystemExit(
            "CARLA PythonAPI is unavailable. Run this script from CARLA's "
            "PythonAPI environment or add its .egg/.whl to PYTHONPATH."
        ) from exc
    return carla, GlobalRoutePlanner


def choose_route_anchors(spawn_points, origin_index, radius_m):
    """Choose a reproducible compact triangular circuit and return to origin."""
    origin = spawn_points[origin_index]
    others = [(distance_2d(origin.location, point.location), i, point)
              for i, point in enumerate(spawn_points) if i != origin_index]
    second = min(others, key=lambda item: abs(item[0] - radius_m))[2]
    candidates = [
        point for _, _, point in others
        if point is not second and 0.45 * radius_m <=
        distance_2d(origin.location, point.location) <= 1.35 * radius_m
    ]
    if not candidates:
        candidates = [point for _, _, point in others if point is not second]
    third = max(candidates, key=lambda point: distance_2d(
        point.location, second.location))
    return [origin, second, third, origin]


def explicit_route_anchors(spawn_points, value, origin_index):
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        return None
    if indices[0] != origin_index:
        indices.insert(0, origin_index)
    if indices[-1] != origin_index:
        indices.append(origin_index)
    invalid = [index for index in indices if not 0 <= index < len(spawn_points)]
    if invalid:
        raise ValueError(f"Invalid route spawn indices: {invalid}")
    return [spawn_points[index] for index in indices]


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def require_disk_space(path, minimum_gb):
    available = free_gb(path)
    if available < minimum_gb:
        raise RuntimeError(
            f"Only {available:.1f} GiB free; at least {minimum_gb:.1f} GiB is required")


def build_closed_route(carla_map, planner_class, anchors, sampling_resolution):
    planner = planner_class(carla_map, sampling_resolution)
    route = []
    for start, end in zip(anchors[:-1], anchors[1:]):
        segment = planner.trace_route(start.location, end.location)
        points = [waypoint.transform for waypoint, _ in segment]
        if route and points:
            points = points[1:]
        route.extend(points)
    if len(route) < 20:
        raise RuntimeError("Generated route is unexpectedly short")
    return route


class RouteController:
    def __init__(self, carla, route, target_speed_kmh, arrival_radius):
        self.carla = carla
        self.route = route
        self.target = target_speed_kmh / 3.6
        self.arrival_radius = arrival_radius
        self.index = 0

    def reset(self):
        self.index = 0

    def step(self, vehicle):
        transform = vehicle.get_transform()
        location = transform.location
        while self.index < len(self.route) - 1 and distance_2d(
                location, self.route[self.index].location) < 3.0:
            self.index += 1
        remaining = distance_2d(location, self.route[-1].location)
        finished = self.index >= len(self.route) - 1 and remaining < self.arrival_radius
        if finished:
            return self.carla.VehicleControl(throttle=0.0, brake=1.0), True

        lookahead = min(self.index + 3, len(self.route) - 1)
        target = self.route[lookahead].location
        desired = math.atan2(target.y - location.y, target.x - location.x)
        heading = math.radians(transform.rotation.yaw)
        error = angle_wrap(desired - heading)
        steer = max(-1.0, min(1.0, 1.5 * error))
        current_speed = speed_mps(vehicle)
        speed_error = self.target - current_speed
        throttle = max(0.0, min(0.65, 0.35 + 0.20 * speed_error))
        brake = max(0.0, min(0.7, -0.25 * speed_error))
        if abs(error) > 0.55:
            throttle = min(throttle, 0.25)
        return self.carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake), False


def sensor_blueprints(world, args):
    library = world.get_blueprint_library()
    cameras = []
    for name in ("left", "right"):
        bp = library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(args.width))
        bp.set_attribute("image_size_y", str(args.height))
        bp.set_attribute("fov", str(args.fov))
        bp.set_attribute("sensor_tick", str(1.0 / args.camera_fps))
        cameras.append((name, bp))
    imu = library.find("sensor.other.imu")
    imu.set_attribute("sensor_tick", str(1.0 / args.sim_fps))
    return cameras, imu


def image_array(image, np, cv2):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def write_image(path, array, cv2):
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Failed to write image: {path}")


def write_metadata(path, args, route, anchors):
    fx = args.width / (2.0 * math.tan(math.radians(args.fov) / 2.0))
    data = {
        "experiment": "repeated_closed_route_weather_cycle",
        "dataset_status": "RECORDING",
        "continuous_recording": True,
        "weather_order": list(WEATHER_NAMES),
        "route_waypoint_count": len(route),
        "anchor_locations": [
            {"x": p.location.x, "y": p.location.y, "z": p.location.z}
            for p in anchors
        ],
        "simulation": {"town": args.town, "sim_fps": args.sim_fps,
                       "fixed_delta_seconds": 1.0 / args.sim_fps,
                       "seed": args.seed},
        "stereo_camera": {"width": args.width, "height": args.height,
                          "fov_degrees": args.fov,
                          "fps": args.camera_fps,
                          "baseline_m": args.baseline_m,
                          "fx": fx, "fy": fx,
                          "cx": args.width / 2.0,
                          "cy": args.height / 2.0,
                          "orbslam3_bf": fx * args.baseline_m},
        "imu": {"fps": args.sim_fps},
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    args = parse_args()
    carla, planner_class = import_carla_api()
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy and opencv-python are required") from exc

    random.seed(args.seed)
    output = args.output.expanduser().resolve()
    left_dir, right_dir = output / "image_0", output / "image_1"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    if any(left_dir.iterdir()) or any(right_dir.iterdir()):
        raise RuntimeError(f"Refusing to record into non-empty dataset: {output}")
    require_disk_space(output, args.min_free_gb)
    (output / "RECORDING_IN_PROGRESS").write_text(
        "Dataset is incomplete while this file exists.\n")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.town)
    original_settings = world.get_settings()
    actors = []
    left_queue, right_queue, imu_queue = queue.Queue(), queue.Queue(), queue.Queue()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.sim_fps
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        spawn_points = world.get_map().get_spawn_points()
        if not 0 <= args.origin_spawn < len(spawn_points):
            raise ValueError(f"origin spawn must be in 0..{len(spawn_points)-1}")
        anchors = explicit_route_anchors(
            spawn_points, args.route_anchor_spawns, args.origin_spawn)
        if anchors is None:
            anchors = choose_route_anchors(
                spawn_points, args.origin_spawn, args.route_radius_m)
        route = build_closed_route(
            world.get_map(), planner_class, anchors, args.route_sampling_m)
        write_metadata(output / "metadata.json", args, route, anchors)

        vehicle_bp = world.get_blueprint_library().filter(args.vehicle_filter)[0]
        vehicle = world.spawn_actor(vehicle_bp, anchors[0])
        actors.append(vehicle)

        cameras, imu_bp = sensor_blueprints(world, args)
        camera_x, camera_z = 1.5, 1.6
        left_tf = carla.Transform(carla.Location(
            x=camera_x, y=-args.baseline_m / 2.0, z=camera_z))
        right_tf = carla.Transform(carla.Location(
            x=camera_x, y=args.baseline_m / 2.0, z=camera_z))
        left = world.spawn_actor(cameras[0][1], left_tf, attach_to=vehicle)
        right = world.spawn_actor(cameras[1][1], right_tf, attach_to=vehicle)
        imu = world.spawn_actor(
            imu_bp, carla.Transform(carla.Location(x=camera_x, z=camera_z)),
            attach_to=vehicle)
        actors.extend([left, right, imu])
        left.listen(left_queue.put)
        right.listen(right_queue.put)
        imu.listen(imu_queue.put)

        controller = RouteController(
            carla, route, args.target_speed_kmh, args.arrival_radius_m)
        pending_left, pending_right = {}, {}
        transforms_by_frame = {}
        image_index = 0
        start_elapsed = None

        with (output / "times.txt").open("w") as times_file,\
             (output / "pose_gt.csv").open("w", newline="") as pose_file,\
             (output / "imu.csv").open("w", newline="") as imu_file,\
             (output / "frame_metadata.csv").open("w", newline="") as frame_file,\
             (output / "lap_events.csv").open("w", newline="") as lap_file:
            pose_writer, imu_writer = csv.writer(pose_file), csv.writer(imu_file)
            frame_writer, lap_writer = csv.writer(frame_file), csv.writer(lap_file)
            pose_writer.writerow(["image_index", "carla_frame", "timestamp",
                                  "x", "y", "z", "roll", "pitch", "yaw",
                                  "lap_id", "weather"])
            imu_writer.writerow(["carla_frame", "timestamp", "accel_x", "accel_y",
                                 "accel_z", "gyro_x", "gyro_y", "gyro_z",
                                 "compass", "lap_id", "weather"])
            frame_writer.writerow(["image_index", "carla_frame", "timestamp",
                                   "lap_id", "weather"])
            lap_writer.writerow(["event", "lap_id", "weather", "carla_frame",
                                 "timestamp"])

            for lap_id, weather_name in enumerate(WEATHER_NAMES):
                weather = getattr(carla.WeatherParameters, weather_name)
                world.set_weather(weather)
                controller.reset()
                lap_start = world.get_snapshot().timestamp.elapsed_seconds
                lap_writer.writerow(["WEATHER_SET", lap_id, weather_name,
                                     world.get_snapshot().frame, lap_start])

                hold_ticks = int(args.weather_hold_seconds * args.sim_fps)
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
                for _ in range(hold_ticks):
                    world.tick()

                finished = False
                while not finished:
                    control, arrived = controller.step(vehicle)
                    vehicle.apply_control(control)
                    world.tick()
                    snapshot = world.get_snapshot()
                    transforms_by_frame[snapshot.frame] = vehicle.get_transform()
                    now = snapshot.timestamp.elapsed_seconds
                    if now - lap_start > args.max_lap_seconds:
                        raise RuntimeError(f"Lap {lap_id} exceeded maximum duration")

                    while not imu_queue.empty():
                        sample = imu_queue.get_nowait()
                        imu_writer.writerow([
                            sample.frame, sample.timestamp,
                            sample.accelerometer.x, sample.accelerometer.y,
                            sample.accelerometer.z, sample.gyroscope.x,
                            sample.gyroscope.y, sample.gyroscope.z, sample.compass,
                            lap_id, weather_name])
                    while not left_queue.empty():
                        sample = left_queue.get_nowait()
                        pending_left[sample.frame] = sample
                    while not right_queue.empty():
                        sample = right_queue.get_nowait()
                        pending_right[sample.frame] = sample
                    for frame in sorted(set(pending_left) & set(pending_right)):
                        left_image, right_image = pending_left.pop(frame), pending_right.pop(frame)
                        timestamp = left_image.timestamp
                        if start_elapsed is None:
                            start_elapsed = timestamp
                        write_image(left_dir / f"{image_index:06d}.png",
                                    image_array(left_image, np, cv2), cv2)
                        write_image(right_dir / f"{image_index:06d}.png",
                                    image_array(right_image, np, cv2), cv2)
                        transform = transforms_by_frame.pop(
                            frame, vehicle.get_transform())
                        loc, rot = transform.location, transform.rotation
                        times_file.write(f"{timestamp - start_elapsed:.9f}\n")
                        pose_writer.writerow([image_index, frame, timestamp,
                                              loc.x, loc.y, loc.z,
                                              rot.roll, rot.pitch, rot.yaw,
                                              lap_id, weather_name])
                        frame_writer.writerow([image_index, frame, timestamp,
                                               lap_id, weather_name])
                        image_index += 1
                        if image_index % args.disk_check_frames == 0:
                            for stream in (times_file, pose_file, imu_file,
                                           frame_file, lap_file):
                                stream.flush()
                            require_disk_space(output, args.min_free_gb)
                    if arrived:
                        vehicle.apply_control(carla.VehicleControl(brake=1.0))
                        finished = speed_mps(vehicle) < 0.15

                lap_writer.writerow(["LAP_COMPLETE", lap_id, weather_name,
                                     snapshot.frame, snapshot.timestamp.elapsed_seconds])
                print(f"Completed lap {lap_id + 1}/{len(WEATHER_NAMES)}: "
                      f"{weather_name}; recorded frames={image_index}", flush=True)

        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["dataset_status"] = "COMPLETE"
        metadata["stereo_pair_count"] = image_index
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "RECORDING_IN_PROGRESS").unlink()
        (output / "DATASET_COMPLETE").write_text(
            f"stereo_pairs={image_index}\n")
        print(f"Dataset complete: {output} ({image_index} stereo pairs)")
    finally:
        for actor in reversed(actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
