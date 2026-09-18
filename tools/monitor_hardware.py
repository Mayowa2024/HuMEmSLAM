#!/usr/bin/env python3
"""Record reproducibility hardware state before, during and after a command."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, text=True, capture_output=True,
                              timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def memory() -> dict[str, int]:
    values = {}
    for line in read_text("/proc/meminfo").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0])
    return values


def cpu_sample(previous=None):
    fields = read_text("/proc/stat").splitlines()[0].split()[1:]
    counters = [int(value) for value in fields]
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    total = sum(counters)
    utilisation = None
    if previous:
        total_delta = total - previous[0]
        idle_delta = idle - previous[1]
        if total_delta > 0:
            utilisation = 100.0 * (total_delta - idle_delta) / total_delta
    return (total, idle), utilisation


def cpu_frequency_mhz() -> float | None:
    values = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        text = read_text(str(path))
        if text.isdigit():
            values.append(int(text) / 1000.0)
    return sum(values) / len(values) if values else None


def temperatures() -> dict[str, float]:
    result = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        value = read_text(str(zone / "temp"))
        if not value:
            continue
        label = read_text(str(zone / "type")) or zone.name
        try:
            temperature = float(value)
            result[label] = temperature / 1000.0 if temperature > 500 else temperature
        except ValueError:
            pass
    return result


def gpu_sample() -> dict[str, str]:
    if not shutil.which("nvidia-smi"):
        return {}
    query = (
        "name,driver_version,temperature.gpu,utilization.gpu,"
        "memory.used,memory.total,clocks.sm,power.draw"
    )
    output = command_output([
        "nvidia-smi", f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ])
    if not output or "failed" in output.lower():
        return {}
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    keys = ["name", "driver_version", "temperature_c", "utilisation_percent",
            "memory_used_mib", "memory_total_mib", "sm_clock_mhz", "power_w"]
    return dict(zip(keys, values)) if len(values) == len(keys) else {}


def snapshot() -> dict:
    mem = memory()
    governors = sorted({read_text(str(path)) for path in
                       Path("/sys/devices/system/cpu").glob(
                           "cpu[0-9]*/cpufreq/scaling_governor") if read_text(str(path))})
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": platform.node(), "platform": platform.platform(),
        "kernel": platform.release(), "cpu_count": os.cpu_count(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in
                           read_text("/proc/cpuinfo").splitlines()
                           if line.startswith("model name")), ""),
        "cpu_governors": governors,
        "cpu_frequency_mhz": cpu_frequency_mhz(),
        "temperatures_c": temperatures(),
        "memory_kib": {key: mem.get(key, 0) for key in
                       ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")},
        "load_average": list(os.getloadavg()), "gpu": gpu_sample(),
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
        "cuda_version": command_output(["nvcc", "--version"]) if shutil.which("nvcc") else "",
    }


class HardwareMonitor:
    def __init__(self, output: Path, interval: float = 5.0):
        self.output = output
        self.interval = max(0.5, interval)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "system_before.json").write_text(
            json.dumps(snapshot(), indent=2) + "\n", encoding="utf-8")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self, exit_code=None):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 2)
        after = snapshot()
        after["monitored_command_exit_code"] = exit_code
        (self.output / "system_after.json").write_text(
            json.dumps(after, indent=2) + "\n", encoding="utf-8")

    def _run(self):
        path = self.output / "hardware_monitor.csv"
        header = ["wall_time", "elapsed_s", "cpu_utilisation_percent",
                  "cpu_frequency_mhz", "load_1m", "memory_available_mib",
                  "swap_used_mib", "max_temperature_c", "gpu_utilisation_percent",
                  "gpu_temperature_c", "gpu_memory_used_mib", "gpu_sm_clock_mhz",
                  "gpu_power_w"]
        start = time.monotonic()
        previous, _ = cpu_sample()
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            while not self.stop_event.is_set():
                counters, cpu_util = cpu_sample(previous)
                previous = counters
                mem = memory()
                gpu = gpu_sample()
                temps = temperatures().values()
                writer.writerow({
                    "wall_time": datetime.now().astimezone().isoformat(),
                    "elapsed_s": f"{time.monotonic() - start:.3f}",
                    "cpu_utilisation_percent": "" if cpu_util is None else f"{cpu_util:.3f}",
                    "cpu_frequency_mhz": cpu_frequency_mhz(),
                    "load_1m": os.getloadavg()[0],
                    "memory_available_mib": mem.get("MemAvailable", 0) / 1024.0,
                    "swap_used_mib": (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024.0,
                    "max_temperature_c": max(temps) if temps else "",
                    "gpu_utilisation_percent": gpu.get("utilisation_percent", ""),
                    "gpu_temperature_c": gpu.get("temperature_c", ""),
                    "gpu_memory_used_mib": gpu.get("memory_used_mib", ""),
                    "gpu_sm_clock_mhz": gpu.get("sm_clock_mhz", ""),
                    "gpu_power_w": gpu.get("power_w", ""),
                })
                stream.flush()
                self.stop_event.wait(self.interval)


def main():
    parser = argparse.ArgumentParser(
        description="Record hardware state while running a command."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command after -- (for example: -- ros2 launch ...)")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Provide a command after --")
    monitor = HardwareMonitor(args.output, args.interval)
    monitor.start()
    code = 1
    try:
        code = subprocess.run(command).returncode
    finally:
        monitor.stop(code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
