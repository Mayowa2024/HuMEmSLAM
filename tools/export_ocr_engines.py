#!/usr/bin/env python3
"""Build bounded TensorRT engines for HuMemSLAM's PP-OCRv5 ONNX models."""

import argparse
import json
from pathlib import Path

import tensorrt as trt


def build(onnx_path, engine_path, shapes, workspace_gib):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
    )
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, *shapes)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path}")
    engine_path.write_bytes(bytes(serialized))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=Path("weights/ocr_onnx"))
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    args = parser.parse_args()
    directory = args.onnx_dir.resolve()
    detector = directory / "PP-OCRv5_mobile_det.engine"
    recognizer = directory / "en_PP-OCRv5_mobile_rec.engine"
    build(
        directory / "PP-OCRv5_mobile_det.onnx", detector,
        ((1, 3, 32, 32), (1, 3, 320, 640), (1, 3, 640, 640)),
        args.workspace_gib,
    )
    build(
        directory / "en_PP-OCRv5_mobile_rec.onnx", recognizer,
        ((1, 3, 48, 320), (4, 3, 48, 320), (18, 3, 48, 640)),
        args.workspace_gib,
    )
    metadata = {
        "detector_engine": detector.name,
        "recognizer_engine": recognizer.name,
        "precision": "TensorRT 11 FP16-eligible kernels",
        "detector_profile": [[1, 3, 32, 32], [1, 3, 320, 640], [1, 3, 640, 640]],
        "recognizer_profile": [[1, 3, 48, 320], [4, 3, 48, 320], [18, 3, 48, 640]],
    }
    (directory / "ocr_engines.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Detector: {detector}\nRecognizer: {recognizer}")


if __name__ == "__main__":
    main()
