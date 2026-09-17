#!/usr/bin/env python3
"""Export official DINOv2 ViT-S/14 patch tokens to ONNX and TensorRT."""

import argparse
import json
import subprocess
import shutil
from pathlib import Path

import torch


class DenseDino(torch.nn.Module):
    def __init__(self, model, grid):
        super().__init__()
        self.model = model
        self.grid = grid

    def forward(self, images):
        tokens = self.model.forward_features(images)["x_norm_patchtokens"]
        return tokens.transpose(1, 2).reshape(
            images.shape[0], tokens.shape[2], self.grid, self.grid
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path,
                   default=Path("weights/dinov2"))
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--workspace-mib", type=int, default=2048)
    args = p.parse_args()
    if args.input_size % 14:
        raise ValueError("Input size must be divisible by DINOv2 patch size 14")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "dinov2_vits14_dense_448.onnx"
    fp16_onnx_path = args.output_dir / "dinov2_vits14_dense_448.fp16.onnx"
    engine_path = args.output_dir / "dinov2_vits14_dense_448.engine"
    if not fp16_onnx_path.exists() or not engine_path.exists():
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", source="github"
        ).eval().cuda().half()
        wrapper = DenseDino(model, args.input_size // 14).eval().cuda().half()
        sample = torch.randn(
            1, 3, args.input_size, args.input_size,
            device="cuda", dtype=torch.float16,
        )
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, sample, fp16_onnx_path, opset_version=17,
                input_names=["images"], output_names=["features"],
                do_constant_folding=True,
            )
    command = [
        "trtexec", f"--onnx={fp16_onnx_path}", f"--saveEngine={engine_path}",
        "--fp16", f"--memPoolSize=workspace:{args.workspace_mib}MiB",
    ]
    executable = shutil.which("trtexec")
    if executable:
        command[0] = executable
        subprocess.run(command, check=True)
    else:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        explicit = getattr(
            trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None
        )
        flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(fp16_onnx_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(i))
                               for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20
        )
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        engine_path.write_bytes(bytes(serialized))
    metadata = {
        "name": "dinov2_vits14_dense", "engine_path": engine_path.name,
        "onnx_path": fp16_onnx_path.name, "input_height": args.input_size,
        "input_width": args.input_size, "input_name": "images",
        "output_name": "features", "feature_shape": [384, 32, 32],
        "precision": "fp16",
    }
    (args.output_dir / "dinov2_vits14_dense_448.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(engine_path)


if __name__ == "__main__":
    main()
