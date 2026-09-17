#!/usr/bin/env python3
"""Export supported pretrained VPR models to ONNX and TensorRT FP16.

TensorRT engines must be built on the deployment computer. EigenPlaces and
SALAD are fetched through their official Torch Hub entry points. MixVPR uses
the official repository plus a separately downloaded official checkpoint.
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import onnx
import tensorrt as trt
import torch
import torch.nn as nn
import torchvision


class DescriptorOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        descriptor = self.model(images)
        if isinstance(descriptor, (tuple, list)):
            descriptor = descriptor[0]
        return torch.nn.functional.normalize(descriptor.flatten(1), dim=1)


class FeatureMixerLayer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension),
            nn.ReLU(), nn.Linear(dimension, dimension),
        )

    def forward(self, value):
        return value + self.mix(value)


class MixVPRAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Sequential(*[FeatureMixerLayer(400) for _ in range(4)])
        self.channel_proj = nn.Linear(1024, 256)
        self.row_proj = nn.Linear(400, 2)

    def forward(self, value):
        value = self.mix(value.flatten(2)).permute(0, 2, 1)
        value = self.channel_proj(value).permute(0, 2, 1)
        return torch.nn.functional.normalize(self.row_proj(value).flatten(1), dim=-1)


class CroppedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)
        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None

    def forward(self, value):
        model = self.model
        value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
        value = model.layer1(value)
        value = model.layer2(value)
        return model.layer3(value)


class MixVPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CroppedResNet50()
        self.aggregator = MixVPRAggregator()

    def forward(self, value):
        return self.aggregator(self.backbone(value))


def load_model(args):
    if args.model == "eigenplaces_r18_512":
        return torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model",
            backbone="ResNet18", fc_output_dim=512,
        )
    if args.model == "salad":



        if "pytorch_lightning" not in sys.modules:
            lightning = types.ModuleType("pytorch_lightning")
            class InferenceLightningModule(nn.Module):
                def save_hyperparameters(self, *unused_args, **unused_kwargs):
                    return None
            lightning.LightningModule = InferenceLightningModule
            sys.modules["pytorch_lightning"] = lightning
        if "utils" not in sys.modules:
            training_utils = types.ModuleType("utils")
            training_utils.get_loss = lambda *unused_args, **unused_kwargs: None
            training_utils.get_miner = lambda *unused_args, **unused_kwargs: None
            sys.modules["utils"] = training_utils
        return torch.hub.load("serizba/salad", "dinov2_salad")
    if args.model == "mixvpr_r50_512":
        if not args.checkpoint:
            raise ValueError("MixVPR requires its official --checkpoint")
        model = MixVPRModel()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("state_dict", state), strict=True)
        return model
    raise ValueError(args.model)


def build_engine(onnx_path, engine_path, workspace_gib):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)


    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
    )
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parser failed:\n{errors}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(serialized))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "eigenplaces_r18_512", "mixvpr_r50_512", "salad"
    ])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    args = parser.parse_args()
    default_sizes = {
        "eigenplaces_r18_512": 320,
        "mixvpr_r50_512": 320,
        "salad": 322,
    }
    size = args.input_size or default_sizes[args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / f"{args.model}.onnx"
    engine_path = args.output_dir / f"{args.model}.engine"
    spec_path = args.output_dir / f"{args.model}.json"

    model = DescriptorOutput(load_model(args).eval()).cpu()
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model, dummy, onnx_path, input_names=["images"],
        output_names=["descriptor"], opset_version=args.opset,
        dynamic_axes=None, do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    build_engine(onnx_path, engine_path, args.workspace_gib)
    metadata = {
        "name": args.model,
        "engine_path": engine_path.name,
        "onnx_path": onnx_path.name,
        "input_height": size,
        "input_width": size,
        "input_name": "images",
        "output_name": "descriptor",
        "normalization": "ImageNet mean/std; output L2 normalised",
        "precision": "TensorRT FP16-eligible",
    }
    spec_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"ONNX: {onnx_path}\nTensorRT: {engine_path}\nSpec: {spec_path}")


if __name__ == "__main__":
    main()
