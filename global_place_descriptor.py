"""TensorRT runtime and configuration for global place descriptors.

The runtime is deliberately independent of ROS so the exact same inference
path can be used by HuMemSLAM and by the controlled descriptor benchmark.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorrt as trt


@dataclass(frozen=True)
class DescriptorSpec:
    name: str
    engine_path: Path
    input_height: int
    input_width: int
    output_name: str = "descriptor"
    input_name: str = ""
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        engine = Path(payload["engine_path"])
        if not engine.is_absolute():
            engine = path.parent / engine
        return cls(
            name=str(payload["name"]),
            engine_path=engine,
            input_height=int(payload["input_height"]),
            input_width=int(payload["input_width"]),
            output_name=str(payload.get("output_name", "descriptor")),
            input_name=str(payload.get("input_name", "")),
            mean=tuple(payload.get("mean", cls.mean)),
            std=tuple(payload.get("std", cls.std)),
        )


def preprocess_descriptor_image(image, spec):
    """Convert a BGR OpenCV image to the model's normalised NCHW tensor."""
    if image is None or image.size == 0:
        raise ValueError("Descriptor image is empty")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb,
        (spec.input_width, spec.input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = rgb.astype(np.float32) / 255.0
    tensor = (tensor - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(
        spec.std, dtype=np.float32
    )
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Descriptor engine produced a zero/invalid vector")
    return vector / norm


class TensorRTGlobalDescriptor:
    """Fixed-batch TensorRT descriptor runtime using PyTorch CUDA buffers."""

    def __init__(self, spec, logger=None):
        self.spec = spec
        self.logger = logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        engine_bytes = spec.engine_path.read_bytes()
        self.runtime = trt.Runtime(trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {spec.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create descriptor execution context")
        self.names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        ]
        inputs = [name for name in self.names if self.engine.get_tensor_mode(name)
                  == trt.TensorIOMode.INPUT]
        outputs = [name for name in self.names if self.engine.get_tensor_mode(name)
                   == trt.TensorIOMode.OUTPUT]
        self.output_names = outputs
        self.input_name = spec.input_name or (inputs[0] if len(inputs) == 1 else "")
        if self.input_name not in inputs:
            raise RuntimeError(f"Descriptor input '{self.input_name}' not in {inputs}")
        self.output_name = spec.output_name
        if self.output_name not in outputs:
            if len(outputs) == 1 and spec.output_name == "descriptor":
                self.output_name = outputs[0]
            else:
                raise RuntimeError(
                    f"Descriptor output '{spec.output_name}' not in {outputs}"
                )
        self.device_tensors = {}
        self.stream = None

    def describe(self, image):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for CUDA buffer allocation") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")

        array = preprocess_descriptor_image(image, self.spec)
        engine_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if any(value < 0 for value in engine_shape):
            if not self.context.set_input_shape(self.input_name, tuple(array.shape)):
                raise RuntimeError(f"TensorRT rejected input shape {array.shape}")
        elif engine_shape != tuple(array.shape):
            raise RuntimeError(
                f"Engine expects {engine_shape}, preprocessor produced {array.shape}"
            )

        if self.stream is None:
            self.stream = torch.cuda.Stream()
        input_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name), torch
        )
        input_tensor = self._buffer(
            self.input_name, tuple(array.shape), input_dtype, torch
        )
        with torch.cuda.stream(self.stream):
            input_tensor.copy_(torch.from_numpy(array), non_blocking=True)
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())

        for name in self.output_names:
            output_shape = tuple(self.context.get_tensor_shape(name))
            if any(value < 0 for value in output_shape):
                raise RuntimeError(f"Unresolved output shape for {name}: {output_shape}")
            output_dtype = self._torch_dtype(
                self.engine.get_tensor_dtype(name), torch
            )
            output = self._buffer(name, output_shape, output_dtype, torch)
            self.context.set_tensor_address(name, output.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT descriptor execution returned false")
        self.stream.synchronize()
        selected = self.device_tensors[self.output_name]
        return l2_normalize(selected.float().cpu().numpy())

    def _buffer(self, name, shape, dtype, torch):
        tensor = self.device_tensors.get(name)
        if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
            self.device_tensors[name] = tensor
        return tensor

    @staticmethod
    def _torch_dtype(dtype, torch):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        if dtype not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")
        return mapping[dtype]
