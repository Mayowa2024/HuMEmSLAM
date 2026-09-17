"""Dense DINOv2 TensorRT features and YOLO-mask landmark pooling."""

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
import tensorrt as trt


@dataclass(frozen=True)
class DenseDescriptorSpec:
    name: str
    engine_path: Path
    input_height: int = 448
    input_width: int = 448
    input_name: str = "images"
    output_name: str = "features"
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
            name=str(payload.get("name", "dinov2_vits14_dense")),
            engine_path=engine,
            input_height=int(payload.get("input_height", 448)),
            input_width=int(payload.get("input_width", 448)),
            input_name=str(payload.get("input_name", "images")),
            output_name=str(payload.get("output_name", "features")),
            mean=tuple(payload.get("mean", cls.mean)),
            std=tuple(payload.get("std", cls.std)),
        )


def preprocess(image, spec):
    if image is None or image.size == 0:
        raise ValueError("Dense descriptor image is empty")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (spec.input_width, spec.input_height),
                     interpolation=cv2.INTER_LINEAR)
    value = rgb.astype(np.float32) / 255.0
    value = (value - np.asarray(spec.mean, np.float32)) / np.asarray(
        spec.std, np.float32
    )
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None])


def l2_normalize(vector):
    vector = np.asarray(vector, np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


class TensorRTDenseDescriptor:
    """Fixed-shape TensorRT runtime returning a CHW spatial feature map."""

    def __init__(self, spec):
        self.spec = spec
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            spec.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"Could not load {spec.engine_path}")
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        if spec.input_name not in self.names or spec.output_name not in self.names:
            raise RuntimeError(f"Expected {spec.input_name}/{spec.output_name}; got {self.names}")
        self.buffers = {}
        self.stream = None

    def describe(self, image):
        import torch
        array = preprocess(image, self.spec)
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        input_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.spec.input_name), torch
        )
        input_tensor = self._buffer(
            self.spec.input_name, array.shape, input_dtype, torch
        )
        with torch.cuda.stream(self.stream):
            input_tensor.copy_(
                torch.from_numpy(array).to(dtype=input_dtype), non_blocking=True
            )
        self.context.set_tensor_address(
            self.spec.input_name, input_tensor.data_ptr()
        )
        output_shape = tuple(self.context.get_tensor_shape(self.spec.output_name))
        output_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.spec.output_name), torch
        )
        output = self._buffer(
            self.spec.output_name, output_shape, output_dtype, torch
        )
        self.context.set_tensor_address(self.spec.output_name, output.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("DINOv2 TensorRT execution returned false")
        self.stream.synchronize()
        value = output.float().cpu().numpy()
        if value.ndim != 4 or value.shape[0] != 1:
            raise RuntimeError(f"Expected BCHW dense features, got {value.shape}")
        return value[0]

    def _buffer(self, name, shape, dtype, torch):
        tensor = self.buffers.get(name)
        if tensor is None or tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            tensor = torch.empty(tuple(shape), dtype=dtype, device="cuda")
            self.buffers[name] = tensor
        return tensor

    @staticmethod
    def _torch_dtype(dtype, torch):
        mapping = {
            trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int8: torch.int8, trt.int32: torch.int32, trt.bool: torch.bool,
        }
        if dtype not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype {dtype}")
        return mapping[dtype]


def pool_mask_features(features, mask, mode="mask_central", centre_sigma=0.35):
    """Pool CHW features inside a resized mask with optional central bias."""
    features = np.asarray(features, np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected CHW features, got {features.shape}")
    _, height, width = features.shape
    resized = cv2.resize(mask.astype(np.float32), (width, height),
                         interpolation=cv2.INTER_AREA)
    weights = np.clip(resized, 0.0, 1.0)
    locations = np.argwhere(weights > 0.05)
    if not len(locations):
        return None
    centre_y, centre_x = np.average(
        locations, axis=0, weights=weights[locations[:, 0], locations[:, 1]]
    )
    if mode in {"centre_3x3", "centre_5x5"}:
        radius = 1 if mode == "centre_3x3" else 2
        local = np.zeros_like(weights)
        cy, cx = int(round(centre_y)), int(round(centre_x))
        local[max(0, cy-radius):min(height, cy+radius+1),
              max(0, cx-radius):min(width, cx+radius+1)] = 1.0
        weights *= local
    elif mode == "mask_central":
        yy, xx = np.mgrid[:height, :width]
        scale = max(1.0, centre_sigma * max(height, width))
        central = np.exp(-((xx-centre_x)**2 + (yy-centre_y)**2) / (2*scale**2))
        weights *= central.astype(np.float32)
    elif mode != "mask":
        raise ValueError(f"Unknown pooling mode: {mode}")
    total = float(weights.sum())
    if total <= 1e-8:
        return None
    descriptor = (features * weights[None]).sum(axis=(1, 2)) / total
    return l2_normalize(descriptor)
