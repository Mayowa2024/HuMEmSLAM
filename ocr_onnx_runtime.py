"""PP-OCRv5 mobile detection/recognition through ONNX Runtime GPU providers."""

from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
import yaml

from slam.types import TextAnchor


def _order_points(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _polygon_score(probability, points):
    height, width = probability.shape
    points = np.asarray(points, dtype=np.float32)
    x0 = max(0, int(np.floor(points[:, 0].min())))
    x1 = min(width - 1, int(np.ceil(points[:, 0].max())))
    y0 = max(0, int(np.floor(points[:, 1].min())))
    y1 = min(height - 1, int(np.ceil(points[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    shifted = points.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
    return float(cv2.mean(probability[y0:y1 + 1, x0:x1 + 1], mask)[0])


def _unclip(points, ratio=1.5):
    points = np.asarray(points, dtype=np.float32)
    area = abs(float(cv2.contourArea(points)))
    perimeter = float(cv2.arcLength(points, True))
    if area <= 0.0 or perimeter <= 0.0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(points.astype(np.int32).tolist(), pyclipper.JT_ROUND,
                   pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(area * float(ratio) / perimeter)
    if not paths:
        return None
    return np.asarray(max(paths, key=len), dtype=np.float32)


def _perspective_crop(image, points):
    points = _order_points(points)
    width = max(
        int(np.linalg.norm(points[0] - points[1])),
        int(np.linalg.norm(points[2] - points[3])),
    )
    height = max(
        int(np.linalg.norm(points[0] - points[3])),
        int(np.linalg.norm(points[1] - points[2])),
    )
    if width < 2 or height < 2:
        return None
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(
        image, transform, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


class OnnxPPOCRv5:
    """Small PP-OCR pipeline retaining PaddleOCR-compatible text anchors."""

    def __init__(
        self,
        detector_path,
        recognizer_path,
        recognizer_yaml,
        confidence_threshold=0.5,
        detector_side=640,
        providers=None,
    ):
        available = set(ort.get_available_providers())



        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False, directory="")
        requested = providers or [
            "TensorrtExecutionProvider", "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.providers = [provider for provider in requested if provider in available]
        if not self.providers:
            self.providers = ["CPUExecutionProvider"]
        provider_options = []
        for provider in self.providers:
            if provider == "TensorrtExecutionProvider":
                provider_options.append({
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(
                        Path(detector_path).resolve().parent / "trt_cache"
                    ),
                })
            else:
                provider_options.append({})
        self.detector = ort.InferenceSession(
            str(detector_path), providers=self.providers,
            provider_options=provider_options,
        )
        self.recognizer = ort.InferenceSession(
            str(recognizer_path), providers=self.providers,
            provider_options=provider_options,
        )
        config = yaml.safe_load(Path(recognizer_yaml).read_text())
        characters = config["PostProcess"]["character_dict"]
        self.characters = ["blank"] + [str(value) for value in characters]
        self.confidence_threshold = float(confidence_threshold)
        self.detector_side = max(32, int(detector_side))
        self.last_timings = {}

    @staticmethod
    def _normalise_detection(image, target_side):
        height, width = image.shape[:2]
        scale = min(1.0, float(target_side) / max(height, width))
        resized_height = max(32, int(round(height * scale / 32.0)) * 32)
        resized_width = max(32, int(round(width * scale / 32.0)) * 32)
        resized = cv2.resize(image, (resized_width, resized_height))
        value = resized.astype(np.float32) / 255.0
        value = (value - np.asarray([0.485, 0.456, 0.406], np.float32))
        value /= np.asarray([0.229, 0.224, 0.225], np.float32)
        return value.transpose(2, 0, 1)[None], (height, width)

    def _detect(self, image):
        tensor, original_shape = self._normalise_detection(
            image, self.detector_side
        )
        output = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})[0]
        probability = np.asarray(output)[0, 0]
        binary = (probability > 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1000]:
            points = cv2.boxPoints(cv2.minAreaRect(contour))
            if min(cv2.minAreaRect(contour)[1]) < 3:
                continue
            score = _polygon_score(probability, points)
            if score < 0.6:
                continue
            expanded = _unclip(points, 1.5)
            if expanded is None:
                continue
            points = cv2.boxPoints(cv2.minAreaRect(expanded))
            points[:, 0] *= original_shape[1] / probability.shape[1]
            points[:, 1] *= original_shape[0] / probability.shape[0]
            boxes.append((_order_points(points), score))
        boxes.sort(key=lambda item: (float(item[0][:, 1].mean()),
                                     float(item[0][:, 0].mean())))
        return boxes

    @staticmethod
    def _recognition_tensor(crops):
        maximum_ratio = max(crop.shape[1] / max(1, crop.shape[0]) for crop in crops)






        width = min(640, max(320, int(np.ceil(48 * maximum_ratio / 8.0)) * 8))
        batch = np.zeros((len(crops), 3, 48, width), dtype=np.float32)
        for index, crop in enumerate(crops):
            resized_width = min(width, max(1, int(np.ceil(48 * crop.shape[1] /
                                                          max(1, crop.shape[0])))))
            resized = cv2.resize(crop, (resized_width, 48)).astype(np.float32)
            resized = resized.transpose(2, 0, 1) / 255.0
            batch[index, :, :, :resized_width] = (resized - 0.5) / 0.5
        return batch

    def _recognise(self, crops):
        if not crops:
            return []
        tensor = self._recognition_tensor(crops)
        predictions = self.recognizer.run(
            None, {self.recognizer.get_inputs()[0].name: tensor}
        )[0]
        results = []
        for prediction in predictions:
            indices = prediction.argmax(axis=1)
            scores = prediction.max(axis=1)
            selected, confidences = [], []
            previous = -1
            for index, score in zip(indices, scores):
                index = int(index)
                if index != 0 and index != previous and index < len(self.characters):
                    selected.append(self.characters[index])
                    confidences.append(float(score))
                previous = index
            results.append(("".join(selected), float(np.mean(confidences))
                            if confidences else 0.0))
        return results

    def predict_anchors(self, image):
        started = time.perf_counter()
        boxes = self._detect(image)
        detected = time.perf_counter()
        crops, kept = [], []
        for box, _ in boxes:
            crop = _perspective_crop(image, box)
            if crop is not None:
                crops.append(crop)
                kept.append(box)
        recognised = self._recognise(crops)
        finished = time.perf_counter()
        self.last_timings = {
            "detection_ms": (detected - started) * 1000.0,
            "recognition_ms": (finished - detected) * 1000.0,
            "total_ms": (finished - started) * 1000.0,
        }
        return [
            TextAnchor(text=text, conf=confidence)
            for text, confidence in recognised
            if text and confidence >= self.confidence_threshold
        ]

    def predict(self, inputs):
        images = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        return [self.predict_anchors(image) for image in images]


class _TensorRTEngine:
    """Small dynamic-shape TensorRT runner backed by reusable CUDA tensors."""

    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT OCR engine {path}")
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in self.names if
                               self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_names = [name for name in self.names if
                             self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        self.buffers = {}
        self.stream = None

    def run(self, array):
        import torch
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        if not self.context.set_input_shape(self.input_name, tuple(array.shape)):
            raise RuntimeError(f"OCR engine rejected shape {array.shape}")
        input_tensor = self._buffer(
            self.input_name, array.shape,
            self._dtype(self.engine.get_tensor_dtype(self.input_name), torch), torch,
        )
        with torch.cuda.stream(self.stream):
            input_tensor.copy_(torch.from_numpy(array), non_blocking=True)
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            output = self._buffer(
                name, shape, self._dtype(self.engine.get_tensor_dtype(name), torch), torch
            )
            self.context.set_tensor_address(name, output.data_ptr())
            outputs.append(output)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT OCR execution returned false")
        self.stream.synchronize()
        return [output.float().cpu().numpy() for output in outputs]

    def _buffer(self, name, shape, dtype, torch):
        value = self.buffers.get(name)
        if value is None or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            value = torch.empty(tuple(shape), dtype=dtype, device="cuda")
            self.buffers[name] = value
        return value

    @staticmethod
    def _dtype(dtype, torch):
        import tensorrt as trt
        mapping = {trt.float32: torch.float32, trt.float16: torch.float16,
                   trt.int32: torch.int32, trt.int8: torch.int8, trt.bool: torch.bool}
        if dtype not in mapping:
            raise RuntimeError(f"Unsupported OCR engine dtype {dtype}")
        return mapping[dtype]


class TensorRTPPOCRv5(OnnxPPOCRv5):
    """Same OCR preprocessing/postprocessing with prebuilt TensorRT engines."""

    def __init__(self, detector_path, recognizer_path, recognizer_yaml,
                 confidence_threshold=0.5, detector_side=640):
        config = yaml.safe_load(Path(recognizer_yaml).read_text())
        self.characters = ["blank"] + [
            str(value) for value in config["PostProcess"]["character_dict"]
        ]
        self.confidence_threshold = float(confidence_threshold)
        self.detector_side = max(32, min(640, int(detector_side)))
        self.detector = _TensorRTEngine(detector_path)
        self.recognizer = _TensorRTEngine(recognizer_path)
        self.providers = ["TensorRT"]
        self.last_timings = {}

    def _detect(self, image):
        tensor, original_shape = self._normalise_detection(image, self.detector_side)
        output = self.detector.run(tensor)[0]
        return self._postprocess_detection(output, original_shape)

    def _postprocess_detection(self, output, original_shape):
        probability = np.asarray(output)[0, 0]
        binary = (probability > 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1000]:
            rectangle = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rectangle)
            if min(rectangle[1]) < 3 or _polygon_score(probability, points) < 0.6:
                continue
            expanded = _unclip(points, 1.5)
            if expanded is None:
                continue
            points = cv2.boxPoints(cv2.minAreaRect(expanded))
            points[:, 0] *= original_shape[1] / probability.shape[1]
            points[:, 1] *= original_shape[0] / probability.shape[0]
            boxes.append((_order_points(points), 1.0))
        boxes.sort(key=lambda item: (float(item[0][:, 1].mean()),
                                     float(item[0][:, 0].mean())))
        return boxes

    def _recognise(self, crops):
        if not crops:
            return []
        tensor = self._recognition_tensor(crops)

        if tensor.shape[0] > 18:
            groups = [crops[index:index + 18] for index in range(0, len(crops), 18)]
            return [result for group in groups for result in self._recognise(group)]
        predictions = self.recognizer.run(tensor)[0]
        results = []
        for prediction in predictions:
            indices, scores = prediction.argmax(axis=1), prediction.max(axis=1)
            selected, confidences, previous = [], [], -1
            for index, score in zip(indices, scores):
                index = int(index)
                if index and index != previous and index < len(self.characters):
                    selected.append(self.characters[index])
                    confidences.append(float(score))
                previous = index
            results.append(("".join(selected), float(np.mean(confidences))
                            if confidences else 0.0))
        return results
