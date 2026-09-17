# Model weights

## PP-OCRv5 TensorRT

`ocr_onnx/` contains the exact mobile detector and English recogniser as ONNX
and TensorRT files. Rebuild the hardware-specific engines with:

```bash
python3 tools/export_ocr_engines.py --onnx-dir weights/ocr_onnx
```

Parity and latency: `test_results/2026-08-22_tensorrt_ocr_report.md`.

Runtime model binaries are intentionally not stored in the Git repository.
TensorRT engines are hardware/runtime-specific, and the current scene engine
also exceeds GitHub's normal per-file limit.

Place these files in this directory before running HuMemSLAM:

| File | SHA-256 of the currently tested local model |
|---|---|
| `resnet50_places365_embed.engine` | `49a1a7ff933df4327e22f62e7a954af1910cca6a24c16fa9f806b1601d0c478b` |
| `resnet50_places365_embed.onnx` | `4b580071b0a318c305b57258889050d3016269bd5339edd461482078c652efb6` |
| `yolo26n-seg.engine` | `96d46d253d7cb060164076f7ccf8db4ee452566e9a37eb3888260922daba8495` |
| `humanSLAM_YOLO_seg.pt` | `efdc5c61d87e78d5d346c2de60bd7264a7e6265fff97068172759d4b42338230` |
| `humanSLAM_YOLO_seg.engine` | `5b68c0685560e1d7d99117dc21c1ee0b73d5959e0f0aa90c6dca9bf6e59ea204` |

The ONNX file is an export/rebuild artefact and is not loaded during normal
runtime. Use `tools/build_places365_embedding_engine.py` to build a compatible
scene TensorRT engine on the target machine.

`humanSLAM_YOLO_seg.pt` is the frozen best checkpoint from the 75-epoch
Mapillary Vistas fine-tuning run completed on 4 August 2026. Before making it
the default HuMemSLAM runtime model, update the stable/OCR class filters to its
underscore-separated class names and export a TensorRT engine on the target
machine for the latency-sensitive pipeline.

`humanSLAM_YOLO_seg.engine` is its fixed-batch-1, 640-pixel, FP16 TensorRT 11
export for the local RTX 4070 Laptop GPU. `humanSLAM_YOLO_seg.engine.json`
records the exact export environment. HuMemSLAM must construct this engine with
`task="segment"`; TensorRT filenames do not let Ultralytics infer that task.

Model licences and redistribution terms remain those of their respective model
authors. Do not publish third-party weights without checking those terms.

## Global VPR descriptors

`global_descriptors/` contains runtime specifications for EigenPlaces, MixVPR,
SALAD and Places365. The deployed HuMemSLAM configuration uses EigenPlaces for
global retrieval. Local TensorRT/ONNX binaries are deliberately excluded from
Git; rebuild a compatible engine on the target machine with
`tools/export_global_descriptor_engine.py`.
