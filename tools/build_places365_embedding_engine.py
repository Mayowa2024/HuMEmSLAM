import argparse
from pathlib import Path

import onnx
import tensorrt as trt
from onnx import TensorProto, helper, shape_inference


def expose_embedding(source_path, output_path):
    model = onnx.load(str(source_path))
    flatten_node = next(
        (node for node in model.graph.node if node.op_type == "Flatten"),
        None,
    )
    if flatten_node is None:
        raise RuntimeError("Could not find the ResNet flatten/embedding node")

    old_name = flatten_node.output[0]
    new_name = "embedding"
    flatten_node.output[0] = new_name
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name

    if not any(output.name == new_name for output in model.graph.output):
        model.graph.output.append(
            helper.make_tensor_value_info(
                new_name,
                TensorProto.FLOAT,
                [1, 2048],
            )
        )

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)


    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(serialized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-onnx", required=True, type=Path)
    parser.add_argument("--output-onnx", required=True, type=Path)
    parser.add_argument("--output-engine", required=True, type=Path)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    args.output_engine.parent.mkdir(parents=True, exist_ok=True)
    expose_embedding(args.source_onnx, args.output_onnx)
    build_engine(args.output_onnx, args.output_engine, fp16=not args.fp32)
    print(f"Embedding ONNX: {args.output_onnx}")
    print(f"TensorRT engine: {args.output_engine}")


if __name__ == "__main__":
    main()
