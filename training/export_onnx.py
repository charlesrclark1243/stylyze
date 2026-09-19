import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from modules.model import StyleTransferModel, read_checkpoint
from torch import nn
from torch.export import Dim

torch.set_num_threads(2)


class ExportWrapper(nn.Module):
    def __init__(self, model: StyleTransferModel):
        super().__init__()
        self.model = model

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        return self.model(content, style, alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Style Transfer Model to ONNX format."
    )

    parser.add_argument(
        "--checkpoint",
        "-i",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save the exported ONNX model.",
    )

    parser.add_argument(
        "--validate",
        "-V",
        action="store_true",
        help="Validate the exported ONNX model after export.",
    )
    parser.add_argument(
        "--benchmark",
        "-B",
        action="store_true",
        help="Run a benchmark on the exported ONNX model after export.",
    )

    return parser.parse_args()


def export_model(checkpoint_path: str, output_path: str):
    model = StyleTransferModel()
    model.load_state_dict(
        read_checkpoint(torch.load(checkpoint_path, map_location="cpu"))
    )
    model.eval()

    content = torch.randn(1, 3, 512, 640)
    style = torch.randn(1, 3, 512, 512)  # deliberately different size
    alpha = torch.tensor(1.0)

    ch, cw = Dim("ch", min=64, max=4096), Dim("cw", min=64, max=4096)
    sh, sw = Dim("sh", min=64, max=4096), Dim("sw", min=64, max=4096)

    wrapper = ExportWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (content, style, alpha),
        output_path,
        input_names=["content", "style", "alpha"],
        output_names=["stylyzed"],
        dynamic_shapes=(
            {2: ch, 3: cw},  # content
            {2: sh, 3: sw},  # style
            None,
        ),
        opset_version=18,
        do_constant_folding=True,
    )

    m = onnx.load(output_path)
    onnx.save_model(m, output_path, save_as_external_data=False)


def validate_onnx(onnx_path: str, model: StyleTransferModel) -> bool:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2

    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    c = torch.rand(1, 3, 384, 512)
    s = torch.rand(1, 3, 600, 400)
    alpha = 0.3

    with torch.no_grad():
        ref = model(c, s, alpha)

    got = sess.run(
        None,
        {
            "content": c.numpy(),
            "style": s.numpy(),
            "alpha": np.array(alpha, dtype=np.float32),
        },
    )[0]
    return np.abs(got - ref.detach().numpy()).max() < 1e-4


def benchmark_onnx(onnx_path: str, model: StyleTransferModel) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2

    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    c = torch.rand(1, 3, 384, 512)
    s = torch.rand(1, 3, 600, 400)
    alpha = 0.3

    results = {}
    for name, fn in [
        ("torch", lambda: model(c, s, alpha)),
        (
            "onnx",
            lambda: sess.run(
                None,
                {
                    "content": c.numpy(),
                    "style": s.numpy(),
                    "alpha": np.array(alpha, dtype=np.float32),
                },
            ),
        ),
    ]:
        fn()  # warm up

        t = time.perf_counter()
        for _ in range(10):
            fn()

        results[name] = (time.perf_counter() - t) / 10

    return results


def load_model(checkpoint_path: str) -> StyleTransferModel:
    model = StyleTransferModel()
    model.load_state_dict(
        read_checkpoint(torch.load(checkpoint_path, map_location="cpu"))
    )
    return model.eval()


def main():
    args = parse_args()
    export_model(args.checkpoint, args.output)
    print(f"Model exported to {args.output}")

    if args.validate:
        same = validate_onnx(
            args.output,
            load_model(args.checkpoint),
        )

        print("Validation result:", "Success" if same else "Failed")

    if args.benchmark:
        results = benchmark_onnx(
            args.output,
            load_model(args.checkpoint),
        )

        for name, t in results.items():
            print(f"{name} average time: {t:.4f} seconds")


if __name__ == "__main__":
    main()
