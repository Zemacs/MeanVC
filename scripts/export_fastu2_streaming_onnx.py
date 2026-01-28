import argparse
import os
from pathlib import Path

import torch
import torch.onnx


def _register_isinstance_symbolic(opset: int) -> None:
    # Force prim::isinstance to True so TorchScript graph takes the int offset path.
    def prim_isinstance(g, _input, _types=None, **_kwargs):
        return g.op("Constant", value_t=torch.tensor(True))

    try:
        torch.onnx.register_custom_op_symbolic("prim::isinstance", prim_isinstance, opset)
    except Exception:
        # If already registered in the process, ignore.
        pass


class StreamingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xs, offset, required_cache_size, att_cache, cnn_cache):
        return self.model.forward_encoder_chunk(
            xs, offset, required_cache_size, att_cache, cnn_cache
        )


def export_onnx(
    pt_path: Path,
    out_path: Path,
    opset: int,
    example_t: int,
    cache_t: int,
) -> None:
    _register_isinstance_symbolic(opset)

    model = torch.jit.load(str(pt_path), map_location="cpu")
    model.eval()

    wrapper = StreamingWrapper(model)
    wrapper.eval()

    # Example inputs: correct ranks with cache_t possibly 0 for first chunk.
    xs = torch.randn(1, example_t, 80)
    offset = torch.tensor(0, dtype=torch.int64)
    required_cache_size = torch.tensor(8, dtype=torch.int64)
    att_cache = torch.zeros((7, 4, cache_t, 128), dtype=torch.float32)
    cnn_cache = torch.zeros((7, 1, 256, cache_t), dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            torch.jit.script(wrapper),
            (xs, offset, required_cache_size, att_cache, cnn_cache),
            str(out_path),
            input_names=[
                "xs",
                "offset",
                "required_cache_size",
                "att_cache",
                "cnn_cache",
            ],
            output_names=["encoder_output", "att_cache_out", "cnn_cache_out"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={
                "xs": {0: "batch", 1: "time"},
                "att_cache": {2: "cache_t"},
                "cnn_cache": {3: "cnn_cache_t"},
                "encoder_output": {0: "batch", 1: "time_out"},
                "att_cache_out": {2: "cache_t_out"},
                "cnn_cache_out": {3: "cnn_cache_t_out"},
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export streaming ONNX for fastu2++ TorchScript."
    )
    parser.add_argument(
        "--pt",
        default="src/ckpt/fastu2++.pt",
        help="Path to TorchScript .pt file.",
    )
    parser.add_argument(
        "--out",
        default="src/ckpt/fastu2++_encoder_chunk_streaming.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--example-t",
        type=int,
        default=19,
        help="Example time length for xs input.",
    )
    parser.add_argument(
        "--cache-t",
        type=int,
        default=0,
        help="Example cache length for att/conv cache.",
    )

    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)

    if not pt_path.exists():
        raise FileNotFoundError(f"TorchScript not found: {pt_path}")

    export_onnx(
        pt_path=pt_path,
        out_path=out_path,
        opset=args.opset,
        example_t=args.example_t,
        cache_t=args.cache_t,
    )

    print(str(out_path))


if __name__ == "__main__":
    main()
