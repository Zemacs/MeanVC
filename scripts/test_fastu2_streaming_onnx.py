import argparse
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort


def run_test(
    pt_path: Path,
    onnx_path: Path,
    seed: int,
    num_cases: int,
    min_len: int,
    max_len: int,
    chunk_size: int,
    stride: int,
    required_cache_size: int,
) -> dict:
    rng = np.random.default_rng(seed)

    model = torch.jit.load(str(pt_path), map_location="cpu")
    model.eval()

    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    stats = {
        "out_max_abs": [],
        "att_max_abs": [],
        "cnn_max_abs": [],
        "out_mean_abs": [],
        "att_mean_abs": [],
        "cnn_mean_abs": [],
    }

    for _ in range(num_cases):
        total_len = int(rng.integers(min_len, max_len + 1))
        fbanks = rng.standard_normal((1, total_len, 80), dtype=np.float32)

        # zero-length caches, correct ranks
        att_cache_t = np.zeros((7, 4, 0, 128), dtype=np.float32)
        cnn_cache_t = np.zeros((7, 1, 256, 0), dtype=np.float32)
        att_cache_o = np.zeros((7, 4, 0, 128), dtype=np.float32)
        cnn_cache_o = np.zeros((7, 1, 256, 0), dtype=np.float32)

        offset_t = 0
        offset_o = 0

        for i in range(0, fbanks.shape[1], stride):
            chunk = fbanks[:, i : i + chunk_size, :]
            if chunk.shape[1] < 10:
                break

            with torch.no_grad():
                out_t, att_t2, cnn_t2 = model.forward_encoder_chunk(
                    torch.from_numpy(chunk),
                    offset_t,
                    required_cache_size,
                    torch.from_numpy(att_cache_t),
                    torch.from_numpy(cnn_cache_t),
                )

            out_t = out_t.cpu().numpy()
            att_t2 = att_t2.cpu().numpy()
            cnn_t2 = cnn_t2.cpu().numpy()

            out_o, att_o2, cnn_o2 = ort_sess.run(
                None,
                {
                    "xs": chunk,
                    "offset": np.array(offset_o, dtype=np.int64),
                    "required_cache_size": np.array(required_cache_size, dtype=np.int64),
                    "att_cache": att_cache_o,
                    "cnn_cache": cnn_cache_o,
                },
            )

            def _record(a, b, key_max, key_mean):
                diff = a - b
                stats[key_max].append(float(np.max(np.abs(diff))))
                stats[key_mean].append(float(np.mean(np.abs(diff))))

            _record(out_t, out_o, "out_max_abs", "out_mean_abs")
            _record(att_t2, att_o2, "att_max_abs", "att_mean_abs")
            _record(cnn_t2, cnn_o2, "cnn_max_abs", "cnn_mean_abs")

            att_cache_t = att_t2
            cnn_cache_t = cnn_t2
            att_cache_o = att_o2
            cnn_cache_o = cnn_o2

            offset_t += out_t.shape[1]
            offset_o += out_o.shape[1]

    return {
        "cases": num_cases,
        "steps": len(stats["out_max_abs"]),
        "max_abs_out": max(stats["out_max_abs"], default=0.0),
        "max_abs_att": max(stats["att_max_abs"], default=0.0),
        "max_abs_cnn": max(stats["cnn_max_abs"], default=0.0),
        "mean_abs_out": float(np.mean(stats["out_mean_abs"])) if stats["out_mean_abs"] else 0.0,
        "mean_abs_att": float(np.mean(stats["att_mean_abs"])) if stats["att_mean_abs"] else 0.0,
        "mean_abs_cnn": float(np.mean(stats["cnn_mean_abs"])) if stats["cnn_mean_abs"] else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare streaming ONNX and TorchScript outputs."
    )
    parser.add_argument(
        "--pt", default="src/ckpt/fastu2++.pt", help="TorchScript .pt path"
    )
    parser.add_argument(
        "--onnx",
        default="src/ckpt/fastu2++_encoder_chunk_streaming.onnx",
        help="ONNX path",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=80)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=19)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--required-cache-size", type=int, default=8)

    args = parser.parse_args()

    pt_path = Path(args.pt)
    onnx_path = Path(args.onnx)

    if not pt_path.exists():
        raise FileNotFoundError(f"TorchScript not found: {pt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    summary = run_test(
        pt_path=pt_path,
        onnx_path=onnx_path,
        seed=args.seed,
        num_cases=args.cases,
        min_len=args.min_len,
        max_len=args.max_len,
        chunk_size=args.chunk_size,
        stride=args.stride,
        required_cache_size=args.required_cache_size,
    )

    print("cases", summary["cases"])
    print("steps", summary["steps"])
    print("max_abs_out", summary["max_abs_out"])
    print("max_abs_att", summary["max_abs_att"])
    print("max_abs_cnn", summary["max_abs_cnn"])
    print("mean_abs_out", summary["mean_abs_out"])
    print("mean_abs_att", summary["mean_abs_att"])
    print("mean_abs_cnn", summary["mean_abs_cnn"])


if __name__ == "__main__":
    main()
