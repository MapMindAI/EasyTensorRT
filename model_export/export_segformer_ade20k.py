"""Export SegFormer-B0 (ADE20K) to ONNX for the `segformer_onnx` Triton model.

The client does the preprocessing (BGR->RGB, 512x512, ImageNet mean/std), so the
graph takes an already-normalized NCHW float32 tensor and returns the 150-class
ADE20K logits at stride 4. Sky is class index 2.

Usage:
    python model_export/export_segformer_ade20k.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "model_repository"
    / "segformer_onnx"
    / "1"
    / "segformer_b0_ade20k_512x512.onnx"
)


class SegformerLogits(torch.nn.Module):
    """Wraps the HF model so the graph output is the raw class logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id)
    model.eval()
    wrapper = SegformerLogits(model).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(args.output),
            input_names=["input"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamo=False,
        )
    print(f"Wrote {args.output}")

    import onnxruntime as ort

    with torch.inference_mode():
        reference = wrapper(dummy).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    exported = session.run(None, {"input": dummy.numpy()})[0]
    print("torch", reference.shape, "onnx", exported.shape)
    print("max abs diff:", float(np.abs(reference - exported).max()))


if __name__ == "__main__":
    main()
