import argparse
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

try:
    from .logging_utils import configure_logging, get_logger
except ImportError:
    from logging_utils import configure_logging, get_logger


logger = get_logger(__name__)

# ADE20K labels: 0 wall, 1 building, 2 sky, 3 floor, 4 tree, ...
ADE20K_SKY_CLASS = 2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_triton_model(client, model_key):
    model_index = client.get_model_repository_index()
    for model in model_index.models:
        if model_key in model.name:
            return model.name
    return None


class SegformerClient:
    """SegFormer-B0 semantic segmentation over ADE20K's 150 classes."""

    def __init__(
        self,
        triton_url="0.0.0.0:8001",
        model_key="segformer",
        model_version="1",
        input_size=512,
    ):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_name = find_triton_model(self.grpc_client, model_key)
        if self.model_name is None:
            raise ValueError(f"Cannot find Triton model containing key: {model_key}")

        self.model_version = model_version
        self.input_size = input_size
        self.input_name = "input"
        self.output_name = "logits"
        logger.info("Start %s from %s", self.model_name, triton_url)

    def _preprocess(self, image_bgr):
        if image_bgr is None:
            raise ValueError("Input image is None")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        x = resized.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(x, axis=0).astype(np.float32)  # [1,3,H,W]

    def run(self, image_bgr):
        """Per-pixel ADE20K class indices at the input image's resolution.

        The graph returns full-resolution logits, which are classified before
        being mapped back to the input image's dimensions.
        """
        height, width = image_bgr.shape[:2]
        x = self._preprocess(image_bgr)

        infer_input = grpcclient.InferInput(self.input_name, x.shape, "FP32")
        infer_input.set_data_from_numpy(x)

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[grpcclient.InferRequestedOutput(self.output_name)],
        )

        logits = response.as_numpy(self.output_name)
        if logits is None:
            raise RuntimeError(f"Triton returned no '{self.output_name}' output")
        classes = np.argmax(logits[0], axis=0).astype(np.uint8)
        return cv2.resize(classes, (width, height), interpolation=cv2.INTER_NEAREST)


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Triton client test")
    parser.add_argument("--triton-url", default="0.0.0.0:8001", help="Triton gRPC endpoint")
    parser.add_argument("--model-key", default="segformer", help="Substring used to find model name")
    parser.add_argument("--model-version", default="1", help="Triton model version")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Optional sky-mask PNG path")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(level=args.log_level)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    client = SegformerClient(
        triton_url=args.triton_url,
        model_key=args.model_key,
        model_version=args.model_version,
    )

    t0 = time.time() * 1000
    classes = client.run(image)
    t1 = time.time() * 1000

    sky = classes == ADE20K_SKY_CLASS
    logger.info("Inference used %.3fms", t1 - t0)
    logger.info("class map: %s %s", classes.shape, classes.dtype)
    logger.info("sky: %.1f%% of pixels", 100.0 * sky.mean())
    if args.output:
        cv2.imwrite(args.output, (sky * 255).astype(np.uint8))
        logger.info("Saved sky mask to %s", args.output)
