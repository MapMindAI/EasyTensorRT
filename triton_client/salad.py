import argparse
import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tritonclient.grpc as grpcclient

try:
    from .logging_utils import configure_logging, get_logger
except ImportError:
    from logging_utils import configure_logging, get_logger


logger = get_logger(__name__)


def find_triton_model(client, model_key):
    model_index = client.get_model_repository_index()
    for model in model_index.models:
        if model_key in model.name:
            return model.name
    return None


class SaladClient:
    def __init__(
        self,
        triton_url="0.0.0.0:8001",
        model_key="salad",
        model_version="1",
        input_size=322,
        use_imagenet_norm=True,
    ):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_name = find_triton_model(self.grpc_client, model_key)
        if self.model_name is None:
            raise ValueError(f"Cannot find Triton model containing key: {model_key}")

        self.model_version = model_version
        self.input_size = input_size
        self.use_imagenet_norm = use_imagenet_norm
        self.input_name = "input"
        self.output_name = "descriptor"
        logger.info("Start %s from %s", self.model_name, triton_url)

    def _preprocess(self, image_bgr):
        if image_bgr is None:
            raise ValueError("Input image is None")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        crop = min(h, w)
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        rgb = rgb[y0 : y0 + crop, x0 : x0 + crop]

        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0

        if self.use_imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, axis=0).astype(np.float16)  # [1,3,H,W]
        return x

    def run(self, image_bgr):
        x = self._preprocess(image_bgr)

        infer_input = grpcclient.InferInput(self.input_name, x.shape, "FP16")
        infer_input.set_data_from_numpy(x)

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[grpcclient.InferRequestedOutput(self.output_name)],
        )

        descriptor = response.as_numpy(self.output_name)
        if descriptor is None:
            raise RuntimeError(f"Triton returned no '{self.output_name}' output")
        return descriptor


def parse_args():
    parser = argparse.ArgumentParser(description="SALAD Triton client test")
    parser.add_argument("--triton-url", default="0.0.0.0:8001", help="Triton gRPC endpoint")
    parser.add_argument("--model-key", default="salad", help="Substring used to find model name")
    parser.add_argument("--model-version", default="1", help="Triton model version")
    parser.add_argument("--image-dir", default="assets", help="Input image folder")
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png,bmp,webp",
        help="Comma-separated image extensions to include",
    )
    parser.add_argument("--output-descriptors", default="data/salad_descriptors.npy", help="Output descriptors .npy path")
    parser.add_argument("--output-matrix", default="data/salad_distance_matrix.npy", help="Output distance matrix .npy path")
    parser.add_argument("--output-matrix-csv", default="data/salad_distance_matrix.csv", help="Output distance matrix .csv path")
    parser.add_argument("--output-matrix-png", default="data/salad_distance_matrix.png", help="Output distance matrix .png path")
    parser.add_argument("--output-image-list", default="data/salad_image_list.txt", help="Output image list for matrix index mapping")
    parser.add_argument(
        "--distance-metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Pairwise distance metric",
    )
    parser.add_argument(
        "--no-imagenet-norm",
        action="store_true",
        help="Disable ImageNet mean/std normalization",
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser.parse_args()


def find_images(image_dir, extensions_csv):
    exts = [e.strip().lower().lstrip(".") for e in extensions_csv.split(",") if e.strip()]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext.upper()}")))
    image_paths = sorted(set(image_paths))
    return image_paths


def pairwise_distances(x, metric="cosine"):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D descriptors [N,D], got {x.shape}")

    if metric == "euclidean":
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        return dist.astype(np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        x_norm = x / norms
        sim = x_norm @ x_norm.T
        sim = np.clip(sim, -1.0, 1.0)
        dist = 1.0 - sim
        return dist.astype(np.float32)

    raise ValueError(f"Unsupported distance metric: {metric}")


def save_matrix_png(matrix, labels, output_path):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
    if len(labels) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("labels length must match square matrix size")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title("Pairwise Distance Matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    configure_logging(level=args.log_level)

    image_paths = find_images(args.image_dir, args.extensions)
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in '{args.image_dir}' with extensions: {args.extensions}"
        )

    client = SaladClient(
        triton_url=args.triton_url,
        model_key=args.model_key,
        model_version=args.model_version,
        use_imagenet_norm=not args.no_imagenet_norm,
    )

    logger.info("Found %d images in %s", len(image_paths), args.image_dir)
    t0 = time.time() * 1000

    descriptors = []
    valid_paths = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Skip unreadable image: %s", image_path)
            continue
        desc = client.run(image)
        desc = np.asarray(desc, dtype=np.float32).reshape(-1)
        descriptors.append(desc)
        valid_paths.append(image_path)

    if len(descriptors) == 0:
        raise RuntimeError("All images failed to decode or infer")

    descriptors = np.stack(descriptors, axis=0)
    distance_matrix = pairwise_distances(descriptors, metric=args.distance_metric)
    t1 = time.time() * 1000

    np.save(args.output_descriptors, descriptors)
    np.save(args.output_matrix, distance_matrix)
    np.savetxt(args.output_matrix_csv, distance_matrix, delimiter=",", fmt="%.8f")
    label_names = [os.path.basename(p) for p in valid_paths]
    save_matrix_png(distance_matrix, label_names, args.output_matrix_png)
    with open(args.output_image_list, "w", encoding="utf-8") as f:
        for p in valid_paths:
            f.write(f"{p}\n")

    logger.info("Processed %d images, used %.3fms", len(valid_paths), t1 - t0)
    logger.info("descriptors shape: %s, dtype: %s", descriptors.shape, descriptors.dtype)
    logger.info(
        "distance matrix shape: %s, metric: %s",
        distance_matrix.shape,
        args.distance_metric,
    )
    logger.info("Saved descriptors to %s", args.output_descriptors)
    logger.info("Saved distance matrix to %s and %s", args.output_matrix, args.output_matrix_csv)
    logger.info("Saved distance matrix heatmap to %s", args.output_matrix_png)
    logger.info("Saved image index map to %s", args.output_image_list)
