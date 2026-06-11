"""Build a FAISS IVFFlat index from downloaded scVI embeddings.

For correlation distance, vectors are mean-centered and L2-normalized so that
inner product equals Pearson correlation. For euclidean, vectors are used as-is.

Index is trained on GPU (k-means) and stored on CPU.

Usage:
    python 01-build_faiss_index.py --metric correlation
    python 01-build_faiss_index.py --metric euclidean --nlist 16384
"""

import argparse
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import faiss
import numpy as np
import scanpy as sc
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="scvi_embeddings.h5ad")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to scvi_index_{metric}.faiss",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "correlation"],
        required=True,
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=40_000,
        help="Number of IVF clusters. Rule of thumb: sqrt(N) to 4*sqrt(N). "
        "For 100M vectors, 40_000 is a reasonable default.",
    )
    parser.add_argument(
        "--n-train-per-cluster",
        type=int,
        default=100,
        help="Training vectors per cluster. Min ~30, recommended ~100-256. "
        "Higher improves cluster quality at the cost of training time.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"scvi_index_{args.metric}.faiss"
    return args


def normalize_for_correlation(vectors: np.ndarray, chunk_size: int) -> None:
    """Mean-center and L2-normalize in-place for correlation distance."""
    n = len(vectors)
    for i in tqdm(range(0, n, chunk_size), desc="Normalizing"):
        chunk = vectors[i : i + chunk_size]
        chunk -= chunk.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        np.maximum(norms, 1e-10, out=norms)
        chunk /= norms


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    t0 = time.time()
    adata = sc.read_h5ad(args.embeddings)
    vectors = adata.obsm["scvi"]  # (np.float32)
    del adata  # free obs metadata from RAM
    n, d = vectors.shape
    print(f"Loaded {n:,} vectors of dimension {d} in {time.time() - t0:.1f}s")

    # Normalize in-place for correlation distance
    if args.metric == "correlation":
        print("Normalizing vectors for correlation distance...")
        t0 = time.time()
        normalize_for_correlation(vectors, args.chunk_size)
        print(f"Normalized in {time.time() - t0:.1f}s")
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlatIP(d)
    else:
        faiss_metric = faiss.METRIC_L2
        quantizer = faiss.IndexFlatL2(d)

    # Sample training vectors
    n_train = min(args.n_train_per_cluster * args.nlist, n)
    print(f"Sampling {n_train:,} training vectors ({args.n_train_per_cluster} per cluster)...")
    train_idx = np.random.choice(n, n_train, replace=False)
    train_vectors = vectors[train_idx]

    # Train on GPU
    print(f"Training IVFFlat index (nlist={args.nlist}) on GPU {args.gpu_device}...")
    t0 = time.time()
    index = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss_metric)
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, args.gpu_device, index)
    index_gpu.train(train_vectors)
    index = faiss.index_gpu_to_cpu(index_gpu)
    del train_vectors
    print(f"Training complete in {time.time() - t0:.1f}s")

    # Add all vectors in chunks
    print(f"Adding {n:,} vectors to index...")
    t0 = time.time()
    for i in tqdm(range(0, n, args.chunk_size), desc="Adding vectors"):
        index.add(vectors[i : i + args.chunk_size])
    print(f"Added all vectors in {time.time() - t0:.1f}s")
    assert index.ntotal == n, f"Expected {n} vectors, got {index.ntotal}"

    # Save
    print(f"Saving index to {args.output}...")
    faiss.write_index(index, args.output)
    print(f"Done. Index contains {index.ntotal:,} vectors.")


if __name__ == "__main__":
    main()
