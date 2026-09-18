#!/usr/bin/env python3
"""Generate deterministic compatibility fixtures using pinned upstream hdbscan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any

import hdbscan
import numpy as np
from hdbscan import BranchDetector
from importlib.metadata import version
from scipy.sparse import csr_matrix, issparse


SCHEMA_VERSION = 1
UPSTREAM_COMMIT = "dfdc9ca2b265ab6d50cf64b428199785700af240"
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "testdata" / "parity"


@dataclass(frozen=True)
class Case:
    name: str
    data: Any
    config: dict[str, Any]
    description: str


def blobs(seed: int, rows: int, dims: int, centers: int, spread: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = rng.uniform(-8.0, 8.0, size=(centers, dims))
    assignments = np.arange(rows) % centers
    return means[assignments] + rng.normal(0.0, spread, size=(rows, dims))


def cases() -> list[Case]:
    duplicates = np.array(
        [[0, 0], [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [5, 5], [5, 5], [6, 5], [5, 6]],
        dtype=np.float64,
    )
    all_noise = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    one_cluster = blobs(7, 18, 2, 1, 0.15)
    nonfinite = np.array([[0, 0], [0.1, 0], [0, 0.1], [math.nan, 2], [math.inf, 3], [5, 5], [5.1, 5], [5, 5.1]], dtype=np.float64)
    disconnected = csr_matrix(np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=np.float64,
    ))
    sparse_connected = csr_matrix(np.array(
        [[0, 1, 4, 5], [1, 0, 3, 4], [4, 3, 0, 1], [5, 4, 1, 0]], dtype=np.float64,
    ))
    common = {"algorithm": "generic", "approx_min_span_tree": False, "gen_min_span_tree": True}

    # A deterministic three-flare example following the published FLASC shape.
    branch_shapes = np.array([
        (radius * np.cos(arm * 2 * np.pi / 3), radius * np.sin(arm * 2 * np.pi / 3))
        for arm in range(3) for radius in (0.15 + np.arange(12) * 0.18)
    ]) + np.random.default_rng(8128).normal(0, 0.005, size=(36, 2))
    return [
        Case("empty", np.empty((0, 2)), {**common}, "Empty dense input; records upstream validation."),
        Case("singleton", np.array([[0.0, 0.0]]), {**common}, "Single point; records upstream validation."),
        Case("duplicates_ties_eom", duplicates, {**common, "min_cluster_size": 2, "min_samples": 2}, "Duplicates and equal-distance ties."),
        Case("duplicates_ties_leaf", duplicates, {**common, "min_cluster_size": 2, "min_samples": 2, "cluster_selection_method": "leaf"}, "Tie-heavy leaf selection."),
        Case("all_noise", all_noise, {**common, "min_cluster_size": 3, "min_samples": 3}, "Small separated dataset expected to be noise."),
        Case("allow_single_cluster", one_cluster, {**common, "min_cluster_size": 5, "allow_single_cluster": True}, "Explicit single-cluster selection."),
        Case("nonfinite_rows", nonfinite, {**common, "min_cluster_size": 2, "min_samples": 2}, "Dense rows containing NaN and positive infinity."),
        Case("high_dimensional", blobs(19, 32, 64, 2, 0.3), {**common, "min_cluster_size": 4, "min_samples": 4}, "Fixed high-dimensional dense data."),
        Case("synthetic_blobs", blobs(42, 60, 2, 3, 0.35), {**common, "min_cluster_size": 5, "min_samples": 5}, "Fixed low-dimensional Gaussian blobs."),
        Case("branch_shapes", branch_shapes, {**common, "min_cluster_size": 3, "min_samples": 3, "branch_detection_data": True, "prediction_data": True}, "Pinned FLASC three-flare example."),
        Case("metric_chebyshev", blobs(22, 24, 3, 2, 0.2), {**common, "metric": "chebyshev", "min_cluster_size": 3, "min_samples": 3}, "Chebyshev metric parity."),
        Case("metric_canberra", np.abs(blobs(23, 24, 3, 2, 0.2)) + 0.1, {**common, "metric": "canberra", "min_cluster_size": 3, "min_samples": 3}, "Canberra metric parity."),
        Case("metric_braycurtis", np.abs(blobs(24, 24, 3, 2, 0.2)) + 0.1, {**common, "metric": "braycurtis", "min_cluster_size": 3, "min_samples": 3}, "Bray-Curtis metric parity."),
        Case("precomputed_sparse_connected", sparse_connected, {**common, "metric": "precomputed", "min_cluster_size": 2, "min_samples": 1}, "Connected precomputed CSR graph."),
        Case("precomputed_disconnected", disconnected, {**common, "metric": "precomputed", "min_cluster_size": 2, "min_samples": 2}, "Disconnected precomputed dense graph."),
    ]


def encode_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return float(value)


def encode_array(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype.names:
        return [
            {name: encode_array(row[name]) for name in array.dtype.names}
            for row in array
        ]
    if array.ndim == 0:
        item = array.item()
        return encode_float(item) if isinstance(item, float) else item
    return [encode_array(item) for item in array]


def expected(case: Case) -> dict[str, Any]:
    try:
        clusterer = hdbscan.HDBSCAN(**case.config).fit(case.data)
        result: dict[str, Any] = {
            "outcome": "success",
            "labels": encode_array(clusterer.labels_),
            "probabilities": encode_array(clusterer.probabilities_),
            "cluster_persistence": encode_array(clusterer.cluster_persistence_),
            "outlier_scores": encode_array(clusterer.outlier_scores_),
            "single_linkage_tree": encode_array(clusterer.single_linkage_tree_.to_numpy()),
            "condensed_tree": encode_array(clusterer.condensed_tree_.to_numpy()),
        }
        try:
            result["minimum_spanning_tree"] = encode_array(clusterer.minimum_spanning_tree_.to_numpy())
        except (AttributeError, ValueError):
            result["minimum_spanning_tree"] = None
        return result
    except Exception as error:  # fixtures intentionally include invalid inputs
        return {
            "outcome": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def document(case: Case) -> dict[str, Any]:
    metric = case.config.get("metric", "minkowski")
    if issparse(case.data):
        matrix = case.data.tocsr()
        input_case = {
            "name": case.name,
            "description": case.description,
            "input_kind": "precomputed_csr",
            "dtype": "float64",
            "rows": int(matrix.shape[0]),
            "cols": int(matrix.shape[1]),
            "csr": {
                "data": encode_array(matrix.data),
                "indices": encode_array(matrix.indices),
                "indptr": encode_array(matrix.indptr),
            },
        }
    else:
        input_case = {
            "name": case.name,
            "description": case.description,
            "input_kind": "precomputed_dense" if metric == "precomputed" else "dense",
            "dtype": "float64",
            "rows": int(case.data.shape[0]),
            "cols": int(case.data.shape[1]),
            "data": encode_array(case.data.reshape(-1)),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "upstream": {
            "package": "hdbscan",
            "version": version("hdbscan"),
            "commit": UPSTREAM_COMMIT,
        },
        "case": input_case,
        "config": case.config,
        "expected": expected(case),
    }
    if case.name == "synthetic_blobs":
        labels, hierarchy = hdbscan.robust_single_linkage(
            case.data, cut=1.0, k=5, alpha=1.0, gamma=5,
            metric="euclidean", algorithm="generic",
        )
        fitted = hdbscan.HDBSCAN(**case.config).fit(case.data)
        score, per_cluster = hdbscan.validity_index(
            case.data, fitted.labels_, metric="euclidean", per_cluster_scores=True,
        )
        result["extended"] = {
            "robust_single_linkage": {"cut": 1.0, "k": 5, "alpha": 1.0, "gamma": 5,
                                      "labels": encode_array(labels), "hierarchy": encode_array(hierarchy)},
            "validity": {"labels": encode_array(fitted.labels_), "score": encode_float(score),
                         "per_cluster": encode_array(per_cluster)},
        }
    if case.name == "branch_shapes":
        fitted = hdbscan.HDBSCAN(**case.config).fit(case.data)
        branches = {}
        for method in ("full", "core"):
            detector = BranchDetector(branch_detection_method=method).fit(
                fitted, labels=np.zeros(case.data.shape[0], dtype=np.intp)
            )
            branches[method] = {
                "labels": encode_array(detector.labels_),
                "probabilities": encode_array(detector.probabilities_),
                "cluster_labels": encode_array(detector.cluster_labels_),
                "branch_labels": encode_array(detector.branch_labels_),
                "branch_probabilities": encode_array(detector.branch_probabilities_),
                "branch_persistences": [encode_array(x) for x in detector.branch_persistences_],
                "centralities": encode_array(detector.centralities_),
                "cluster_points": [encode_array(x) for x in detector.cluster_points_],
                "graph_edge_counts": [
                    len({tuple(sorted((int(row[0]), int(row[1])))) for row in graph if row[0] != row[1]})
                    for graph in detector._approximation_graphs
                ],
                "condensed_tree_counts": [len(x) for x in detector._condensed_trees],
                "linkage_tree_counts": [len(x) for x in detector._linkage_trees],
            }
        result["branches"] = branches
    return result


def render() -> dict[str, bytes]:
    rendered: dict[str, bytes] = {}
    for case in cases():
        payload = json.dumps(document(case), indent=2, sort_keys=True, allow_nan=False) + "\n"
        rendered[f"{case.name}.json"] = payload.encode()
    hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(rendered.items())
    }
    rendered["SHA256SUMS.json"] = (json.dumps(hashes, indent=2, sort_keys=True) + "\n").encode()
    return rendered


def write_files(rendered: dict[str, bytes], destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("*.json"):
        path.unlink()
    for name, payload in rendered.items():
        (destination / name).write_bytes(payload)


def check_files(rendered: dict[str, bytes]) -> int:
    with tempfile.TemporaryDirectory() as directory:
        generated = pathlib.Path(directory)
        write_files(rendered, generated)
        expected_names = sorted(rendered)
        actual_names = sorted(path.name for path in OUTPUT.glob("*.json"))
        if expected_names != actual_names:
            print(f"fixture file set differs: generated={expected_names}, committed={actual_names}")
            return 1
        changed = [name for name in expected_names if (generated / name).read_bytes() != (OUTPUT / name).read_bytes()]
        if changed:
            print("fixtures differ: " + ", ".join(changed))
            return 1
    print(f"verified {len(rendered) - 1} fixtures against hdbscan {version('hdbscan')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed fixtures without modifying them")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        return check_files(rendered)
    write_files(rendered, OUTPUT)
    print(f"wrote {len(rendered) - 1} fixtures to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
