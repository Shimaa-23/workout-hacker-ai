#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    script_path = Path(__file__).resolve()
    rust_inference_dir = script_path.parents[1]
    repo_root = script_path.parents[3]

    parser = argparse.ArgumentParser(
        description="Export sklearn RandomForest to Rust-friendly JSON"
    )
    parser.add_argument(
        "--model",
        default=str(repo_root / "models" / "exercise_classifier_rf.joblib"),
        help="Path to joblib model file",
    )
    parser.add_argument(
        "--output",
        default=str(rust_inference_dir / "model_rf.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=60,
        help="Expected number of input features",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()

    model = joblib.load(model_path)

    if model.__class__.__name__ != "RandomForestClassifier":
        raise ValueError(
            f"Expected RandomForestClassifier, got {model.__class__.__name__}"
        )

    feature_count = int(getattr(model, "n_features_in_", args.features))
    if feature_count != args.features:
        print(
            f"Warning: model reports n_features_in_={feature_count}, "
            f"but --features={args.features} was passed. Using model value."
        )

    forest = {
        "n_features": feature_count,
        "n_classes": int(len(model.classes_)),
        "classes": [int(c) for c in model.classes_],
        "trees": [],
    }

    for est in model.estimators_:
        tree = est.tree_
        node_count = int(tree.node_count)
        values = tree.value[:, 0, :]
        forest["trees"].append(
            {
                "children_left": [int(v) for v in tree.children_left.tolist()],
                "children_right": [int(v) for v in tree.children_right.tolist()],
                "feature": [int(v) for v in tree.feature.tolist()],
                "threshold": [float(v) for v in tree.threshold.tolist()],
                "values": [
                    [float(x) for x in values[i].tolist()] for i in range(node_count)
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(forest, f, separators=(",", ":"))

    print(f"Exported RF JSON model to: {output_path}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Feature count: {feature_count}")
    print(f"Trees: {len(model.estimators_)}")
    print(f"Output size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
