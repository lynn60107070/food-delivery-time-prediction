#!/usr/bin/env python3
"""
Batch scoring CLI for deployment: load joblib sklearn pipeline + optional metadata, write predictions.

Example (from repo root):
  python deployment/score.py --input data/raw/test.csv --output reports/scored.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve()
# Repo: deployment/score.py -> project root is parent. Bundle: score.py next to src/ -> root is here.
ROOT = _SCRIPT.parent if (_SCRIPT.parent / "src").is_dir() else _SCRIPT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring import load_metadata, load_model, predict_delivery_time  # noqa: E402


def _default_model_path() -> str:
    if (ROOT / "model_full.pkl").is_file():
        return str(ROOT / "model_full.pkl")
    return str(ROOT / "models" / "model_full.pkl")


def _default_metadata_path() -> str:
    if (ROOT / "model_metadata.json").is_file():
        return str(ROOT / "model_metadata.json")
    return str(ROOT / "models" / "model_metadata.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score delivery-time model (batch CSV).")
    p.add_argument("--input", "-i", required=True, help="Path to raw CSV (same schema as training/test).")
    p.add_argument("--output", "-o", required=True, help="Path to write predictions CSV.")
    p.add_argument(
        "--model-path",
        default=None,
        help="Joblib sklearn Pipeline (default: model_full.pkl in bundle root or models/).",
    )
    p.add_argument(
        "--metadata-path",
        default=None,
        help="model_metadata.json (default: next to model or models/).",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not load metadata JSON; rely on pipeline feature_names_in_ only.",
    )
    p.add_argument(
        "--id-column",
        default="order_id",
        help="If present, include this column in the output (default: order_id).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path or _default_model_path()
    metadata_path = args.metadata_path or _default_metadata_path()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    metadata = None
    if not args.no_metadata:
        meta_path = Path(metadata_path)
        if meta_path.is_file():
            metadata = load_metadata(meta_path)
        else:
            print(f"Warning: metadata not found at {meta_path}, using pipeline columns only.", file=sys.stderr)

    pipeline = load_model(model_path)
    df = pd.read_csv(input_path)
    preds = predict_delivery_time(df, pipeline, metadata=metadata)

    out = pd.DataFrame({"predicted_delivery_time_min": preds})
    if args.id_column and args.id_column in df.columns:
        out.insert(0, args.id_column, df[args.id_column].values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
