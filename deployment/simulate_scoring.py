#!/usr/bin/env python3
"""
Simulated “live” batch scoring for demos: sample unseen-style rows, score with the packaged model,
append results to a single CSV for Power BI (refresh to update).

By default applies **demanding-scenario tweaks** (longer distance, heavier traffic, peak hours, more
deliveries) before prediction so dashboards can show more “stress” cases. Disable with --no-scenario-tweaks.

Example (from repo root):
  python deployment/simulate_scoring.py --batch-size 30
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parent if (_SCRIPT.parent / "src").is_dir() else _SCRIPT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import preprocess_for_scoring  # noqa: E402
from src.scoring import (  # noqa: E402
    add_predicted_sla_status,
    load_metadata,
    load_model,
    predict_delivery_time,
    predict_delivery_time_preprocessed,
)

# Columns written to CSV for Power BI (post-tweak values when tweaks are on)
EXTRA_CONTEXT_COLS = [
    "distance_km",
    "traffic_density",
    "weather",
    "order_hour",
    "num_deliveries",
]

PEAK_HOURS = [12, 13, 14, 15, 18, 19, 20, 21, 22]
TRAFFIC_OPTIONS = ["Low", "Medium", "High", "Jam"]
TRAFFIC_WEIGHTS = [1, 2, 3, 4]


def apply_demanding_scenario_tweaks(prep: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bias inputs toward harder trips: distance, traffic, peak hours, multiple deliveries."""
    out = prep.copy()
    n = len(out)
    p_traffic = np.array(TRAFFIC_WEIGHTS, dtype=float)
    p_traffic /= p_traffic.sum()
    traffic_vals = rng.choice(TRAFFIC_OPTIONS, size=n, p=p_traffic)
    out["traffic_density"] = traffic_vals.astype(str)

    out["distance_km"] = rng.uniform(5.0, 15.0, size=n)

    out["order_hour"] = rng.choice(PEAK_HOURS, size=n)
    if "is_peak_hour" in out.columns:
        out["is_peak_hour"] = out["order_hour"].isin(PEAK_HOURS).astype(int)

    if "num_deliveries" in out.columns:
        out["num_deliveries"] = rng.integers(2, 5, size=n, endpoint=False)

    return out


def apply_mild_scenario_tweaks(prep: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Light randomization of trip context: more balanced traffic and hours than stress mode."""
    out = prep.copy()
    n = len(out)
    p_traffic = np.array([2.0, 2.0, 1.5, 1.5], dtype=float)
    p_traffic /= p_traffic.sum()
    traffic_vals = rng.choice(TRAFFIC_OPTIONS, size=n, p=p_traffic)
    out["traffic_density"] = traffic_vals.astype(str)

    out["distance_km"] = rng.uniform(2.0, 12.0, size=n)

    out["order_hour"] = rng.integers(8, 23, size=n)
    if "is_peak_hour" in out.columns:
        out["is_peak_hour"] = out["order_hour"].isin(PEAK_HOURS).astype(int)

    if "num_deliveries" in out.columns:
        out["num_deliveries"] = rng.integers(1, 5, size=n, endpoint=False)

    return out


def _default_model_path() -> str:
    if (ROOT / "model_full.pkl").is_file():
        return str(ROOT / "model_full.pkl")
    return str(ROOT / "models" / "model_full.pkl")


def _default_metadata_path() -> str:
    if (ROOT / "model_metadata.json").is_file():
        return str(ROOT / "model_metadata.json")
    return str(ROOT / "models" / "model_metadata.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate a scoring run: sample rows, predict, append to test_prediction.csv."
    )
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "raw" / "test.csv",
        help="Raw CSV with same columns as training/test (default: data/raw/test.csv).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of rows to sample and score this run.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "test_prediction.csv",
        help="Append-only scored output for Power BI (default: reports/test_prediction.csv).",
    )
    p.add_argument("--model-path", default=None, help="Override model .pkl path.")
    p.add_argument("--metadata-path", default=None, help="Override model_metadata.json path.")
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not load metadata JSON (use pipeline feature_names_in_ only).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling + scenario tweaks (default: different each run).",
    )
    p.add_argument(
        "--no-scenario-tweaks",
        action="store_true",
        help="Score from raw preprocessed rows only (no distance/traffic/peak/delivery bias).",
    )
    p.add_argument(
        "--demo-delay-bump",
        action="store_true",
        help="After prediction, add random 0 / 10 / 15 minutes to some rows (demo only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    out_path = Path(args.output)
    if not source.is_file():
        print(f"Error: source not found: {source}", file=sys.stderr)
        return 1

    model_path = args.model_path or _default_model_path()
    metadata_path = args.metadata_path or _default_metadata_path()

    metadata = None
    if not args.no_metadata and Path(metadata_path).is_file():
        metadata = load_metadata(metadata_path)
    elif not args.no_metadata:
        print(f"Warning: metadata not found at {metadata_path}; continuing without.", file=sys.stderr)

    raw = pd.read_csv(source)
    if raw.empty:
        print("Error: source CSV is empty.", file=sys.stderr)
        return 1

    n = min(args.batch_size, len(raw))
    batch = raw.sample(n=n, random_state=args.seed)

    rng = np.random.default_rng(args.seed)

    pipeline = load_model(model_path)

    prep = preprocess_for_scoring(batch.copy())
    if not args.no_scenario_tweaks:
        prep = apply_demanding_scenario_tweaks(prep, rng)
        preds = predict_delivery_time_preprocessed(prep, pipeline, metadata=metadata)
    else:
        preds = predict_delivery_time(batch, pipeline, metadata=metadata)
        prep = preprocess_for_scoring(batch.copy())

    if args.demo_delay_bump:
        bump = rng.choice([0, 10, 15], size=len(preds))
        preds = preds + bump

    order_col = "ID" if "ID" in batch.columns else "order_id"
    if order_col not in batch.columns:
        print("Error: expected ID or order_id column in source.", file=sys.stderr)
        return 1

    scored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    out = pd.DataFrame(
        {
            "order_id": batch[order_col].astype(str).str.strip().values,
            "predicted_delivery_time_min": preds,
            "simulation_batch_id": batch_id,
            "scored_at_utc": scored_at,
        }
    )

    for col in EXTRA_CONTEXT_COLS:
        if col not in prep.columns:
            out[col] = np.nan
            continue
        s = prep[col]
        if col in ("traffic_density", "weather"):
            out[col] = s.astype(str).values
        else:
            out[col] = pd.to_numeric(s, errors="coerce").values

    out = add_predicted_sla_status(out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        existing = pd.read_csv(out_path)
        if (
            "predicted_SLA_status" not in existing.columns
            and "predicted_delivery_time_min" in existing.columns
        ):
            existing = add_predicted_sla_status(existing)
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out

    combined.to_csv(out_path, index=False)
    print(f"Scored {len(out)} rows | batch_id={batch_id}")
    print(f"Appended to {out_path.resolve()}")
    print(f"Total rows in file: {len(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
