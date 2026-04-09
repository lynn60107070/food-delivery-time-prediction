#!/usr/bin/env python3
"""
Append rows to reports/test_prediction.csv with scored_at_utc spread across a date range.

Guarantees at least one order per calendar day in [start, end] (inclusive), with random
times within each day (same format as simulate_scoring: %%Y-%%m-%%dT%%H:%%M:%%SZ).

Examples:
  python deployment/append_test_prediction_date_range.py --total-rows 2500 --seed 42

  # More rows per month, natural On-Time / Delayed mix (no stress tweaks), model loaded once:
  python deployment/append_test_prediction_date_range.py --runs 5 --total-rows 2000 --no-scenario-tweaks --seed 1
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parent if (_SCRIPT.parent / "src").is_dir() else _SCRIPT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.simulate_scoring import (  # noqa: E402
    EXTRA_CONTEXT_COLS,
    apply_demanding_scenario_tweaks,
    apply_mild_scenario_tweaks,
)
from src.data_preprocessing import preprocess_for_scoring  # noqa: E402
from src.scoring import (  # noqa: E402
    add_predicted_sla_status,
    load_metadata,
    load_model,
    predict_delivery_time_preprocessed,
)


def _default_model_path() -> str:
    if (ROOT / "model_full.pkl").is_file():
        return str(ROOT / "model_full.pkl")
    return str(ROOT / "models" / "model_full.pkl")


def _default_metadata_path() -> str:
    if (ROOT / "model_metadata.json").is_file():
        return str(ROOT / "model_metadata.json")
    return str(ROOT / "models" / "model_metadata.json")


def _daterange_inclusive(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def build_scored_at_utc_strings(
    n: int,
    start: date,
    end: date,
    rng: np.random.Generator,
) -> list[str]:
    """n timestamps in ISO UTC; every calendar day in [start, end] appears at least once."""
    days = _daterange_inclusive(start, end)
    n_days = len(days)
    if n < n_days:
        raise ValueError(
            f"--total-rows ({n}) must be >= number of days ({n_days}) "
            f"from {start} to {end} inclusive."
        )

    def _utc_ts(d: date) -> str:
        h = int(rng.integers(0, 24))
        mi = int(rng.integers(0, 60))
        s = int(rng.integers(0, 60))
        dt = datetime(d.year, d.month, d.day, h, mi, s, tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    scored = [_utc_ts(d) for d in days]
    day_indices = np.arange(n_days)
    for _ in range(n - n_days):
        d = days[int(rng.choice(day_indices))]
        scored.append(_utc_ts(d))

    rng.shuffle(scored)
    return scored


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append test_prediction rows with scored_at_utc across a date range (one+ row per day)."
    )
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "raw" / "test.csv",
        help="Raw test CSV (default: data/raw/test.csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "test_prediction.csv",
        help="Append target (default: reports/test_prediction.csv).",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default="2026-01-01",
        help="First calendar day (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default="2026-04-08",
        help="Last calendar day (YYYY-MM-DD).",
    )
    p.add_argument(
        "--total-rows",
        type=int,
        default=2500,
        help="Rows per run (must be >= days in range).",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat append this many times with disjoint test rows (single model load).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (sampling + timestamps + tweaks).")
    p.add_argument("--model-path", default=None, help="Override model .pkl path.")
    p.add_argument("--metadata-path", default=None, help="Override model_metadata.json path.")
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not load metadata JSON.",
    )
    sc = p.add_mutually_exclusive_group()
    sc.add_argument(
        "--no-scenario-tweaks",
        action="store_true",
        help="Use preprocessed rows as-is (most natural SLA mix).",
    )
    sc.add_argument(
        "--mild-scenario-tweaks",
        action="store_true",
        help="Light randomization of distance/traffic/hours (balanced, not stress-test).",
    )
    return p.parse_args()


def _apply_scenario_tweaks(
    prep: pd.DataFrame,
    rng: np.random.Generator,
    *,
    no_scenario_tweaks: bool,
    mild_scenario_tweaks: bool,
) -> pd.DataFrame:
    if no_scenario_tweaks:
        return prep
    if mild_scenario_tweaks:
        return apply_mild_scenario_tweaks(prep, rng)
    return apply_demanding_scenario_tweaks(prep, rng)


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    out_path = Path(args.output)
    if not source.is_file():
        print(f"Error: source not found: {source}", file=sys.stderr)
        return 1

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end < start:
        print("Error: end-date must be on or after start-date.", file=sys.stderr)
        return 1

    runs = max(1, int(args.runs))
    n = args.total_rows

    raw = pd.read_csv(source)
    if raw.empty:
        print("Error: source CSV is empty.", file=sys.stderr)
        return 1

    need = n * runs
    if len(raw) < need:
        print(
            f"Error: need at least {need} rows in source for {runs} run(s) x {n} rows; got {len(raw)}.",
            file=sys.stderr,
        )
        return 1

    n_days = len(_daterange_inclusive(start, end))
    if n < n_days:
        print(
            f"Error: --total-rows ({n}) must be >= number of days ({n_days}) in the date range.",
            file=sys.stderr,
        )
        return 1

    model_path = args.model_path or _default_model_path()
    metadata_path = args.metadata_path or _default_metadata_path()

    metadata = None
    if not args.no_metadata and Path(metadata_path).is_file():
        metadata = load_metadata(metadata_path)
    elif not args.no_metadata:
        print(f"Warning: metadata not found at {metadata_path}; continuing without.", file=sys.stderr)

    order_col = "ID" if "ID" in raw.columns else "order_id"
    if order_col not in raw.columns:
        print("Error: expected ID or order_id column in source.", file=sys.stderr)
        return 1

    pipeline = load_model(model_path)

    if runs > 1:
        pool = raw.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        pool = None

    pieces: list[pd.DataFrame] = []

    for run in range(runs):
        rng = np.random.default_rng(args.seed + run * 100_003)

        try:
            scored_at_list = build_scored_at_utc_strings(n, start, end, rng)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if runs > 1:
            batch = pool.iloc[run * n : (run + 1) * n].copy()
        else:
            batch = raw.sample(n=n, replace=False, random_state=args.seed)

        prep = preprocess_for_scoring(batch.copy())
        prep = _apply_scenario_tweaks(
            prep,
            rng,
            no_scenario_tweaks=args.no_scenario_tweaks,
            mild_scenario_tweaks=args.mild_scenario_tweaks,
        )
        preds = predict_delivery_time_preprocessed(prep, pipeline, metadata=metadata)

        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        out = pd.DataFrame(
            {
                "order_id": batch[order_col].astype(str).str.strip().values,
                "predicted_delivery_time_min": preds,
                "simulation_batch_id": batch_id,
                "scored_at_utc": scored_at_list,
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
        pieces.append(out)
        delayed_pct = 100.0 * (out["predicted_SLA_status"] == "Delayed").mean()
        print(f"Run {run + 1}/{runs} | batch_id={batch_id} | rows={len(out)} | Delayed ~{delayed_pct:.1f}%")

    combined_new = pd.concat(pieces, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        existing = pd.read_csv(out_path)
        if (
            "predicted_SLA_status" not in existing.columns
            and "predicted_delivery_time_min" in existing.columns
        ):
            existing = add_predicted_sla_status(existing)
        combined = pd.concat([existing, combined_new], ignore_index=True)
    else:
        combined = combined_new

    combined.to_csv(out_path, index=False)
    d_pct = 100.0 * (combined_new["predicted_SLA_status"] == "Delayed").mean()
    print(
        f"Appended {len(combined_new)} rows in {runs} run(s) | "
        f"this append Delayed ~{d_pct:.1f}% | "
        f"date span {start} .. {end} ({n_days} days, at least 1 row/day per run)"
    )
    print(f"Total rows in {out_path}: {len(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
