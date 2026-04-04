#!/usr/bin/env python3
"""
Bundle model artifacts for deployment (copy + manifest + optional archive).

Example (from repo root):
  python deployment/package_models.py --out deployment/dist/model_bundle
  python deployment/package_models.py --out deployment/dist/model_bundle --zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package model.pkl + metadata for deployment.")
    p.add_argument(
        "--out",
        "-o",
        default=str(ROOT / "deployment" / "dist" / "model_bundle"),
        help="Output directory for copied artifacts.",
    )
    p.add_argument(
        "--model-path",
        default=str(ROOT / "models" / "model_full.pkl"),
        help="Primary model to ship (default: models/model_full.pkl).",
    )
    p.add_argument(
        "--metadata-path",
        default=str(ROOT / "models" / "model_metadata.json"),
        help="Metadata JSON (default: models/model_metadata.json).",
    )
    p.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="Additional file to copy into bundle (repeatable), e.g. models/model_validation.pkl",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="Also write a .zip next to --out with the same contents.",
    )
    p.add_argument(
        "--include-src",
        action="store_true",
        help="Copy src/ and deployment/score.py into the bundle for offline scoring.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    model_path = Path(args.model_path)
    meta_path = Path(args.metadata_path)

    missing: list[str] = []
    if not model_path.is_file():
        missing.append(str(model_path))
    if not meta_path.is_file():
        missing.append(str(meta_path))
    if missing:
        print(
            "Error: required files missing. Train/save models first (notebooks/04_modeling.ipynb):\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    files_manifest: list[dict[str, Any]] = []

    def copy_one(src: Path, dest_name: str | None = None) -> None:
        name = dest_name or src.name
        dest = out_dir / name
        shutil.copy2(src, dest)
        files_manifest.append(
            {
                "path": name,
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
            }
        )

    copy_one(model_path, "model_full.pkl")
    copy_one(meta_path, "model_metadata.json")
    req_src = ROOT / "deployment" / "requirements-inference.txt"
    if req_src.is_file():
        copy_one(req_src, "requirements-inference.txt")

    score_script = ROOT / "deployment" / "score.py"
    if score_script.is_file():
        copy_one(score_script, "score.py")

    if args.include_src:
        src_dst = out_dir / "src"
        shutil.copytree(ROOT / "src", src_dst, dirs_exist_ok=True)

    for extra in args.extra_model:
        ep = Path(extra)
        if not ep.is_file():
            print(f"Warning: skipping missing --extra-model {ep}", file=sys.stderr)
            continue
        copy_one(ep)

    manifest = {
        "bundle_format": 1,
        "root": str(ROOT),
        "files": files_manifest,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Bundle written to {out_dir.resolve()}")
    print(f"Manifest: {manifest_path}")

    if args.zip:
        zip_path = out_dir.with_suffix(".zip") if out_dir.suffix != ".zip" else out_dir.parent / (out_dir.name + ".zip")
        if zip_path == out_dir:
            zip_path = out_dir.parent / (out_dir.name + "_bundle.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out_dir.rglob("*"):
                if p.is_file():
                    arc = p.relative_to(out_dir)
                    zf.write(p, arcname=str(Path(out_dir.name) / arc))
        print(f"Zip archive: {zip_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
