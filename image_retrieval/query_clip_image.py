#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autotag.clip_index.qdrant_clip import QdrantClipIndexer
from autotag.config import Settings
from autotag.utils.crop_json import crop_json_path_for
from autotag.utils.jsonio import atomic_write_json, read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reverse image search for SKU using CLIP + Qdrant")
    p.add_argument("--query", required=True, help="Path to query image")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dedupe-by", choices=["none", "sku", "name"], default="name", help="Collapse duplicate retrieval hits (default: name)")
    p.add_argument("--output", default=None, help="Output folder for retrieval results (default: image_retrieval/results)")
    p.add_argument("--copy-assets", action="store_true", help="Copy matched crop and annotated images into result folder")
    return p.parse_args()


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return read_json(path)
    except Exception:
        return None
    return None


def _resolve_match_artifacts(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    image_id = str(payload.get("image_id", ""))
    crop_id = str(payload.get("crop_id", ""))
    crop_path = Path(str(payload.get("crop_path", ""))) if payload.get("crop_path") else None
    image_json_path = Path(str(payload.get("output_json_path", ""))) if payload.get("output_json_path") else None
    crop_json_path = crop_json_path_for(settings.output_json_dir, image_id, crop_id) if image_id and crop_id else None

    crop_json = _safe_read_json(crop_json_path) if crop_json_path else None
    image_json = _safe_read_json(image_json_path) if image_json_path else None

    annotated_image_path = None
    if crop_json:
        annotated_image_path = ((crop_json.get("image_annotation") or {}).get("labeled_image_path"))
    if not annotated_image_path and image_json:
        annotated_image_path = ((image_json.get("image_annotation") or {}).get("labeled_image_path"))

    matched_name = (
        payload.get("title")
        or payload.get("name")
        or payload.get("product_name")
        or payload.get("sku_name")
        or payload.get("display_name")
    )
    if crop_json:
        det = crop_json.get("detection") or {}
        topk = ((det.get("text_match") or {}).get("top_k") or [])
        if topk:
            top_payload = topk[0].get("payload") or {}
            matched_name = (
                top_payload.get("title")
                or top_payload.get("name")
                or top_payload.get("product_name")
                or top_payload.get("sku_name")
                or top_payload.get("display_name")
            )

    return {
        "image_id": image_id,
        "crop_id": crop_id,
        "crop_path": str(crop_path) if crop_path else None,
        "crop_json_path": str(crop_json_path) if crop_json_path else None,
        "image_json_path": str(image_json_path) if image_json_path else None,
        "annotated_image_path": str(annotated_image_path) if annotated_image_path else None,
        "matched_name": matched_name,
    }


def _normalize_variant_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = " ".join(str(name).strip().lower().split())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized or None


def _copy_if_exists(src: str | None, dst_dir: Path) -> str | None:
    if not src:
        return None
    p = Path(src)
    if not p.exists() or not p.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / p.name
    if not dst.exists():
        shutil.copy2(p, dst)
    return str(dst)


def main() -> int:
    args = parse_args()
    settings = Settings.from_env(ROOT)
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL is required for retrieval")

    query_path = Path(args.query).expanduser().resolve()
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    result_root = Path(args.output).expanduser().resolve() if args.output else (ROOT / "image_retrieval" / "results")
    result_root.mkdir(parents=True, exist_ok=True)

    indexer = QdrantClipIndexer(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_clip_collection,
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
        device=settings.device,
    )

    raw_limit = args.top_k if args.dedupe_by == "none" else max(args.top_k * 5, args.top_k)
    hits = indexer.search_similar(query_path, limit=raw_limit)
    enriched_all = []
    for rank, hit in enumerate(hits, 1):
        payload = hit.get("payload") or {}
        artifacts = _resolve_match_artifacts(payload, settings)
        row = {
            "rank": rank,
            "score": hit.get("score"),
            "point_id": hit.get("id"),
            "sku": payload.get("sku"),
            "clip_model": payload.get("clip_model"),
            "clip_pretrained": payload.get("clip_pretrained"),
            **artifacts,
        }
        enriched_all.append(row)

    if args.dedupe_by in {"sku", "name"}:
        deduped: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for row in enriched_all:
            if args.dedupe_by == "name":
                key = _normalize_variant_name(row.get("matched_name")) or str(row.get("sku") or f"__point__:{row.get('point_id')}")
            else:
                key = str(row.get("sku") or "")
                if not key:
                    key = f"__point__:{row.get('point_id')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)
            if len(deduped) >= args.top_k:
                break
        enriched = deduped
    else:
        enriched = enriched_all[: args.top_k]

    for i, row in enumerate(enriched, 1):
        if args.copy_assets:
            asset_dir = result_root / "assets" / f"rank_{i:02d}"
            row["copied_crop_path"] = _copy_if_exists(row.get("crop_path"), asset_dir)
            row["copied_annotated_image_path"] = _copy_if_exists(row.get("annotated_image_path"), asset_dir)
            row["copied_crop_json_path"] = _copy_if_exists(row.get("crop_json_path"), asset_dir)

    for i, row in enumerate(enriched, 1):
        row["rank"] = i

    output = {
        "query_image": str(query_path),
        "qdrant_collection": settings.qdrant_clip_collection,
        "top_k": args.top_k,
        "dedupe_by": args.dedupe_by,
        "raw_hits_considered": len(enriched_all),
        "results": enriched,
    }
    out_json = result_root / "last_query_results.json"
    atomic_write_json(out_json, output)

    print(f"Query image: {query_path}")
    print(f"Results written: {out_json}")
    for row in enriched:
        print(
            f"[{row['rank']}] score={row['score']:.4f} sku={row.get('sku')} name={row.get('matched_name') or '-'}"
        )
        print(f"    crop: {row.get('crop_path')}")
        print(f"    crop_json: {row.get('crop_json_path')}")
        print(f"    annotated_image: {row.get('annotated_image_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
