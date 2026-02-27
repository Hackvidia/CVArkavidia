#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autotag.clip_index.qdrant_clip import QdrantClipIndexer
from autotag.config import Settings
from autotag.detector import UltralyticsDetector
from autotag.utils.files import ensure_dir, sha256_file
from autotag.utils.image_io import crop_image, get_image_size, render_labeled_image
from autotag.utils.crop_json import crop_json_path_for
from autotag.utils.jsonio import atomic_write_json, read_json


PIPELINE_VERSION = "0.1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_bbox(bbox: dict[str, int], width: int | None, height: int | None) -> dict[str, int]:
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    if width is not None:
        x1 = min(max(0, x1), width)
        x2 = min(max(0, x2), width)
    if height is not None:
        y1 = min(max(0, y1), height)
        y2 = min(max(0, y2), height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _bbox_valid(bbox: dict[str, int], min_size: int) -> bool:
    return (bbox["x2"] - bbox["x1"]) >= min_size and (bbox["y2"] - bbox["y1"]) >= min_size


def _safe_read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        if path.exists():
            return read_json(path)
    except Exception:
        return None
    return None


def _readable_name_from_payload(payload: dict[str, Any]) -> str | None:
    return (
        payload.get("title")
        or payload.get("name")
        or payload.get("product_name")
        or payload.get("sku_name")
        or payload.get("display_name")
    )


def _normalize_variant_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = " ".join(str(name).strip().lower().split())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized or None


def _resolve_match_artifacts(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    image_id = str(payload.get("image_id", ""))
    crop_id = str(payload.get("crop_id", ""))
    crop_path = Path(str(payload.get("crop_path", ""))) if payload.get("crop_path") else None
    image_json_path = Path(str(payload.get("output_json_path", ""))) if payload.get("output_json_path") else None
    crop_json_path = crop_json_path_for(settings.output_json_dir, image_id, crop_id) if image_id and crop_id else None

    crop_json = _safe_read_json(crop_json_path)
    image_json = _safe_read_json(image_json_path)
    annotated_image_path = None
    if crop_json:
        annotated_image_path = (crop_json.get("image_annotation") or {}).get("labeled_image_path")
    if not annotated_image_path and image_json:
        annotated_image_path = (image_json.get("image_annotation") or {}).get("labeled_image_path")

    payload_name = _readable_name_from_payload(payload)
    if not payload_name and crop_json:
        det = crop_json.get("detection") or {}
        topk = ((det.get("text_match") or {}).get("top_k") or [])
        if topk:
            payload_name = _readable_name_from_payload(topk[0].get("payload") or {})

    return {
        "matched_image_id": image_id or None,
        "matched_crop_id": crop_id or None,
        "matched_crop_path": str(crop_path) if crop_path else None,
        "matched_crop_json_path": str(crop_json_path) if crop_json_path else None,
        "matched_image_json_path": str(image_json_path) if image_json_path else None,
        "matched_annotated_image_path": str(annotated_image_path) if annotated_image_path else None,
        "matched_name": payload_name,
    }


def _dedupe_hits(hits: list[dict[str, Any]], mode: str, top_k: int, settings: Settings | None = None) -> list[dict[str, Any]]:
    if mode == "none":
        return hits[:top_k]
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        payload = hit.get("payload") or {}
        if mode == "name":
            resolved_name = None
            if settings is not None:
                try:
                    resolved_name = _resolve_match_artifacts(payload, settings).get("matched_name")
                except Exception:
                    resolved_name = None
            key = _normalize_variant_name(resolved_name or _readable_name_from_payload(payload)) or str(payload.get("sku") or f"__point__:{hit.get('id')}")
        else:
            key = str(payload.get("sku") or f"__point__:{hit.get('id')}")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
        if len(deduped) >= top_k:
            break
    return deduped


def _review_flag(top1_score: float | None, margin: float | None, score_threshold: float, margin_threshold: float) -> tuple[bool, str | None]:
    reasons: list[str] = []
    if top1_score is None:
        reasons.append("no_top1_score")
    elif top1_score < score_threshold:
        reasons.append("score_below_threshold")
    if margin is not None and margin < margin_threshold:
        reasons.append("margin_below_threshold")
    return (len(reasons) > 0, ",".join(reasons) if reasons else None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-box reverse image retrieval using YOLO + CLIP(Qdrant)")
    p.add_argument("--query", required=True, help="Path to query image")
    p.add_argument("--top-k", type=int, default=5, help="Top K candidates per detected box")
    p.add_argument("--dedupe-by", choices=["none", "sku", "name"], default="name", help="Dedupe retrieval candidates per detected box")
    p.add_argument("--output", default=None, help="Output root (default: image_retrieval/results)")
    p.add_argument("--score-threshold", type=float, default=0.50, help="Review flag if top1 score below this threshold")
    p.add_argument("--margin-threshold", type=float, default=None, help="Review flag if top1-top2 margin below this threshold")
    p.add_argument("--candidate-delta", type=float, default=0.05, help="Include candidates whose score is within this delta of top1")
    p.add_argument("--max-crops", type=int, default=None, help="Override max YOLO detections/crops")
    p.add_argument("--yolo-min-confidence", type=float, default=0.60, help="Skip YOLO detections below this confidence before cropping")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env(ROOT)
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL is required")

    query_path = Path(args.query).expanduser().resolve()
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    qsha = sha256_file(query_path)
    width, height = get_image_size(query_path)
    query_ext = query_path.suffix.lower() or ".jpg"
    result_root = Path(args.output).expanduser().resolve() if args.output else (ROOT / "image_retrieval" / "results")
    run_dir = ensure_dir(result_root / f"{query_path.stem}_{qsha[:8]}")
    query_crop_dir = ensure_dir(run_dir / "query_crops")
    query_annotated_path = run_dir / f"query_annotated{query_ext}"
    result_json_path = run_dir / "result.json"
    last_result_json_path = result_root / "last_query_results.json"

    score_threshold = args.score_threshold if args.score_threshold is not None else 0.50
    margin_threshold = args.margin_threshold if args.margin_threshold is not None else settings.auto_accept_min_margin

    detector = UltralyticsDetector(
        settings.yolo_model_path,
        max_crops_per_image=args.max_crops or settings.max_crops_per_image,
    )
    indexer = QdrantClipIndexer(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_clip_collection,
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
        device=settings.device,
    )

    pipeline_errors: list[str] = []
    detections_out: list[dict[str, Any]] = []
    filtered_out_low_conf = 0

    try:
        raw_dets = detector.detect(query_path)
    except Exception as exc:
        output = {
            "query": {
                "query_image_path": str(query_path),
                "query_image_sha256": qsha,
                "width": width,
                "height": height,
                "processed_at": _utc_now(),
            },
            "pipeline": {
                "version": PIPELINE_VERSION,
                "detector_backend": "ultralytics",
                "retrieval_backend": "clip_qdrant",
                "status": "failed",
                "errors": [f"detector_error:{exc}"],
            },
            "thresholds": {
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
                "candidate_delta": args.candidate_delta,
                "top_k_requested": args.top_k,
                "yolo_min_confidence": args.yolo_min_confidence,
            },
            "summary": {
                "box_count": 0,
                "filtered_out_low_conf_count": 0,
                "matched_box_count": 0,
                "review_flag_count": 0,
                "variant_count": 0,
                "sku_counts": [],
            },
            "artifacts": {
                "query_annotated_image_path": str(query_annotated_path),
                "query_crop_dir": str(query_crop_dir),
                "result_json_path": str(result_json_path),
            },
            "detections": [],
        }
        atomic_write_json(result_json_path, output)
        atomic_write_json(last_result_json_path, output)
        print(json.dumps({"status": "failed", "result_json": str(result_json_path)}, indent=2))
        return 1

    for idx, pred in enumerate(raw_dets):
        pred_conf = float(pred.confidence or 0.0)
        if pred_conf < args.yolo_min_confidence:
            filtered_out_low_conf += 1
            continue
        bbox = _clip_bbox(pred.bbox, width, height)
        query_crop_id = f"{qsha}_{idx:03d}"
        crop_path = query_crop_dir / f"{query_crop_id}{query_ext}"

        det_entry: dict[str, Any] = {
            "query_crop_id": query_crop_id,
            "bbox": bbox,
            "yolo_confidence": pred_conf,
            "yolo_class": pred.class_name,
            "query_crop_path": str(crop_path),
            "retrieval": {
                "top1": None,
                "top_k_candidates": [],
                "raw_top1_score": None,
                "raw_top2_score": None,
                "margin": None,
                "review_flag": True,
                "review_reason": "not_processed",
                "raw_hits_considered": 0,
            },
            "assignment": {
                "sku": None,
                "name": None,
                "status": "unmatched_error",
            },
        }

        if not _bbox_valid(bbox, settings.min_box_size):
            det_entry["retrieval"]["review_reason"] = "invalid_bbox"
            detections_out.append(det_entry)
            continue

        try:
            crop_image(query_path, bbox, crop_path)
        except Exception as exc:
            det_entry["retrieval"]["review_reason"] = f"crop_error:{exc}"
            detections_out.append(det_entry)
            continue

        try:
            raw_limit = args.top_k if args.dedupe_by == "none" else max(args.top_k * 5, args.top_k)
            raw_hits = indexer.search_similar(crop_path, limit=raw_limit)
            deduped_hits = _dedupe_hits(raw_hits, args.dedupe_by, args.top_k, settings=settings)
            det_entry["retrieval"]["raw_hits_considered"] = len(raw_hits)

            top1 = deduped_hits[0] if deduped_hits else None
            top2 = deduped_hits[1] if len(deduped_hits) > 1 else None
            top1_score = float(top1["score"]) if top1 else None
            top2_score = float(top2["score"]) if top2 else None
            margin = (top1_score - top2_score) if (top1_score is not None and top2_score is not None) else None
            review_flag, review_reason = _review_flag(top1_score, margin, score_threshold, margin_threshold)

            det_entry["retrieval"]["raw_top1_score"] = top1_score
            det_entry["retrieval"]["raw_top2_score"] = top2_score
            det_entry["retrieval"]["margin"] = margin
            det_entry["retrieval"]["review_flag"] = review_flag
            det_entry["retrieval"]["review_reason"] = review_reason

            if not top1:
                det_entry["assignment"]["status"] = "unmatched_error"
                det_entry["retrieval"]["review_flag"] = True
                det_entry["retrieval"]["review_reason"] = "no_hits"
                detections_out.append(det_entry)
                continue

            # Build threshold-filtered top-k candidates (top1 always included).
            candidates_out: list[dict[str, Any]] = []
            for rank, hit in enumerate(deduped_hits, 1):
                score = float(hit.get("score", 0.0))
                if rank > 1 and top1_score is not None and (top1_score - score) > args.candidate_delta:
                    continue
                payload = dict(hit.get("payload") or {})
                artifacts = _resolve_match_artifacts(payload, settings)
                candidate_row = {
                    "rank": rank,
                    "score": score,
                    "sku": payload.get("sku"),
                    "name": artifacts.get("matched_name") or _readable_name_from_payload(payload),
                    "payload": payload,
                    "matched_artifacts": {
                        "matched_image_id": artifacts.get("matched_image_id"),
                        "matched_crop_id": artifacts.get("matched_crop_id"),
                        "matched_crop_path": artifacts.get("matched_crop_path"),
                        "matched_crop_json_path": artifacts.get("matched_crop_json_path"),
                        "matched_image_json_path": artifacts.get("matched_image_json_path"),
                        "matched_annotated_image_path": artifacts.get("matched_annotated_image_path"),
                    },
                }
                candidates_out.append(candidate_row)
                if len(candidates_out) >= args.top_k:
                    break

            if not candidates_out:
                # Defensive: top1 should always be present if hits exist.
                payload = dict(top1.get("payload") or {})
                artifacts = _resolve_match_artifacts(payload, settings)
                candidates_out = [{
                    "rank": 1,
                    "score": float(top1["score"]),
                    "sku": payload.get("sku"),
                    "name": artifacts.get("matched_name") or _readable_name_from_payload(payload),
                    "payload": payload,
                    "matched_artifacts": {
                        "matched_image_id": artifacts.get("matched_image_id"),
                        "matched_crop_id": artifacts.get("matched_crop_id"),
                        "matched_crop_path": artifacts.get("matched_crop_path"),
                        "matched_crop_json_path": artifacts.get("matched_crop_json_path"),
                        "matched_image_json_path": artifacts.get("matched_image_json_path"),
                        "matched_annotated_image_path": artifacts.get("matched_annotated_image_path"),
                    },
                }]

            top1_out = candidates_out[0]
            det_entry["retrieval"]["top1"] = top1_out
            det_entry["retrieval"]["top_k_candidates"] = candidates_out
            det_entry["assignment"]["sku"] = top1_out.get("sku")
            det_entry["assignment"]["name"] = top1_out.get("name")
            det_entry["assignment"]["status"] = "matched_review_flag" if review_flag else "matched"
            detections_out.append(det_entry)
        except Exception as exc:
            det_entry["retrieval"]["review_flag"] = True
            det_entry["retrieval"]["review_reason"] = f"retrieval_error:{exc}"
            det_entry["assignment"]["status"] = "unmatched_error"
            detections_out.append(det_entry)
            pipeline_errors.append(f"crop_{query_crop_id}_retrieval_error:{exc}")

    # Render annotated query image with assignment names/SKUs and review status.
    render_input = []
    for det in detections_out:
        label_value = det["assignment"].get("name") or det["assignment"].get("sku") or "NO-MATCH"
        status = "needs_review" if (det["retrieval"] or {}).get("review_flag") else "auto_accepted"
        render_input.append(
            {
                "bbox": det["bbox"],
                "assignment": {
                    "sku": label_value,
                    "label_status": status,
                },
            }
        )
    rendered, render_warning = render_labeled_image(query_path, render_input, query_annotated_path)
    if render_warning:
        pipeline_errors.append(render_warning)

    # Summary counts (both strict SKU and merged-by-name variant views)
    sku_counter: dict[str, int] = defaultdict(int)
    sku_names: dict[str, str | None] = {}
    variant_counter: dict[str, int] = defaultdict(int)
    variant_display_names: dict[str, str | None] = {}
    variant_sku_ids: dict[str, set[str]] = defaultdict(set)
    matched_box_count = 0
    review_flag_count = 0
    for det in detections_out:
        if det["retrieval"].get("review_flag"):
            review_flag_count += 1
        sku = det["assignment"].get("sku")
        assigned_name = det["assignment"].get("name")
        if sku:
            matched_box_count += 1
            sku_counter[str(sku)] += 1
            sku_names[str(sku)] = assigned_name

            variant_key = _normalize_variant_name(assigned_name) or str(sku)
            variant_counter[variant_key] += 1
            if assigned_name:
                variant_display_names[variant_key] = assigned_name
            variant_sku_ids[variant_key].add(str(sku))

    sku_counts = [
        {"sku": sku, "name": sku_names.get(sku), "count": count}
        for sku, count in sorted(sku_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    variant_counts = [
        {
            "variant_key": key,
            "name": variant_display_names.get(key),
            "count": count,
            "sku_ids": sorted(variant_sku_ids.get(key, set())),
        }
        for key, count in sorted(variant_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    if not raw_dets:
        pipeline_status = "no_boxes"
    elif pipeline_errors:
        pipeline_status = "partial_error"
    elif any(det["assignment"]["status"] == "unmatched_error" for det in detections_out):
        pipeline_status = "partial_error"
    else:
        pipeline_status = "ok"

    output = {
        "query": {
            "query_image_path": str(query_path),
            "query_image_sha256": qsha,
            "width": width,
            "height": height,
            "processed_at": _utc_now(),
        },
        "pipeline": {
            "version": PIPELINE_VERSION,
            "detector_backend": "ultralytics",
            "retrieval_backend": "clip_qdrant",
            "status": pipeline_status,
            "errors": pipeline_errors,
        },
        "thresholds": {
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "candidate_delta": args.candidate_delta,
            "top_k_requested": args.top_k,
            "dedupe_by": args.dedupe_by,
            "yolo_min_confidence": args.yolo_min_confidence,
        },
        "summary": {
            "box_count": len(detections_out),
            "filtered_out_low_conf_count": filtered_out_low_conf,
            "matched_box_count": matched_box_count,
            "review_flag_count": review_flag_count,
            "variant_count": len(variant_counter),
            "variant_counts": variant_counts,
            "sku_counts": sku_counts,
        },
        "artifacts": {
            "query_annotated_image_path": str(query_annotated_path),
            "query_crop_dir": str(query_crop_dir),
            "result_json_path": str(result_json_path),
            "rendered_query_annotation": rendered,
        },
        "detections": detections_out,
    }

    atomic_write_json(result_json_path, output)
    atomic_write_json(last_result_json_path, output)

    print(f"Query image: {query_path}")
    print(f"Status: {pipeline_status}")
    print(f"Boxes: {len(detections_out)}")
    print(f"Variants: {len(variant_counter)}")
    print(f"Result JSON: {result_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
