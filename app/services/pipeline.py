from collections import Counter
from typing import Any, Dict, List

from app.schemas import BoundingBox, MatchedItem, SearchResponse
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.services.retrieval import RetrievalService


class ImageRetrievalPipeline:
    def __init__(
        self,
        detection_service: DetectionService,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
    ):
        self.detection = detection_service
        self.embedding = embedding_service
        self.retrieval = retrieval_service

    def run(self, image) -> SearchResponse:
        result, _ = self.run_with_debug(image)
        return result

    def run_with_debug(self, image) -> tuple[SearchResponse, dict[str, Any]]:
        detections = self.detection.detect_and_crop(image)

        matched_items: List[MatchedItem] = []
        counter: Dict[str, int] = Counter()
        debug_matches: list[dict[str, Any]] = []

        for idx, detection in enumerate(detections):
            vector = self.embedding.embed_image(detection.crop)
            if hasattr(self.retrieval, "match_with_debug"):
                debug = self.retrieval.match_with_debug(vector)
                match = debug.selected
                accepted = debug.accepted
                reason = debug.reason
                threshold = debug.threshold
                top_candidates = debug.top_candidates
            else:
                match = self.retrieval.match(vector)
                accepted = match is not None
                reason = "accepted" if accepted else "no_match"
                threshold = None
                top_candidates = []

            x1, y1, x2, y2 = detection.bbox
            debug_entry = {
                "detection_index": idx,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "yolo_conf": detection.yolo_conf,
                "accepted": accepted,
                "reason": reason,
                "threshold": threshold,
                "top_candidates": [
                    {"name": c.name, "score": c.score} for c in top_candidates
                ],
            }
            debug_matches.append(debug_entry)

            if match is None:
                continue

            counter[match.name] += 1
            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            matched_items.append(
                MatchedItem(
                    name=match.name,
                    score=match.score,
                    sku_uuid=getattr(match, "sku_uuid", None),
                    bbox=bbox,
                    yolo_conf=detection.yolo_conf,
                )
            )

        response = SearchResponse(
            total_detections=len(detections),
            matched_detections=len(matched_items),
            unique_items=len(counter),
            counts=dict(counter),
            items=matched_items,
        )
        return response, {"matches": debug_matches}
