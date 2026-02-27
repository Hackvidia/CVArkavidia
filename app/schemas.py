from typing import Dict, List

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class MatchedItem(BaseModel):
    name: str
    score: float
    sku_uuid: str | None = None
    bbox: BoundingBox
    yolo_conf: float


class SearchResponse(BaseModel):
    total_detections: int
    matched_detections: int
    unique_items: int
    counts: Dict[str, int]
    items: List[MatchedItem]


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    clip_loaded: bool
    qdrant_connected: bool
