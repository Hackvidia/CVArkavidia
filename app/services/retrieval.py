from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.services.qdrant_clip_indexer import QdrantClipIndexer


@dataclass
class RetrievalResult:
    name: str
    score: float
    sku_uuid: Optional[str] = None


@dataclass
class CandidateResult:
    name: Optional[str]
    score: float


@dataclass
class RetrievalDebug:
    accepted: bool
    reason: str
    threshold: float
    top_candidates: list[CandidateResult]
    selected: Optional[RetrievalResult]


class RetrievalService:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_collection: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_vector_name: Optional[str] = None,
        qdrant_sku_collection: Optional[str] = None,
        sku_uuid_field: str = "metadata.uuid",
        score_threshold: float = 0.30,
        top_k: int = 3,
        indexer: Optional[QdrantClipIndexer] = None,
    ):
        self.indexer = indexer
        self.client = indexer.client if indexer is not None else QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection = indexer.collection_name if indexer is not None else qdrant_collection
        self.vector_name = qdrant_vector_name
        self.sku_collection = qdrant_sku_collection
        self.sku_uuid_field = sku_uuid_field
        self.score_threshold = score_threshold
        self.top_k = top_k

    @staticmethod
    def _readable_name_from_payload(payload: dict[str, Any]) -> Optional[str]:
        direct_keys = ("display_name", "name", "title", "product_name", "sku_name")
        for key in direct_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in direct_keys:
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _readable_uuid(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _value_by_path(payload: dict[str, Any], path: str) -> Any:
        current: Any = payload
        for key in (path or "").split("."):
            if not key:
                return None
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _extract_sku_uuid(self, payload: dict[str, Any]) -> Optional[str]:
        # Primary source is nearest-vector payload.sku, then configured/fallback fields.
        candidate_paths = ["sku", self.sku_uuid_field, "uuid", "metadata.sku", "metadata.sku_uuid", "metadata.uuid"]
        for path in candidate_paths:
            raw = self._value_by_path(payload, path)
            value = self._readable_uuid(raw)
            if value:
                return value
        return None

    def _lookup_sku_payload(self, sku_uuid: str) -> Optional[dict[str, Any]]:
        if not self.sku_collection:
            return None

        # 1) Try direct ID retrieval.
        try:
            points = self.client.retrieve(
                collection_name=self.sku_collection,
                ids=[sku_uuid],
                with_payload=True,
            )
            if points and getattr(points[0], "payload", None):
                return dict(points[0].payload or {})
        except Exception:
            pass

        # 2) Fallback to filtering by configured UUID field.
        try:
            points, _ = self.client.scroll(
                collection_name=self.sku_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=self.sku_uuid_field,
                            match=models.MatchValue(value=sku_uuid),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points and getattr(points[0], "payload", None):
                return dict(points[0].payload or {})
        except Exception:
            pass

        return None

    def healthcheck(self) -> bool:
        if self.indexer is not None:
            clip_ok = self.indexer.healthcheck()
        else:
            try:
                self.client.get_collection(self.collection)
                clip_ok = True
            except Exception:
                clip_ok = False

        if not clip_ok:
            return False

        if self.sku_collection:
            try:
                self.client.get_collection(self.sku_collection)
            except Exception:
                return False
        return True

    def match(self, embedding: list[float]) -> Optional[RetrievalResult]:
        debug = self.match_with_debug(embedding)
        return debug.selected

    def match_with_debug(self, embedding: list[float]) -> RetrievalDebug:
        if self.indexer is not None:
            points = self.indexer.search_vector(
                vector=embedding,
                limit=self.top_k,
                vector_name=self.vector_name,
            )
        else:
            if self.vector_name:
                query_vector = models.NamedVector(name=self.vector_name, vector=embedding)
            else:
                query_vector = embedding
            points = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=self.top_k,
                with_payload=True,
            )

        if not points:
            return RetrievalDebug(
                accepted=False,
                reason="no_candidates",
                threshold=self.score_threshold,
                top_candidates=[],
                selected=None,
            )

        candidates: list[CandidateResult] = []
        for point in points:
            payload = point.payload or {}
            name = self._readable_name_from_payload(payload)
            candidates.append(CandidateResult(name=name, score=float(point.score)))

        top = points[0]
        if float(top.score) < self.score_threshold:
            return RetrievalDebug(
                accepted=False,
                reason="below_threshold",
                threshold=self.score_threshold,
                top_candidates=candidates,
                selected=None,
            )

        payload = top.payload or {}
        sku_uuid = self._extract_sku_uuid(payload)
        sku_payload = self._lookup_sku_payload(sku_uuid) if sku_uuid else None
        name = self._readable_name_from_payload(sku_payload or payload)
        if not name:
            return RetrievalDebug(
                accepted=False,
                reason="missing_payload_name",
                threshold=self.score_threshold,
                top_candidates=candidates,
                selected=None,
            )

        selected = RetrievalResult(name=name, score=float(top.score), sku_uuid=sku_uuid)
        return RetrievalDebug(
            accepted=True,
            reason="accepted",
            threshold=self.score_threshold,
            top_candidates=candidates,
            selected=selected,
        )
