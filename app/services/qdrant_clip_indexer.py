from __future__ import annotations

from typing import Any, List, Optional

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantClipIndexer:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: Optional[str],
        collection_name: str,
        model_name: str,
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        import open_clip

        resolved_model = self._normalize_model_name(model_name)
        resolved_device = self._normalize_device(device)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=resolved_model,
            pretrained=pretrained,
            device=resolved_device,
        )
        self.model.eval()
        self.device = resolved_device

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        normalized = (model_name or "").strip()
        if normalized.lower().startswith("clip-"):
            normalized = normalized[5:]
        return normalized.replace("/", "-")

    @staticmethod
    def _normalize_device(device: str) -> str:
        import torch

        requested = (device or "cpu").strip().lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def embed_image(self, image: Image.Image) -> List[float]:
        import torch

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().float().tolist()

    def search_vector(self, vector: list[float], limit: int, vector_name: Optional[str] = None) -> list[Any]:
        if vector_name:
            query_vector: list[float] | models.NamedVector = models.NamedVector(
                name=vector_name,
                vector=vector,
            )
        else:
            query_vector = vector

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

    def healthcheck(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False
