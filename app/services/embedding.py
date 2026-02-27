from typing import List

from PIL import Image

from app.services.qdrant_clip_indexer import QdrantClipIndexer


class EmbeddingService:
    def __init__(self, indexer: QdrantClipIndexer):
        self.indexer = indexer

    def embed_image(self, image: Image.Image) -> List[float]:
        return self.indexer.embed_image(image)
