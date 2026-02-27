from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    yolo_weights_path: str = Field(..., alias="YOLO_WEIGHTS_PATH")
    yolo_conf_threshold: float = Field(0.25, alias="YOLO_CONF_THRESHOLD")
    yolo_iou_threshold: float = Field(0.45, alias="YOLO_IOU_THRESHOLD")

    clip_model_name: str = Field("ViT-B-32", alias="CLIP_MODEL_NAME")
    clip_pretrained: str = Field("openai", alias="CLIP_PRETRAINED")

    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        ...,
        validation_alias=AliasChoices("QDRANT_CLIP_COLLECTION", "QDRANT_COLLECTION"),
    )
    qdrant_vector_name: Optional[str] = Field(None, alias="QDRANT_VECTOR_NAME")
    qdrant_sku_collection: Optional[str] = Field(None, alias="QDRANT_SKU_COLLECTION")
    sku_uuid_field: str = Field("metadata.uuid", alias="SKU_UUID_FIELD")

    match_score_threshold: float = Field(0.30, alias="MATCH_SCORE_THRESHOLD")
    top_k: int = Field(3, alias="TOP_K")

    device: str = Field("cpu", alias="DEVICE")
    max_image_size: int = Field(4096, alias="MAX_IMAGE_SIZE")

    @property
    def qdrant_clip_collection(self) -> str:
        return self.qdrant_collection


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
