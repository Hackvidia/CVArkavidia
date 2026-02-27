from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import Settings, get_settings
from app.schemas import HealthResponse, SearchResponse
from app.services.detection import DetectionService
from app.services.embedding import EmbeddingService
from app.services.pipeline import ImageRetrievalPipeline
from app.services.qdrant_clip_indexer import QdrantClipIndexer
from app.services.retrieval import RetrievalService


@dataclass
class ServiceState:
    pipeline: Optional[ImageRetrievalPipeline] = None
    detection_loaded: bool = False
    embedding_loaded: bool = False
    retrieval_loaded: bool = False
    startup_error: Optional[str] = None


def build_pipeline(settings: Settings) -> ImageRetrievalPipeline:
    detection_service = DetectionService(
        weights_path=settings.yolo_weights_path,
        conf_threshold=settings.yolo_conf_threshold,
        iou_threshold=settings.yolo_iou_threshold,
    )
    indexer = QdrantClipIndexer(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_clip_collection,
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
        device=settings.device,
    )
    embedding_service = EmbeddingService(indexer=indexer)
    retrieval_service = RetrievalService(
        indexer=indexer,
        qdrant_vector_name=settings.qdrant_vector_name,
        qdrant_sku_collection=settings.qdrant_sku_collection,
        sku_uuid_field=settings.sku_uuid_field,
        score_threshold=settings.match_score_threshold,
        top_k=settings.top_k,
    )
    return ImageRetrievalPipeline(detection_service, embedding_service, retrieval_service)


def create_app() -> FastAPI:
    app = FastAPI(title="Image Retrieval API", version="0.1.0")
    app.state.services = ServiceState()

    @app.on_event("startup")
    def startup_event() -> None:
        state: ServiceState = app.state.services
        try:
            settings = get_settings()
            state.pipeline = build_pipeline(settings)
            state.detection_loaded = True
            state.embedding_loaded = True
            state.retrieval_loaded = state.pipeline.retrieval.healthcheck()
            if not state.retrieval_loaded:
                state.startup_error = "Qdrant collection is not reachable."
        except Exception as exc:
            state.startup_error = str(exc)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        state: ServiceState = app.state.services
        if state.pipeline and state.retrieval_loaded:
            status = "ok"
        elif state.startup_error:
            status = "degraded"
        else:
            status = "starting"

        return HealthResponse(
            status=status,
            yolo_loaded=state.detection_loaded,
            clip_loaded=state.embedding_loaded,
            qdrant_connected=state.retrieval_loaded,
        )

    async def parse_upload_image(image: UploadFile, settings: Settings) -> Image.Image:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail={"code": "EMPTY_IMAGE", "detail": "Uploaded file is empty."},
            )

        try:
            image_obj = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_IMAGE", "detail": "Unsupported or invalid image format."},
            )

        width, height = image_obj.size
        if width > settings.max_image_size or height > settings.max_image_size:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "IMAGE_TOO_LARGE",
                    "detail": f"Image dimensions exceed max allowed size {settings.max_image_size}.",
                },
            )
        return image_obj

    @app.post("/search", response_model=SearchResponse)
    async def search(image: UploadFile = File(...)) -> SearchResponse:
        state: ServiceState = app.state.services
        if state.pipeline is None:
            detail = state.startup_error or "Service is not initialized"
            raise HTTPException(status_code=500, detail={"code": "INIT_FAILED", "detail": detail})

        settings = get_settings()
        image_obj = await parse_upload_image(image, settings)

        try:
            return state.pipeline.run(image_obj)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "PIPELINE_FAILED", "detail": str(exc)},
            )

    @app.post("/search/debug")
    async def search_debug(image: UploadFile = File(...)) -> dict:
        state: ServiceState = app.state.services
        if state.pipeline is None:
            detail = state.startup_error or "Service is not initialized"
            raise HTTPException(status_code=500, detail={"code": "INIT_FAILED", "detail": detail})

        settings = get_settings()
        image_obj = await parse_upload_image(image, settings)

        try:
            response, debug = state.pipeline.run_with_debug(image_obj)
            payload = response.model_dump()
            payload["debug"] = debug
            return payload
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "PIPELINE_FAILED", "detail": str(exc)},
            )

    return app


app = create_app()
