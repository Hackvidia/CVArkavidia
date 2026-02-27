from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


class FakePipeline:
    def run(self, image):
        return {
            "total_detections": 2,
            "matched_detections": 2,
            "unique_items": 1,
            "counts": {"box": 2},
            "items": [
                {
                    "name": "box",
                    "score": 0.91,
                    "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                    "yolo_conf": 0.88,
                },
                {
                    "name": "box",
                    "score": 0.90,
                    "bbox": {"x1": 20, "y1": 20, "x2": 40, "y2": 40},
                    "yolo_conf": 0.87,
                },
            ],
        }

    def run_with_debug(self, image):
        return self.run(image), {
            "matches": [
                {
                    "detection_index": 0,
                    "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                    "yolo_conf": 0.88,
                    "accepted": True,
                    "reason": "accepted",
                    "threshold": 0.3,
                    "top_candidates": [{"name": "box", "score": 0.91}],
                }
            ]
        }


client = TestClient(app)
app.state.services.pipeline = FakePipeline()
app.state.services.detection_loaded = True
app.state.services.embedding_loaded = True
app.state.services.retrieval_loaded = True


def _png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_search_contract_returns_expected_keys():
    response = client.post(
        "/search",
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "total_detections",
        "matched_detections",
        "unique_items",
        "counts",
        "items",
    }
    assert body["counts"] == {"box": 2}


def test_search_rejects_invalid_image_payload():
    response = client.post(
        "/search",
        files={"image": ("input.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_IMAGE"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded", "starting"}
    assert "yolo_loaded" in body
    assert "clip_loaded" in body
    assert "qdrant_connected" in body


def test_search_debug_includes_matching_visibility():
    response = client.post(
        "/search/debug",
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "debug" in body
    assert "matches" in body["debug"]
