from dataclasses import dataclass

from app.services.pipeline import ImageRetrievalPipeline


@dataclass
class FakeDetection:
    bbox: tuple[int, int, int, int]
    yolo_conf: float
    crop: str


@dataclass
class FakeMatch:
    name: str
    score: float


class DetectionStub:
    def detect_and_crop(self, image):
        return [
            FakeDetection((0, 0, 10, 10), 0.9, "crop1"),
            FakeDetection((11, 11, 20, 20), 0.8, "crop2"),
            FakeDetection((21, 21, 30, 30), 0.7, "crop3"),
        ]


class EmbeddingStub:
    def embed_image(self, crop):
        return [float(len(str(crop)))]


class RetrievalStub:
    def __init__(self):
        self.calls = 0

    def match(self, embedding):
        self.calls += 1
        if self.calls in (1, 2):
            return FakeMatch(name="apple", score=0.77)
        return FakeMatch(name="orange", score=0.71)


def test_pipeline_dedupes_counts_by_name():
    pipeline = ImageRetrievalPipeline(DetectionStub(), EmbeddingStub(), RetrievalStub())
    result = pipeline.run(image=None)

    assert result.total_detections == 3
    assert result.matched_detections == 3
    assert result.unique_items == 2
    assert result.counts == {"apple": 2, "orange": 1}
    assert len(result.items) == 3
