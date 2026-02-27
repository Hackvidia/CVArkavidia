from app.services.retrieval import RetrievalService


class Point:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class ClientStub:
    def __init__(self, points):
        self._points = points

    def search(self, **kwargs):
        return self._points

    def get_collection(self, name):
        return {"name": name}

    def retrieve(self, **kwargs):
        return []

    def scroll(self, **kwargs):
        return ([], None)


def test_match_returns_none_when_score_below_threshold():
    service = RetrievalService(
        qdrant_url="http://localhost:6333",
        qdrant_collection="test",
        score_threshold=0.8,
    )
    service.client = ClientStub([Point(0.79, {"name": "box"})])

    assert service.match([0.1, 0.2]) is None


def test_match_returns_none_when_payload_name_missing():
    service = RetrievalService(
        qdrant_url="http://localhost:6333",
        qdrant_collection="test",
        score_threshold=0.5,
    )
    service.client = ClientStub([Point(0.95, {"label": "box"})])

    assert service.match([0.1, 0.2]) is None


def test_match_reads_name_from_nested_metadata():
    service = RetrievalService(
        qdrant_url="http://localhost:6333",
        qdrant_collection="test",
        score_threshold=0.5,
    )
    service.client = ClientStub([Point(0.95, {"metadata": {"name": "box"}})])

    result = service.match([0.1, 0.2])
    assert result is not None
    assert result.name == "box"


class SkuClientStub(ClientStub):
    def retrieve(self, **kwargs):
        sku_uuid = kwargs.get("ids", [None])[0]
        if sku_uuid == "sku-uuid-123":
            return [Point(1.0, {"metadata": {"display_name": "Tropical Minyak Goreng Pouch 1L"}})]
        return []


def test_match_prefers_name_from_sku_collection_lookup():
    service = RetrievalService(
        qdrant_url="http://localhost:6333",
        qdrant_collection="clip_collection",
        qdrant_sku_collection="skus",
        sku_uuid_field="sku",
        score_threshold=0.5,
    )
    service.client = SkuClientStub([Point(0.93, {"name": "clip-name", "sku": "sku-uuid-123"})])

    result = service.match([0.1, 0.2])
    assert result is not None
    assert result.name == "Tropical Minyak Goreng Pouch 1L"
    assert result.sku_uuid == "sku-uuid-123"
