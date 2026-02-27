# Image Retrieval API (YOLO + CLIP + Qdrant)

FastAPI service for reverse image retrieval on detected objects:
- Detect items with custom YOLO weights.
- Crop each detection.
- Embed each crop using CLIP.
- Query Qdrant for nearest matches.
- Return deduplicated item counts by `payload.name` and per-box matches.

## API

### `GET /health`
Returns service readiness:
- `status`: `ok | degraded | starting`
- `yolo_loaded`, `clip_loaded`, `qdrant_connected`

### `POST /search`
`multipart/form-data` with field `image`.

Response:
```json
{
  "total_detections": 3,
  "matched_detections": 2,
  "unique_items": 1,
  "counts": {"box": 2},
  "items": [
    {
      "name": "box",
      "score": 0.93,
      "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 150},
      "yolo_conf": 0.88
    }
  ]
}
```

Unmatched detections are dropped from `counts` and `items`.

### `POST /search/debug`
Same input as `/search`, but includes a `debug.matches` array per detection with:
- `accepted` and `reason` (`accepted`, `below_threshold`, `missing_payload_name`, `no_candidates`)
- applied `threshold`
- `top_candidates` from Qdrant (name + score)
- detection metadata (`bbox`, `yolo_conf`, `detection_index`)

## Configuration
Copy `.env.example` to `.env` and set values:
- `YOLO_WEIGHTS_PATH` (required)
- `QDRANT_URL` (required)
- `QDRANT_COLLECTION` (required)
- `QDRANT_SKU_COLLECTION` (optional, second collection for SKU metadata lookup by UUID, e.g. `skus`)
- `SKU_UUID_FIELD` (default `sku`, fallback field used when nearest payload does not have `sku`)
- `MATCH_SCORE_THRESHOLD` (default `0.30`)
- `DEVICE=cpu` (CPU-only runtime)

Important assumptions:
- Qdrant collection is pre-indexed.
- When `QDRANT_SKU_COLLECTION` is set, top CLIP hit reads `payload.sku` first, then fetches canonical SKU payload/name from that collection by ID.
- Payload includes canonical name at `payload.name`.
- Embedding dimension matches CLIP model output.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`requirements.txt` is pinned to CPU-only PyTorch wheels to avoid CUDA downloads during install/build.

## Tests

```bash
pytest -q
```
