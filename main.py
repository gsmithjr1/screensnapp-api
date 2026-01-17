# main.py
import os
import base64
from typing import Optional, List

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

# Clarifai gRPC
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="ScreenSnapp API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Helpers
# ----------------------------
def require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise HTTPException(status_code=500, detail=f"Server misconfigured: {name} is missing")
    return val


def get_optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


def require_bearer(authorization: Optional[str] = Header(default=None)):
    """
    If API_BEARER_TOKEN is set in Railway, require:
      Authorization: Bearer <token>
    If API_BEARER_TOKEN is empty, auth is skipped (dev-friendly).
    """
    expected = get_optional_env("API_BEARER_TOKEN")
    if not expected:
        return True

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    got = authorization.split(" ", 1)[1].strip()
    if got != expected:
        raise HTTPException(status_code=401, detail="Invalid Bearer token")

    return True


def clarifai_ocr_extract_text(image_bytes: bytes) -> str:
    """
    Runs Clarifai OCR model and extracts text.
    Required env:
      CLARIFAI_PAT
      CLARIFAI_USER_ID
      CLARIFAI_APP_ID
      CLARIFAI_OCR_MODEL_ID
    Optional env:
      CLARIFAI_OCR_MODEL_VERSION_ID  (if you want to pin a version)
    """
    pat = require_env("CLARIFAI_PAT")
    user_id = require_env("CLARIFAI_USER_ID")
    app_id = require_env("CLARIFAI_APP_ID")
    model_id = require_env("CLARIFAI_OCR_MODEL_ID")
    model_version_id = get_optional_env("CLARIFAI_OCR_MODEL_VERSION_ID")

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (("authorization", f"Key {pat}"),)

    image = resources_pb2.Image(base64=base64.b64encode(image_bytes))
    input_ = resources_pb2.Input(data=resources_pb2.Data(image=image))

    if model_version_id:
        req = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            version_id=model_version_id,
            inputs=[input_],
        )
    else:
        req = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[input_],
        )

    resp = stub.PostModelOutputs(req, metadata=metadata)

    if resp.status.code != status_code_pb2.SUCCESS:
        # Clarifai error info
        raise HTTPException(
            status_code=502,
            detail=f"Clarifai error: {resp.status.code} - {resp.status.description}",
        )

    # OCR models often return text in regions -> data.text.raw / data.regions[].data.text.raw
    output = resp.outputs[0]
    data = output.data

    texts: List[str] = []

    # Some models fill data.text.raw
    if getattr(data, "text", None) and getattr(data.text, "raw", ""):
        texts.append(data.text.raw)

    # Many OCR models fill regions with per-region text
    for region in data.regions:
        try:
            raw = region.data.text.raw
            if raw:
                texts.append(raw)
        except Exception:
            pass

    combined = " ".join([t.strip() for t in texts if t and t.strip()])
    return combined.strip()


def tmdb_search_multi(query: str) -> dict:
    """
    Uses TMDB v3 API Key.
    Required env:
      TMDB_API_KEY
    """
    api_key = require_env("TMDB_API_KEY")

    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": api_key,
        "query": query,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error: {r.status_code} {r.text[:200]}")
    return r.json()


def choose_best_tmdb_result(tmdb_json: dict) -> Optional[dict]:
    """
    Picks the top result (you can improve ranking later).
    """
    results = tmdb_json.get("results", []) or []
    if not results:
        return None

    # Prefer movie/tv over people
    def score(item: dict) -> int:
        media_type = item.get("media_type")
        base = 0
        if media_type == "movie":
            base += 30
        elif media_type == "tv":
            base += 25
        else:
            base -= 10

        # Popularity + vote_count helps a bit
        base += int(item.get("popularity", 0) or 0)
        base += int((item.get("vote_count", 0) or 0) / 10)
        return base

    results.sort(key=score, reverse=True)
    return results[0]


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/identify-screen")
async def identify_screen(
    file: UploadFile = File(...),
    _auth_ok: bool = Depends(require_bearer),
):
    """
    Accepts multipart/form-data with field name "file"
    Returns:
      - extracted_text
      - tmdb_best_match (if found)
      - tmdb_results_count
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Missing file")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # 1) OCR via Clarifai
        extracted_text = clarifai_ocr_extract_text(content)

        if not extracted_text:
            return {
                "status": "ok",
                "extracted_text": "",
                "tmdb_query": "",
                "tmdb_results_count": 0,
                "tmdb_best_match": None,
                "message": "No text detected. Try getting the title more centered/clear.",
            }

        # Keep the query reasonable (TMDB doesn't need giant text blobs)
        tmdb_query = extracted_text.strip()
        if len(tmdb_query) > 120:
            tmdb_query = tmdb_query[:120]

        # 2) TMDB search
        tmdb_json = tmdb_search_multi(tmdb_query)
        best = choose_best_tmdb_result(tmdb_json)

        return {
            "status": "ok",
            "extracted_text": extracted_text,
            "tmdb_query": tmdb_query,
            "tmdb_results_count": len(tmdb_json.get("results", []) or []),
            "tmdb_best_match": best,
            "tmdb_raw": tmdb_json,  # you can remove later if you want smaller responses
        }

    except HTTPException:
        raise
    except Exception as e:
        # Never crash the server; return a clean error
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
