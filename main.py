import os
import base64
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# Load env vars (local dev). On Railway, env vars come from Variables tab.
load_dotenv()

# ----------------------------
# ENV / CONFIG (safe reads)
# ----------------------------
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT", "").strip()
CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID", "clarifai").strip()
CLARIFAI_APP_ID = os.getenv("CLARIFAI_APP_ID", "main").strip()

# Concepts model (general image recognition)
CLARIFAI_MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "general-image-recognition").strip()
CLARIFAI_MODEL_VERSION_ID = os.getenv("CLARIFAI_MODEL_VERSION_ID", "").strip()

# OCR model (your custom / chosen OCR)
CLARIFAI_OCR_MODEL_ID = os.getenv("CLARIFAI_OCR_MODEL_ID", "").strip()
CLARIFAI_OCR_MODEL_VERSION_ID = os.getenv("CLARIFAI_OCR_MODEL_VERSION_ID", "").strip()

# Protect API (optional but recommended)
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

# TMDB
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="ScreenSnapp API",
    description="Clarifai Concepts + OCR + TMDB matching (Bearer protected).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer(auto_error=False)


def require_env(name: str) -> str:
    """
    Return env var value or throw a 500 (without crashing the server).
    """
    val = os.getenv(name, "").strip()
    if not val:
        raise HTTPException(status_code=500, detail=f"Server misconfigured: {name} is missing")
    return val


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Bearer auth. If API_BEARER_TOKEN is blank, we allow requests (useful for debugging).
    If you WANT auth enforced, set API_BEARER_TOKEN in Railway.
    """
    expected = os.getenv("API_BEARER_TOKEN", "").strip()
    if not expected:
        return "auth-disabled"

    if credentials is None or not (credentials.credentials or "").strip():
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    incoming = (credentials.credentials or "").strip()
    if incoming != expected:
        raise HTTPException(status_code=401, detail="Invalid Bearer token")

    return incoming


# ----------------------------
# Clarifai gRPC helpers
# ----------------------------
def _clarifai_stub_and_meta():
    pat = require_env("CLARIFAI_PAT")
    user_id = os.getenv("CLARIFAI_USER_ID", "clarifai").strip()
    app_id = os.getenv("CLARIFAI_APP_ID", "main").strip()

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", "Key " + pat),)
    user_data = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

    return stub, metadata, user_data


def _post_clarifai_model(image_bytes: bytes, model_id: str, version_id: str = ""):
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data")

    stub, metadata, user_data = _clarifai_stub_and_meta()

    # Clarifai expects base64-encoded bytes in the "base64" field
    b64 = base64.b64encode(image_bytes)

    req = service_pb2.PostModelOutputsRequest(
        user_app_id=user_data,
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=b64)
                )
            )
        ],
    )

    if version_id:
        req.version_id = version_id

    resp = stub.PostModelOutputs(req, metadata=metadata)

    if resp.status.code != status_code_pb2.SUCCESS:
        raise HTTPException(
            status_code=500,
            detail=f"Clarifai error: {resp.status.description}",
        )

    return resp


def call_clarifai_concepts(image_bytes: bytes):
    model_id = os.getenv("CLARIFAI_MODEL_ID", "general-image-recognition").strip()
    version_id = os.getenv("CLARIFAI_MODEL_VERSION_ID", "").strip()

    resp = _post_clarifai_model(image_bytes, model_id, version_id)

    concepts = resp.outputs[0].data.concepts if resp.outputs else []
    return [{"name": c.name, "confidence": round(float(c.value), 4)} for c in concepts]


def call_clarifai_ocr(image_bytes: bytes):
    ocr_model_id = require_env("CLARIFAI_OCR_MODEL_ID")
    ocr_version_id = os.getenv("CLARIFAI_OCR_MODEL_VERSION_ID", "").strip()

    resp = _post_clarifai_model(image_bytes, ocr_model_id, ocr_version_id)

    texts = []
    try:
        out = resp.outputs[0]

        # some models put OCR here
        if out.data.text and out.data.text.raw:
            texts.append(out.data.text.raw)

        # some put OCR inside regions
        for r in out.data.regions:
            if r.data.text and r.data.text.raw:
                texts.append(r.data.text.raw)

    except Exception:
        pass

    texts = [t.strip() for t in texts if t and t.strip()]

    # unique preserve order
    unique = []
    for t in texts:
        if t not in unique:
            unique.append(t)

    combined = " ".join(unique)
    return {"tokens": unique[:50], "combined": combined[:5000]}


# ----------------------------
# TMDB helper
# ----------------------------
def tmdb_search_best(query: str):
    tmdb_key = require_env("TMDB_API_KEY")
    q = (query or "").strip()
    if len(q) < 3:
        return None

    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": tmdb_key,
        "query": q,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }

    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    filtered = [x for x in results if x.get("media_type") in ("movie", "tv")]
    if not filtered:
        return None

    best = filtered[0]
    media_type = best.get("media_type")
    tmdb_id = best.get("id")

    title = best.get("title") if media_type == "movie" else best.get("name")
    year = ""
    if media_type == "movie":
        year = (best.get("release_date") or "")[:4]
    else:
        year = (best.get("first_air_date") or "")[:4]

    fandango_url = f"https://www.fandango.com/search?q={requests.utils.quote(title or '')}"

    return {
        "tmdb_id": tmdb_id,
        "media_type": media_type,
        "title": title,
        "year": year,
        "overview": best.get("overview"),
        "poster_path": best.get("poster_path"),
        "tmdb_score": best.get("vote_average"),
        "fandango_search_url": fandango_url,
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
async def health_check():
    # do NOT require_env here; health should work even when misconfigured
    return {
        "status": "ok",
        "auth_enabled": bool(os.getenv("API_BEARER_TOKEN", "").strip()),
        "clarifai_pat_set": bool(os.getenv("CLARIFAI_PAT", "").strip()),
        "clarifai_user_id": os.getenv("CLARIFAI_USER_ID", "clarifai").strip(),
        "clarifai_app_id": os.getenv("CLARIFAI_APP_ID", "main").strip(),
        "concepts_model_id": os.getenv("CLARIFAI_MODEL_ID", "general-image-recognition").strip(),
        "ocr_model_id_set": bool(os.getenv("CLARIFAI_OCR_MODEL_ID", "").strip()),
        "tmdb_key_set": bool(os.getenv("TMDB_API_KEY", "").strip()),
    }


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    token: str = Depends(verify_token),
):
    try:
        image_bytes = await file.read()
        predictions = call_clarifai_concepts(image_bytes)
        return {"predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


class ImageURL(BaseModel):
    url: str


@app.post("/analyze-image-url")
async def analyze_image_url(
    image_url: ImageURL,
    token: str = Depends(verify_token),
):
    try:
        r = requests.get(image_url.url, timeout=15)
        r.raise_for_status()
        image_bytes = r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image URL: {e}")

    try:
        predictions = call_clarifai_concepts(image_bytes)
        return {"predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.post("/identify-screen")
async def identify_screen(
    file: UploadFile = File(...),
    token: str = Depends(verify_token),
):
    """
    1) Concepts model (general image tags)
    2) OCR model (extracts text from screen)
    3) TMDB search using OCR combined text
    """
    try:
        image_bytes = await file.read()

        concepts = call_clarifai_concepts(image_bytes)
        ocr = call_clarifai_ocr(image_bytes)
        match = tmdb_search_best(ocr.get("combined", ""))

        return {"concepts": concepts, "ocr": ocr, "match": match}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")



