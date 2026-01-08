import os
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

print("=== ENV CHECK ===")
print("CLARIFAI_PAT exists:", bool(os.getenv("CLARIFAI_PAT")))
print("CLARIFAI_USER_ID:", os.getenv("CLARIFAI_USER_ID"))
print("CLARIFAI_APP_ID:", os.getenv("CLARIFAI_APP_ID"))
print("=================")

app = FastAPI()

@app.post("/identify-screen")
async def identify_screen(file: UploadFile = File(...)):
    ...


# Load env vars (local dev). On Railway, env vars come from Variables tab.
load_dotenv()

# ----------------------------
# ENV / CONFIG
# ----------------------------
PAT = os.getenv("CLARIFAI_PAT", "").strip()

# Public Clarifai defaults (safe + common)
USER_ID = os.getenv("CLARIFAI_USER_ID", "clarifai").strip()
APP_ID = os.getenv("CLARIFAI_APP_ID", "main").strip()

# Concepts model (general recognition)
MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "general-image-recognition").strip()
MODEL_VERSION_ID = os.getenv("CLARIFAI_MODEL_VERSION_ID", "").strip()

# Your API protection
BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

# TMDB
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY is missing")


# OCR model (you must set these in Railway Variables)
OCR_MODEL_ID = os.getenv("CLARIFAI_OCR_MODEL_ID", "").strip()
OCR_MODEL_VERSION_ID = os.getenv("CLARIFAI_OCR_MODEL_VERSION_ID", "").strip()

# Fail fast so Railway logs show exactly what's missing
if not PAT:
    raise RuntimeError("CLARIFAI_PAT is not set (Railway Variables → CLARIFAI_PAT).")
if not BEARER_TOKEN:
    raise RuntimeError("API_BEARER_TOKEN is not set (Railway Variables → API_BEARER_TOKEN).")
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY is not set (Railway Variables → TMDB_API_KEY).")
if not OCR_MODEL_ID:
    raise RuntimeError("CLARIFAI_OCR_MODEL_ID is not set (Railway Variables → CLARIFAI_OCR_MODEL_ID).")

# ----------------------------
# Clarifai gRPC setup
# ----------------------------
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (("authorization", "Key " + PAT),)
user_data = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="ScreenSnapp Clarifai Image Analysis API",
    description="Analyze images using Clarifai models + OCR + TMDB (protected by Bearer auth).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    incoming = (credentials.credentials or "").strip()
    if incoming != BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return incoming

# ----------------------------
# Helpers
# ----------------------------
def _post_clarifai_model(image_bytes: bytes, model_id: str, version_id: str = ""):
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data")

    req = service_pb2.PostModelOutputsRequest(
        user_app_id=user_data,
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=image_bytes)
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
    """General image recognition concepts."""
    resp = _post_clarifai_model(image_bytes, MODEL_ID, MODEL_VERSION_ID)

    concepts = resp.outputs[0].data.concepts if resp.outputs else []
    return [
        {"name": c.name, "confidence": round(float(c.value), 4)}
        for c in concepts
    ]


def call_clarifai_ocr(image_bytes: bytes):
    """OCR: returns tokens + combined text."""
    resp = _post_clarifai_model(image_bytes, OCR_MODEL_ID, OCR_MODEL_VERSION_ID)

    texts = []
    try:
        out = resp.outputs[0]

        # Some OCR models put it here
        if out.data.text.raw:
            texts.append(out.data.text.raw)

        # Some put it in regions
        for r in out.data.regions:
            if r.data.text.raw:
                texts.append(r.data.text.raw)

    except Exception:
        pass

    texts = [t.strip() for t in texts if t and t.strip()]

    unique = []
    for t in texts:
        if t not in unique:
            unique.append(t)

    combined = " ".join(unique)
    return {
        "tokens": unique[:50],
        "combined": combined[:5000],
    }


def tmdb_search_best(query: str):
    """Search TMDB and return best movie/tv match."""
    q = (query or "").strip()
    if len(q) < 3:
        return None

    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": TMDB_API_KEY,
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
    return {
        "status": "healthy",
        "service": "ScreenSnapp Clarifai Image Analysis API",
        "version": "1.0.0",
        "auth_enabled": True,
        "model_id": MODEL_ID,
        "ocr_model_id_set": bool(OCR_MODEL_ID),
        "tmdb_key_set": bool(TMDB_API_KEY),
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

        return {
            "concepts": concepts,
            "ocr": ocr,
            "match": match,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")



