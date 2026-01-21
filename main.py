import os
import re
import requests
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dotenv import load_dotenv

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# ----------------------------
# Load .env locally (Railway uses Variables)
# ----------------------------
load_dotenv()

# ----------------------------
# Required ENV
# ----------------------------
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT", "").strip()
CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID", "").strip()
CLARIFAI_APP_ID = os.getenv("CLARIFAI_APP_ID", "").strip()

# OCR model you trained / are using
CLARIFAI_OCR_MODEL_ID = os.getenv("CLARIFAI_OCR_MODEL_ID", "").strip()
# optional (only if you use it)
CLARIFAI_OCR_MODEL_VERSION_ID = os.getenv("CLARIFAI_OCR_MODEL_VERSION_ID", "").strip()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

# If you want to allow starting without TMDB/Clarifai temporarily, set this to "1"
ALLOW_MISSING_KEYS = os.getenv("ALLOW_MISSING_KEYS", "").strip() == "1"

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="ScreenSnapp API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def require_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"{name} is missing (Railway Variables -> add {name})")


@app.on_event("startup")
def validate_env():
    if ALLOW_MISSING_KEYS:
        return

    require_env("API_BEARER_TOKEN", API_BEARER_TOKEN)
    require_env("CLARIFAI_PAT", CLARIFAI_PAT)
    require_env("CLARIFAI_USER_ID", CLARIFAI_USER_ID)
    require_env("CLARIFAI_APP_ID", CLARIFAI_APP_ID)
    require_env("CLARIFAI_OCR_MODEL_ID", CLARIFAI_OCR_MODEL_ID)
    require_env("TMDB_API_KEY", TMDB_API_KEY)


def auth_guard(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Expects: Authorization: Bearer <API_BEARER_TOKEN>
    """
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    incoming = (creds.credentials or "").strip()
    if not incoming or incoming != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    return True


def clarifai_ocr(image_bytes: bytes) -> str:
    """
    Sends image to Clarifai OCR model and tries to extract text.
    """
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (("authorization", f"Key {CLARIFAI_PAT}"),)

    input_data = resources_pb2.Input(
        data=resources_pb2.Data(
            image=resources_pb2.Image(base64=image_bytes)
        )
    )

    model = resources_pb2.Model(
        id=CLARIFAI_OCR_MODEL_ID
    )

    # If you have a version id, include it
    if CLARIFAI_OCR_MODEL_VERSION_ID:
        model.model_version.id = CLARIFAI_OCR_MODEL_VERSION_ID

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=CLARIFAI_USER_ID, app_id=CLARIFAI_APP_ID),
        model_id=model.id,
        inputs=[input_data],
    )

    # If version ID is present, Clarifai expects it via model_version_id field
    if CLARIFAI_OCR_MODEL_VERSION_ID:
        request.model_version_id = CLARIFAI_OCR_MODEL_VERSION_ID

    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        # This is usually where wrong PAT / wrong model id shows up
        raise HTTPException(
            status_code=502,
            detail=f"Clarifai error: {response.status.description} ({response.status.code})"
        )

    # Try multiple common OCR result shapes
    output = response.outputs[0]

    # 1) Some OCR models populate data.text.raw
    if output.data and output.data.text and output.data.text.raw:
        return output.data.text.raw.strip()

    # 2) Some return regions with text
    texts: List[str] = []

    if output.data and output.data.regions:
        for r in output.data.regions:
            try:
                if r.data and r.data.text and r.data.text.raw:
                    texts.append(r.data.text.raw.strip())
            except Exception:
                pass

    if texts:
        return " ".join(texts).strip()

    # 3) Some return concepts with name-like text (less common for OCR)
    if output.data and output.data.concepts:
        concepts = [c.name for c in output.data.concepts if c.name]
        if concepts:
            return " ".join(concepts).strip()

    return ""


def clean_query(text: str) -> str:
    # Keep it simple: remove weird spacing + very long text
    text = re.sub(r"\s+", " ", text).strip()
    # If OCR returns a ton of junk, limit query length
    return text[:120]


def tmdb_search(query: str) -> Dict[str, Any]:
    """
    Uses TMDB v3 API key (the short hex key), not the long JWT read token.
    """
    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }

    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error: {r.status_code} {r.text}")

    data = r.json()
    results = data.get("results") or []
    if not results:
        return {"found": False, "query": query, "results": []}

    best = results[0]
    media_type = best.get("media_type")
    tmdb_id = best.get("id")

    title = best.get("title") or best.get("name") or ""
    date = best.get("release_date") or best.get("first_air_date") or ""
    overview = best.get("overview") or ""
    poster_path = best.get("poster_path")

    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    tmdb_page = None
    if media_type == "movie":
        tmdb_page = f"https://www.themoviedb.org/movie/{tmdb_id}"
    elif media_type == "tv":
        tmdb_page = f"https://www.themoviedb.org/tv/{tmdb_id}"

    return {
        "found": True,
        "query": query,
        "best": {
            "media_type": media_type,
            "id": tmdb_id,
            "title": title,
            "date": date,
            "overview": overview,
            "poster_url": poster_url,
            "tmdb_page": tmdb_page,
        },
        "results_count": len(results),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/identify-screen")
async def identify_screen(
    file: UploadFile = File(...),
    _auth: bool = Depends(auth_guard),
):
    if not file:
        raise HTTPException(status_code=400, detail="Missing file")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    # OCR
    text = clarifai_ocr(content)
    if not text:
        raise HTTPException(status_code=422, detail="No OCR text extracted")

    query = clean_query(text)
    if not TMDB_API_KEY:
        return {"ocr_text": text, "query": query, "tmdb": {"found": False, "reason": "TMDB_API_KEY missing"}}

    # TMDB search
    tmdb = tmdb_search(query)

    return {
        "ocr_text": text,
        "query": query,
        "tmdb": tmdb,
    }
