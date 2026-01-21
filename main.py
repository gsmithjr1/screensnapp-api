# main.py
import os
import base64
import requests
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ScreenSnapp API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ENV / CONFIG
# ----------------------------
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT", "").strip()
CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID", "").strip()
CLARIFAI_APP_ID = os.getenv("CLARIFAI_APP_ID", "").strip()

CLARIFAI_MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "").strip()
CLARIFAI_MODEL_VERSION_ID = os.getenv("CLARIFAI_MODEL_VERSION_ID", "").strip()  # optional but recommended

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()  # optional for later

security = HTTPBearer(auto_error=False)

# ----------------------------
# AUTH
# ----------------------------
def require_api_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if not API_BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_BEARER_TOKEN missing")

    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid token")

    if creds.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    return True


# ----------------------------
# RESPONSE MODEL
# ----------------------------
class IdentifyResponse(BaseModel):
    title: Optional[str] = None
    confidence: Optional[float] = None
    model_id: str
    model_version_id: Optional[str] = None
    concepts: List[Dict[str, Any]]


# ----------------------------
# HELPERS
# ----------------------------
def _check_clarifai_env():
    missing = []
    if not CLARIFAI_PAT:
        missing.append("CLARIFAI_PAT")
    if not CLARIFAI_USER_ID:
        missing.append("CLARIFAI_USER_ID")
    if not CLARIFAI_APP_ID:
        missing.append("CLARIFAI_APP_ID")
    if not CLARIFAI_MODEL_ID:
        missing.append("CLARIFAI_MODEL_ID")

    if missing:
        raise HTTPException(status_code=500, detail=f"Server misconfigured: missing {', '.join(missing)}")


def _clarifai_outputs_url() -> str:
    """
    If CLARIFAI_MODEL_VERSION_ID is set, use it.
    Otherwise Clarifai will use the latest model version.
    """
    base = f"https://api.clarifai.com/v2/users/{CLARIFAI_USER_ID}/apps/{CLARIFAI_APP_ID}/models/{CLARIFAI_MODEL_ID}"
    if CLARIFAI_MODEL_VERSION_ID:
        return f"{base}/versions/{CLARIFAI_MODEL_VERSION_ID}/outputs"
    return f"{base}/outputs"


def _clarifai_request(image_bytes: bytes) -> Dict[str, Any]:
    _check_clarifai_env()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = _clarifai_outputs_url()

    headers = {
        "Authorization": f"Key {CLARIFAI_PAT}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": b64
                    }
                }
            }
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Clarifai request failed: {str(e)}")

    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        raise HTTPException(status_code=502, detail=f"Clarifai error: {body}")

    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Clarifai returned non-JSON response")


def _extract_concepts(clarifai_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = clarifai_json.get("outputs", [])
    if not outputs:
        return []

    data = outputs[0].get("data", {})
    concepts = data.get("concepts", []) or []

    cleaned = []
    for c in concepts[:10]:
        cleaned.append({
            "name": c.get("name") or c.get("id"),
            "value": c.get("value"),
            "id": c.get("id"),
        })
    return cleaned


def _title_from_concepts(concepts: List[Dict[str, Any]]) -> (Optional[str], Optional[float]):
    """
    Pick the top concept as the title.
    """
    if not concepts:
        return None, None

    top = concepts[0]
    title = top.get("name")
    value = top.get("value")
    if value is None:
        return title, None

    # round to 2 decimals for UI
    return title, round(float(value), 2)


# ----------------------------
# ROUTES
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/identify", response_model=IdentifyResponse)
async def identify_image(
    authorized: bool = Depends(require_api_token),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    clarifai_json = _clarifai_request(image_bytes)
    concepts = _extract_concepts(clarifai_json)
    title, confidence = _title_from_concepts(concepts)

    return IdentifyResponse(
        title=title,
        confidence=confidence,
        model_id=CLARIFAI_MODEL_ID,
        model_version_id=CLARIFAI_MODEL_VERSION_ID or None,
        concepts=concepts,
    )


@app.post("/identify-screen", response_model=IdentifyResponse)
async def identify_screen(
    authorized: bool = Depends(require_api_token),
    file: UploadFile = File(...),
):
    return await identify_image(authorized=authorized, file=file)
