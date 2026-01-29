# main.py
import os
import base64
import requests
from typing import Optional, List, Dict, Any, Literal

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

# Confidence thresholds (tune these later)
HIGH_CONF = float(os.getenv("HIGH_CONF", "0.85"))
MED_CONF = float(os.getenv("MED_CONF", "0.65"))

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
# RESPONSE MODELS
# ----------------------------
class Match(BaseModel):
    title: str
    score: float
    id: Optional[str] = None

class IdentifyResponseV2(BaseModel):
    best_title: Optional[str] = None
    best_score: Optional[float] = None
    confidence_level: Literal["high", "medium", "low", "none"] = "none"
    matches: List[Match] = []
    model_id: str
    model_version_id: Optional[str] = None


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
        "inputs": [{"data": {"image": {"base64": b64}}}]
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


def _extract_top_matches(clarifai_json: Dict[str, Any], limit: int = 5) -> List[Match]:
    outputs = clarifai_json.get("outputs", [])
    if not outputs:
        return []

    data = outputs[0].get("data", {})
    concepts = data.get("concepts", []) or []

    matches: List[Match] = []
    for c in concepts[:limit]:
        name = c.get("name") or c.get("id") or "unknown"
        val = c.get("value")
        try:
            score = float(val) if val is not None else 0.0
        except Exception:
            score = 0.0

        matches.append(Match(
            title=name,
            score=round(score, 4),
            id=c.get("id"),
        ))
    return matches


def _confidence_level(best_score: Optional[float]) -> str:
    if best_score is None:
        return "none"
    if best_score >= HIGH_CONF:
        return "high"
    if best_score >= MED_CONF:
        return "medium"
    return "low"


# ----------------------------
# ROUTES
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/identify", response_model=IdentifyResponseV2)
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
    matches = _extract_top_matches(clarifai_json, limit=5)

    best_title = matches[0].title if matches else None
    best_score = matches[0].score if matches else None
    level = _confidence_level(best_score)

    # Don’t force a title if low confidence (prevents wrong answers)
    final_title = best_title if level in ("high", "medium") else None

    return IdentifyResponseV2(
        best_title=final_title,
        best_score=best_score,
        confidence_level=level,
        matches=matches,
        model_id=CLARIFAI_MODEL_ID,
        model_version_id=CLARIFAI_MODEL_VERSION_ID or None,
    )


@app.post("/identify-screen", response_model=IdentifyResponseV2)
async def identify_screen(
    authorized: bool = Depends(require_api_token),
    file: UploadFile = File(...),
):
    return await identify_image(authorized=authorized, file=file)
