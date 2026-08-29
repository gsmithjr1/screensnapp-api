import os
import json
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


# ============================================================
# ENV / CONFIG
# ============================================================

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT", "").strip()
CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID", "").strip()
CLARIFAI_APP_ID = os.getenv("CLARIFAI_APP_ID", "").strip()

CLARIFAI_MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "").strip()
CLARIFAI_MODEL_VERSION_ID = os.getenv(
    "CLARIFAI_MODEL_VERSION_ID",
    ""
).strip()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

CLARIFAI_OCR_MODEL_ID = os.getenv(
    "CLARIFAI_OCR_MODEL_ID",
    ""
).strip()

XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()


# Confidence thresholds

HIGH_CONF = float(os.getenv("HIGH_CONF", "0.85"))
MED_CONF = float(os.getenv("MED_CONF", "0.65"))


security = HTTPBearer(auto_error=False)


# ============================================================
# AUTH
# ============================================================

def require_api_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if not API_BEARER_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: API_BEARER_TOKEN missing"
        )

    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    if creds.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    return True


# ============================================================
# RESPONSE MODELS
# ============================================================

class Match(BaseModel):
    title: str
    score: float
    id: Optional[str] = None


class IdentifyResponseV2(BaseModel):
    best_title: Optional[str] = None
    best_score: Optional[float] = None

    confidence_level: Literal[
        "high",
        "medium",
        "low",
        "none"
    ] = "none"

    matches: List[Match] = []

    model_id: str
    model_version_id: Optional[str] = None


class GrokIdentifyResponse(BaseModel):
    title: str
    year: Optional[int] = None
    type: Literal[
        "movie",
        "tv",
        "unknown"
    ]


# ============================================================
# CLARIFAI HELPERS
# ============================================================

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
        raise HTTPException(
            status_code=500,
            detail=(
                "Server misconfigured: missing "
                + ", ".join(missing)
            )
        )


def _clarifai_model_outputs_url(
    model_id: str,
    model_version_id: str = ""
) -> str:

    base = (
        f"https://api.clarifai.com/v2/users/"
        f"{CLARIFAI_USER_ID}/apps/"
        f"{CLARIFAI_APP_ID}/models/"
        f"{model_id}"
    )

    if model_version_id:
        return (
            f"{base}/versions/"
            f"{model_version_id}/outputs"
        )

    return f"{base}/outputs"


def _clarifai_post_outputs(
    image_bytes: bytes,
    model_id: str,
    model_version_id: str = ""
) -> Dict[str, Any]:

    b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

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

    url = _clarifai_model_outputs_url(
        model_id,
        model_version_id
    )

    try:
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Clarifai request failed: {str(e)}"
            )
        )

    if r.status_code >= 400:

        try:
            body = r.json()

        except Exception:
            body = {
                "raw": r.text
            }

        raise HTTPException(
            status_code=502,
            detail=f"Clarifai error: {body}"
        )

    try:
        return r.json()

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Clarifai returned non-JSON response"
        )


def _extract_top_concepts(
    clarifai_json: Dict[str, Any],
    limit: int = 5
) -> List[Match]:

    outputs = clarifai_json.get(
        "outputs",
        []
    )

    if not outputs:
        return []

    data = outputs[0].get(
        "data",
        {}
    )

    concepts = data.get(
        "concepts",
        []
    ) or []

    matches: List[Match] = []

    for c in concepts[:limit]:

        name = (
            c.get("name")
            or c.get("id")
            or "unknown"
        )

        val = c.get("value")

        try:
            score = (
                float(val)
                if val is not None
                else 0.0
            )

        except Exception:
            score = 0.0

        matches.append(
            Match(
                title=name,
                score=round(score, 4),
                id=c.get("id"),
            )
        )

    return matches


def _extract_embedding_vector(
    clarifai_json: Dict[str, Any]
) -> List[float]:

    outputs = clarifai_json.get(
        "outputs",
        []
    )

    if not outputs:
        return []

    data = outputs[0].get(
        "data",
        {}
    )

    embeddings = data.get(
        "embeddings",
        []
    ) or []

    if not embeddings:
        return []

    vec = embeddings[0].get(
        "vector",
        []
    ) or []

    out: List[float] = []

    for x in vec:

        try:
            out.append(
                float(x)
            )

        except Exception:
            pass

    return out


def _confidence_level(
    best_score: Optional[float]
) -> str:

    if best_score is None:
        return "none"

    if best_score >= HIGH_CONF:
        return "high"

    if best_score >= MED_CONF:
        return "medium"

    return "low"


# ============================================================
# GROK HELPER
# ============================================================

def _call_grok_for_image(
    image_bytes: bytes,
    content_type: str
) -> Dict[str, Any]:

    if not XAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: XAI_API_KEY missing"
        )

    if content_type not in (
        "image/jpeg",
        "image/jpg",
        "image/png"
    ):
        raise HTTPException(
            status_code=400,
            detail="Grok requires a JPG or PNG image"
        )

    # Convert image to base64

    b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    data_url = (
        f"data:{content_type};base64,{b64}"
    )

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # --------------------------------------------------------
    # GROK REQUEST
    # --------------------------------------------------------

    payload = {
        "model": "grok-4.6",

        # Lower reasoning should reduce recognition time.
        "reasoning": {
            "effort": "low"
        },

        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": (
                            "Identify the exact movie or television "
                            "series shown in this image. "
                            "Return the title and original release year. "
                            "Classify it as movie or tv. "
                            "If there is not enough visual evidence to "
                            "identify the title reliably, return UNKNOWN. "
                            "Do not explain your reasoning."
                        ),
                    },
                ],
            }
        ],

        # Force Grok to give ScreenSnapp a small JSON result.

        "text": {
            "format": {
                "type": "json_schema",
                "name": "screensnapp_identification",
                "strict": True,
                "schema": {
                    "type": "object",

                    "properties": {

                        "title": {
                            "type": "string"
                        },

                        "year": {
                            "type": [
                                "integer",
                                "null"
                            ]
                        },

                        "type": {
                            "type": "string",
                            "enum": [
                                "movie",
                                "tv",
                                "unknown"
                            ]
                        }
                    },

                    "required": [
                        "title",
                        "year",
                        "type"
                    ],

                    "additionalProperties": False
                }
            }
        }
    }

    # --------------------------------------------------------
    # SEND REQUEST
    # --------------------------------------------------------

    try:
        r = requests.post(
            "https://api.x.ai/v1/responses",
            headers=headers,
            json=payload,
            timeout=120,
        )

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Grok recognition timed out"
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Grok request failed: {str(e)}"
        )

    # --------------------------------------------------------
    # CHECK GROK RESPONSE
    # --------------------------------------------------------

    if r.status_code >= 400:

        try:
            body = r.json()

        except Exception:
            body = {
                "raw": r.text
            }

        raise HTTPException(
            status_code=502,
            detail=f"Grok API error: {body}"
        )

    try:
        data = r.json()

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Grok returned a non-JSON response"
        )

    # --------------------------------------------------------
    # FIND GROK'S OUTPUT TEXT
    # --------------------------------------------------------

    output_text = None

    for item in data.get("output", []):

        if item.get("type") != "message":
            continue

        for content in item.get(
            "content",
            []
        ):

            if content.get("type") == "output_text":

                output_text = content.get(
                    "text"
                )

                break

        if output_text:
            break

    if not output_text:
        raise HTTPException(
            status_code=502,
            detail="Grok returned no identification"
        )

    # --------------------------------------------------------
    # CONVERT GROK JSON TEXT INTO PYTHON DICTIONARY
    # --------------------------------------------------------

    try:
        result = json.loads(
            output_text
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not parse Grok identification"
        )

    return result


# ============================================================
# ROUTES
# ============================================================


@app.get("/")
def root():
    return {
        "name": "ScreenSnapp API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "ok": True
    }


# ============================================================
# DEBUG ENV
# ============================================================

@app.get("/debug/env")
def debug_env(
    authorized: bool = Depends(
        require_api_token
    )
):

    def safe(v: str) -> str:

        if not v:
            return ""

        return (
            v[:4]
            + "..."
            + v[-4:]
        )

    return {

        "CLARIFAI_USER_ID": safe(
            CLARIFAI_USER_ID
        ),

        "CLARIFAI_APP_ID": safe(
            CLARIFAI_APP_ID
        ),

        "CLARIFAI_MODEL_ID": safe(
            CLARIFAI_MODEL_ID
        ),

        "CLARIFAI_MODEL_VERSION_ID": safe(
            CLARIFAI_MODEL_VERSION_ID
        ),

        "CLARIFAI_OCR_MODEL_ID": safe(
            CLARIFAI_OCR_MODEL_ID
        ),

        "TMDB_API_KEY_SET": bool(
            TMDB_API_KEY
        ),

        "PAT_SET": bool(
            CLARIFAI_PAT
        ),

        "XAI_API_KEY_SET": bool(
            XAI_API_KEY
        ),
    }


# ============================================================
# CLARIFAI IDENTIFY
# ============================================================

@app.post(
    "/identify",
    response_model=IdentifyResponseV2
)
async def identify_image(
    authorized: bool = Depends(
        require_api_token
    ),
    file: UploadFile = File(...),
):

    _check_clarifai_env()

    if (
        not file.content_type
        or not file.content_type.startswith(
            "image/"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    clarifai_json = _clarifai_post_outputs(
        image_bytes,
        CLARIFAI_MODEL_ID,
        CLARIFAI_MODEL_VERSION_ID
    )

    matches = _extract_top_concepts(
        clarifai_json,
        limit=5
    )

    best_title = (
        matches[0].title
        if matches
        else None
    )

    best_score = (
        matches[0].score
        if matches
        else None
    )

    level = _confidence_level(
        best_score
    )

    final_title = (
        best_title
        if level in (
            "high",
            "medium"
        )
        else None
    )

    return IdentifyResponseV2(
        best_title=final_title,
        best_score=best_score,
        confidence_level=level,
        matches=matches,
        model_id=CLARIFAI_MODEL_ID,
        model_version_id=(
            CLARIFAI_MODEL_VERSION_ID
            or None
        ),
    )


# ============================================================
# CLARIFAI EMBEDDING
# ============================================================

@app.post("/embed")
async def embed_image(
    authorized: bool = Depends(
        require_api_token
    ),
    file: UploadFile = File(...),
):

    _check_clarifai_env()

    if (
        not file.content_type
        or not file.content_type.startswith(
            "image/"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    clarifai_json = _clarifai_post_outputs(
        image_bytes,
        CLARIFAI_MODEL_ID,
        CLARIFAI_MODEL_VERSION_ID
    )

    vec = _extract_embedding_vector(
        clarifai_json
    )

    if not vec:
        raise HTTPException(
            status_code=502,
            detail=(
                "No embeddings returned "
                "(model may not be an embedding model)"
            )
        )

    return {
        "dim": len(vec),
        "vector": vec
    }


# ============================================================
# GROK IDENTIFY
# ============================================================

@app.post(
    "/identify-grok",
    response_model=GrokIdentifyResponse
)
async def identify_grok(
    authorized: bool = Depends(
        require_api_token
    ),
    file: UploadFile = File(...),
):

    if (
        not file.content_type
        or not file.content_type.startswith(
            "image/"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    result = _call_grok_for_image(
        image_bytes,
        file.content_type
    )

    return GrokIdentifyResponse(
        title=result.get(
            "title",
            "UNKNOWN"
        ),
        year=result.get(
            "year"
        ),
        type=result.get(
            "type",
            "unknown"
        ),
    )
