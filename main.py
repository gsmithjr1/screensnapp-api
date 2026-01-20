import os
import re
import json
import logging
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("screensnapp-api")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="ScreenSnapp API", version="1.0.0")

# If you need CORS (usually not needed for iOS native, but harmless)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ENV / CONFIG
# ----------------------------
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT", "").strip()
CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID", "").strip()   # ex: "clarifai" or your user
CLARIFAI_APP_ID = os.getenv("CLARIFAI_APP_ID", "").strip()     # ex: "main" or your app id
CLARIFAI_OCR_MODEL_ID = os.getenv("CLARIFAI_OCR_MODEL_ID", "").strip()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

# Optional "sanity" var you used earlier
TEST_PERSIST = os.getenv("TEST_PERSIST", "").strip()

# Clarifai REST endpoint
CLARIFAI_API_BASE = "https://api.clarifai.com/v2"


# ----------------------------
# Helpers
# ----------------------------
def require_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"{name} is missing (Railway Variables -> {name})")


def bearer_auth(authorization: Optional[str] = Header(default=None)):
    """
    If API_BEARER_TOKEN is set, require:
      Authorization: Bearer <token>
    If API_BEARER_TOKEN is empty, allow requests without auth.
    """
    if not API_BEARER_TOKEN:
        return  # auth disabled

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = parts[1].strip()
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def extract_text_from_clarifai_response(data: Dict[str, Any]) -> str:
    """
    Tries a few common Clarifai response formats.
    OCR models often return "regions" with text, or "concepts" etc.
    We'll gather any obvious strings we can find.
    """
    texts: List[str] = []

    try:
        outputs = data.get("outputs", [])
        if not outputs:
            return ""

        output0 = outputs[0]
        d = output0.get("data", {})

        # Many text/OCR models: data["regions"][i]["data"]["text"]["raw"]
        regions = d.get("regions", [])
        for r in regions:
            raw = (
                r.get("data", {})
                 .get("text", {})
                 .get("raw", "")
            )
            if raw:
                texts.append(raw)

        # Some models return data["text"]["raw"]
        raw_text = d.get("text", {}).get("raw", "")
        if raw_text:
            texts.append(raw_text)

        # Some return concepts with name strings (less common for OCR)
        concepts = d.get("concepts", [])
        for c in concepts:
            name = c.get("name", "")
            if name and isinstance(name, str):
                texts.append(name)

    except Exception:
        # If parsing fails, return empty and let caller handle
        return ""

    # Cleanup: join, remove duplicates while preserving order
    seen = set()
    cleaned = []
    for t in texts:
        t2 = t.strip()
        if t2 and t2.lower() not in seen:
            seen.add(t2.lower())
            cleaned.append(t2)

    return " ".join(cleaned).strip()


def guess_title_from_text(ocr_text: str) -> str:
    """
    Very basic guess: pick the longest "title-like" phrase.
    You can improve this later (or switch to embeddings matching).
    """
    if not ocr_text:
        return ""

    # Remove junk characters
    s = re.sub(r"[\r\n\t]+", " ", ocr_text)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Split into chunks that look like words
    candidates = re.split(r"[|•·]+", s)
    candidates = [c.strip() for c in candidates if c.strip()]

    # Prefer longer candidates but not insane length
    candidates.sort(key=lambda x: len(x), reverse=True)
    for c in candidates:
        if 3 <= len(c) <= 80:
            # Avoid strings that are mostly numbers
            if sum(ch.isalpha() for ch in c) >= 3:
                return c

    return s[:80]


def tmdb_search(query: str) -> Dict[str, Any]:
    """
    Search both movie + tv and return best hit.
    """
    require_env("TMDB_API_KEY", TMDB_API_KEY)

    if not query:
        return {"found": False, "reason": "Empty query"}

    url = "https://api.themoviedb.org/3/search/multi"
    params = {"api_key": TMDB_API_KEY, "query": query, "include_adult": "false"}
    r = requests.get(url, params=params, timeout=20)

    if r.status_code != 200:
        return {
            "found": False,
            "reason": "TMDB error",
            "status_code": r.status_code,
            "body": r.text[:500],
        }

    data = r.json()
    results = data.get("results", [])
    if not results:
        return {"found": False, "reason": "No results"}

    # Pick top result
    best = results[0]
    media_type = best.get("media_type")

    title = best.get("title") or best.get("name") or ""
    overview = best.get("overview") or ""
    poster_path = best.get("poster_path")
    release_date = best.get("release_date") or best.get("first_air_date") or ""
    tmdb_id = best.get("id")

    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    return {
        "found": True,
        "media_type": media_type,
        "tmdb_id": tmdb_id,
        "title": title,
        "overview": overview,
        "release_date": release_date,
        "poster_url": poster_url,
        "raw": best,  # keep for debugging
    }


def clarifai_ocr(image_bytes: bytes) -> Dict[str, Any]:
    """
    Calls Clarifai REST predict endpoint for the OCR model.
    """
    require_env("CLARIFAI_PAT", CLARIFAI_PAT)
    require_env("CLARIFAI_USER_ID", CLARIFAI_USER_ID)
    require_env("CLARIFAI_APP_ID", CLARIFAI_APP_ID)
    require_env("CLARIFAI_OCR_MODEL_ID", CLARIFAI_OCR_MODEL_ID)

    url = f"{CLARIFAI_API_BASE}/users/{CLARIFAI_USER_ID}/apps/{CLARIFAI_APP_ID}/models/{CLARIFAI_OCR_MODEL_ID}/outputs"

    headers = {
        "Authorization": f"Key {CLARIFAI_PAT}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": image_bytes.hex()  # placeholder
                    }
                }
            }
        ]
    }

    # Clarifai expects base64, not hex — do it correctly:
    import base64
    payload["inputs"][0]["data"]["image"]["base64"] = base64.b64encode(image_bytes).decode("utf-8")

    resp = requests.post(url, headers=headers, json=payload, timeout=30)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Clarifai request failed",
                "status_code": resp.status_code,
                "body": resp.text[:1000],
            },
        )

    data = resp.json()

    # Clarifai can return status codes inside JSON too
    status = data.get("status", {})
    if status and status.get("code") not in (None, 10000):
        raise HTTPException(
            status_code=502,
            detail={"error": "Clarifai non-success status", "status": status},
        )

    return data


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "screensnapp-api",
        "has_auth": bool(API_BEARER_TOKEN),
        "test_persist": TEST_PERSIST or None,
    }


@app.post("/identify-screen")
async def identify_screen(
    file: UploadFile = File(...),
    _auth: None = Depends(bearer_auth),
):
    """
    iOS should POST multipart/form-data with key = "file"
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(f"Received file: filename={file.filename} content_type={file.content_type} size={len(contents)}")

        # 1) Clarifai OCR
        clarifai_data = clarifai_ocr(contents)
        ocr_text = extract_text_from_clarifai_response(clarifai_data)

        # 2) Guess a title to search in TMDB
        guessed_title = guess_title_from_text(ocr_text)

        # 3) TMDB search
        tmdb_result = tmdb_search(guessed_title)

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "filename": file.filename,
                "content_type": file.content_type,
                "ocr_text": ocr_text,
                "guessed_title": guessed_title,
                "tmdb": tmdb_result,
            },
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        # Missing env vars etc.
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error in /identify-screen")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


# Helpful: a route to verify your server sees env vars (DON'T expose in public long-term)
@app.get("/debug/env")
def debug_env(_auth: None = Depends(bearer_auth)):
    return {
        "API_BEARER_TOKEN_set": bool(API_BEARER_TOKEN),
        "CLARIFAI_PAT_set": bool(CLARIFAI_PAT),
        "CLARIFAI_USER_ID": CLARIFAI_USER_ID or None,
        "CLARIFAI_APP_ID": CLARIFAI_APP_ID or None,
        "CLARIFAI_OCR_MODEL_ID_set": bool(CLARIFAI_OCR_MODEL_ID),
        "TMDB_API_KEY_set": bool(TMDB_API_KEY),
        "TEST_PERSIST": TEST_PERSIST or None,
    }
