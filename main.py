import os
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# Load env vars (local dev). On Railway, env vars come from Variables tab.
load_dotenv()

# ----------------------------
# ENV / CONFIG
# ----------------------------
PAT = os.getenv("CLARIFAI_PAT", "").strip()
USER_ID = os.getenv("CLARIFAI_USER_ID", "nxi9k6mtpija").strip()
APP_ID = os.getenv("CLARIFAI_APP_ID", "ScreenSnapp-Vision").strip()

MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "set-2").strip()
MODEL_VERSION_ID = os.getenv("CLARIFAI_MODEL_VERSION_ID", "").strip()

BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

# Fail fast with clear error (prevents mystery 502s)
if not PAT:
    raise RuntimeError("CLARIFAI_PAT is not set (Railway Variables → CLARIFAI_PAT).")
if not BEARER_TOKEN:
    raise RuntimeError("API_BEARER_TOKEN is not set (Railway Variables → API_BEARER_TOKEN).")

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
    description="Analyze images using Clarifai models (protected by Bearer auth).",
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
def call_clarifai_with_bytes(image_bytes: bytes):
    """Send raw bytes to Clarifai and return concept predictions."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data")

    # Build request
    req = service_pb2.PostModelOutputsRequest(
        user_app_id=user_data,
        model_id=MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=image_bytes)  # Clarifai expects BYTES here
                )
            )
        ],
    )

    # Only include version_id if you actually set it
    if MODEL_VERSION_ID:
        req.version_id = MODEL_VERSION_ID

    # Call Clarifai
    resp = stub.PostModelOutputs(req, metadata=metadata)

    if resp.status.code != status_code_pb2.SUCCESS:
        raise HTTPException(
            status_code=500,
            detail=f"Clarifai error: {resp.status.description}",
        )

    concepts = resp.outputs[0].data.concepts if resp.outputs else []
    predictions = [
        {"name": c.name, "confidence": round(float(c.value), 4)}
        for c in concepts
    ]
    return predictions

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
        "model_version_id_set": bool(MODEL_VERSION_ID),
    }

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    token: str = Depends(verify_token),
):
    try:
        image_bytes = await file.read()
        predictions = call_clarifai_with_bytes(image_bytes)
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
    # fetch image
    try:
        r = requests.get(image_url.url, timeout=15)
        r.raise_for_status()
        image_bytes = r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image URL: {e}")

    try:
        predictions = call_clarifai_with_bytes(image_bytes)
        return {"predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
