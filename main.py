import os
import base64
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Clarifai credentials
PAT = os.getenv("CLARIFAI_PAT", "7607dc924f7d48cb9498d01f28fcb71d")
USER_ID = os.getenv("CLARIFAI_USER_ID", "nxi9k6mtpija")
APP_ID = os.getenv("CLARIFAI_APP_ID", "ScreenSnapp-Vision")
MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "set-2")
MODEL_VERSION_ID = os.getenv("CLARIFAI_MODEL_VERSION_ID", "f2fb3217afa341ce87545e1ba7bf0b64")

# Bearer token for authentication - prioritize environment variable, fallback to test token
BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not BEARER_TOKEN:
    raise RuntimeError("API_BEARER_TOKEN is not set")

# Setup Clarifai gRPC
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', 'Key ' + PAT),)
user_data = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

# Security scheme
security = HTTPBearer()

app = FastAPI(
    title="Clarifai Image Analysis API",
    description="API for analyzing images using Clarifai AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Clarifai Image Analysis API",
        "version": "1.0.0",
        "auth_enabled": True,
        "token_source": "environment" if os.getenv("API_BEARER_TOKEN") else "default"
    }

@app.post("/analyze-image")
async def predict_from_image(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    print("Received file: ")
    # Read image and encode
    image_bytes = await file.read()

    # Send image to Clarifai
    response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=user_data,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=image_bytes)
                    )
                )
            ]
        ),
        metadata=metadata
    )

    # Check for errors
    if response.status.code != status_code_pb2.SUCCESS:
        return JSONResponse(
            status_code=500,
            content={"error": response.status.description}
        )

    # Extract predictions
    predictions = []
    for concept in response.outputs[0].data.concepts:
        predictions.append({
            "name": concept.name,
            "confidence": round(concept.value, 4)
        })

    return {"predictions": predictions}



import base64

class ImageURL(BaseModel):
    url: str

@app.post("/analyze-image-url")
async def predict_from_url(
    image_url: ImageURL,
    token: str = Depends(verify_token)
):
    # 1) Fetch image from URL
    try:
        resp = requests.get(image_url.url, timeout=10)
        resp.raise_for_status()
        image_bytes = resp.content  # <-- RAW BYTES
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching or encoding image: {e}"
        )

    # 2) Send raw bytes directly to Clarifai (NO base64.decode)
    try:
        clarifai_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=user_data,
                model_id=MODEL_ID,
                version_id=MODEL_VERSION_ID,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=image_bytes)
                        )
                    )
                ]
            ),
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Clarifai call failed: {e}"
        )

    # 3) Check Clarifai status
    if clarifai_response.status.code != status_code_pb2.SUCCESS:
        raise HTTPException(
            status_code=500,
            detail=f"Clarifai error: {clarifai_response.status.description}"
        )

    # 4) Extract predictions
    predictions = []
    for concept in clarifai_response.outputs[0].data.concepts:
        predictions.append({
            "name": concept.name,
            "confidence": round(concept.value, 4)
        })

    return {"predictions": predictions}

