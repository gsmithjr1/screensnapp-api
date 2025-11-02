# Clarifai Image Analysis API

A FastAPI-based service that analyzes images using Clarifai's AI vision models. The API provides endpoints for analyzing both uploaded images and images from URLs.

## Features

- **Image Analysis**: Analyze uploaded images or images from URLs
- **AI-Powered**: Uses Clarifai's "ScreenSnapp-Vision" model for predictions
- **RESTful API**: Clean, documented endpoints with OpenAPI/Swagger docs
- **Authentication**: Bearer token-based security
- **Docker Support**: Easy containerization and deployment
- **Health Monitoring**: Built-in health check endpoint

## API Endpoints

### Health Check
- **GET** `/health` - Service health status
- **No authentication required**

### Image Analysis
- **POST** `/analyze-image` - Analyze uploaded image file
- **POST** `/analyze-image-url` - Analyze image from URL
- **Requires Bearer token authentication**

### Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation

## Environment Variables

Create a `.env` file in the root directory:

```bash
# Clarifai API Configuration
CLARIFAI_PAT=your_personal_access_token
CLARIFAI_USER_ID=your_user_id
CLARIFAI_APP_ID=your_app_id
CLARIFAI_MODEL_ID=your_model_id
CLARIFAI_MODEL_VERSION_ID=your_model_version_id

# API Configuration
API_BEARER_TOKEN=your_custom_bearer_token
```

### Current Configuration (Hardcoded for testing)
- **PAT**: `7607dc924f7d48cb9498d01f28fcb71d`
- **USER_ID**: `nxi9k6mtpija`
- **APP_ID**: `ScreenSnapp-Vision`
- **MODEL_ID**: `set-2`
- **MODEL_VERSION_ID**: `f2fb3217afa341ce87545e1ba7bf0b64`
- **TEST_BEARER_TOKEN**: `test_token_12345`

## Authentication

The API uses Bearer token authentication. Include the token in the Authorization header:

```
Authorization: Bearer test_token_12345
```

## Sample Response Schema

### Success Response
```json
{
  "predictions": [
    {
      "name": "concept_name",
      "confidence": 0.9876
    },
    {
      "name": "another_concept",
      "confidence": 0.8543
    }
  ]
}
```

### Error Response
```json
{
  "detail": "Error message description"
}
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip
- Docker (optional)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd clarifai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Clarifai credentials
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -t clarifai-api .
   docker run -p 8000:8000 clarifai-api
   ```

## Testing the API

### Using cURL

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Analyze Image File:**
```bash
curl -X POST "http://localhost:8000/analyze-image" \
  -H "Authorization: Bearer test_token_12345" \
  -F "file=@images/image.png"
```

**Analyze Image URL:**
```bash
curl -X POST "http://localhost:8000/analyze-image-url" \
  -H "Authorization: Bearer test_token_12345" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

### Using Postman

Import the provided `clarifai.postman_collection.json` file into Postman for easy testing.

## Project Structure

```
clarifai/
├── main.py                      # FastAPI application
├── requirements.txt             # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── README.md                   # This file
├── clarifai.postman_collection.json  # Postman collection
└── images/                     # Sample images for testing
    ├── image.png
    ├── image (1).png
    ├── image (2).png
    ├── image (3).png
    └── image (4).png
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Clarifai gRPC**: Official Clarifai Python client
- **Uvicorn**: ASGI server for running FastAPI
- **Python-dotenv**: Environment variable management
- **Python-multipart**: File upload handling

## Production Deployment

### Environment Variables
- Set all Clarifai credentials as environment variables
- Use a strong, unique Bearer token
- Consider using a secrets management service

### Security Considerations
- Change the default Bearer token
- Use HTTPS in production
- Implement rate limiting
- Add request logging and monitoring

### Scaling
- Use multiple worker processes with Uvicorn
- Consider using Gunicorn with Uvicorn workers
- Implement load balancing for high traffic

## Troubleshooting

### Common Issues

1. **Clarifai Authentication Error**
   - Verify your PAT is correct and has proper permissions
   - Check if your model is accessible

2. **File Upload Issues**
   - Ensure the file is a valid image format
   - Check file size limits

3. **Docker Issues**
   - Ensure Docker is running
   - Check if port 8000 is available

### Logs
- Check application logs for detailed error messages
- Monitor Clarifai API responses for model-specific issues

## Support

For issues related to:
- **API functionality**: Check the logs and Clarifai documentation
- **Clarifai models**: Refer to [Clarifai documentation](https://docs.clarifai.com/)
- **FastAPI**: Check [FastAPI documentation](https://fastapi.tiangolo.com/)

## License

This project is provided as-is for demonstration and development purposes.
