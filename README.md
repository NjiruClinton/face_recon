# Face Recognition API

A FastAPI-based face recognition service using InsightFace.

## Docker Setup

### Prerequisites
- Docker
- Docker Compose

### Running the Service

1. Build and start the service:
```bash
docker-compose up --build
```

2. The API will be available at:
- Main endpoint: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

- `POST /embed`: Extract face embedding from an uploaded image
- `POST /compare`: Compare two face embeddings for similarity

### Health Check

The service includes a health check that verifies the API is responding correctly.

### Development

For development with auto-reload, you can mount the code directory:
```bash
docker-compose up
```

The current docker-compose.yaml already includes volume mounting for development.
