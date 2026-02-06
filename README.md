# Face Recognition API
A FastAPI-based face recognition service using InsightFace.

### Docker deployment

```bash
docker build --no-cache -t face-service .
docker run -p 8001:8001 face-service
```

2. The API will be available at:
- Main endpoint: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

- `POST /embed`: Extract face embedding from an uploaded image
- `POST /compare`: Compare two face embeddings for similarity
