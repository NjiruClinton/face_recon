# Face Recognition API
A FastAPI-based face recognition service using InsightFace.

### Docker deployment

```bash
docker build --no-cache -t face-service .
docker run -p 8001:8001 face-service
```

1. The API will be available at:
- Main endpoint: http://localhost:8001
- Interactive docs: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### API Endpoints

- `POST /embed`: Extract face embedding from an uploaded image
- `POST /compare`: Compare two face embeddings for similarity
- `POST /liveliness`: Check if a face is live/real using silent anti-spoofing techniques

#### Liveliness Detection

The `/liveliness` endpoint performs silent face anti-spoofing to detect:
- Photos or printed images
- Screen displays/digital images
- Masks or other spoofing attempts

**Features:**
- **Multi-frame Analysis**: Accepts 1-10 image frames for improved accuracy
- **Texture Analysis**: Analyzes natural skin texture patterns
- **Edge Detection**: Examines edge characteristics (real faces have softer edges)
- **Color Distribution**: Checks for natural color variations in facial regions  
- **Frequency Analysis**: Analyzes frequency domain patterns (screens have different patterns)
- **Reflection Detection**: Detects screen-like reflections and uniform illumination
- **Consistency Checking**: Cross-frame analysis for temporal consistency

**Request**: Upload 1-10 image files, each containing exactly one face
**Single Frame Response**: 
```json
{
  "overall_is_live": true,
  "overall_confidence": 0.783,
  "frame_count": 1,
  "valid_frames": 1,
  "frames": [
    {
      "frame_index": 0,
      "face_detected": true,
      "is_live": true,
      "confidence": 0.783,
      "reason": "Face appears to be live based on anti-spoofing analysis",
      "detailed_scores": {
        "texture": 0.85,
        "edges": 0.72,
        "color": 0.68,
        "frequency": 0.81,
        "reflection": 0.75
      }
    }
  ],
  "aggregated_scores": {
    "texture": 0.85,
    "edges": 0.72,
    "color": 0.68,
    "frequency": 0.81,
    "reflection": 0.75
  },
  "reason": "Analyzed 1 valid frames. Consistent live detection across 100.0% of frames.",
  "model": "silent_anti_spoofing_v1"
}
```

**Multi-Frame Response**: 
```json
{
  "overall_is_live": true,
  "overall_confidence": 0.821,
  "frame_count": 3,
  "valid_frames": 3,
  "consistency_ratio": 1.0,
  "frames": [
    {
      "frame_index": 0,
      "face_detected": true,
      "is_live": true,
      "confidence": 0.78,
      "detailed_scores": {...}
    },
    {
      "frame_index": 1, 
      "face_detected": true,
      "is_live": true,
      "confidence": 0.82,
      "detailed_scores": {...}
    },
    {
      "frame_index": 2,
      "face_detected": true, 
      "is_live": true,
      "confidence": 0.86,
      "detailed_scores": {...}
    }
  ],
  "aggregated_scores": {
    "texture": 0.83,
    "edges": 0.74,
    "color": 0.71,
    "frequency": 0.82,
    "reflection": 0.76
  },
  "reason": "Analyzed 3 valid frames. Consistent live detection across 100.0% of frames.",
  "model": "silent_anti_spoofing_v1"
}
```
