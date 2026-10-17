from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from pydantic import BaseModel
from services.liveliness_service import LivelinessService
from typing import List

app = FastAPI()

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0)

# Initialize liveliness service for anti-spoofing
liveliness_service = LivelinessService()

class CompareRequest(BaseModel):
    embeddingA: list[float]
    embeddingB: list[float]

def read_image(file: UploadFile):
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return img

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    img = read_image(file)
    faces = face_app.get(img)

    if len(faces) != 1:
        raise HTTPException(
            status_code=400,
            detail="Exactly one face required"
        )

    embedding = faces[0].embedding.tolist()

    return {
        "embedding": embedding,
        "model": "arcface_buffalo_l"
    }

@app.post("/compare")
async def compare(req: CompareRequest):
    a = np.array(req.embeddingA)
    b = np.array(req.embeddingB)

    similarity = float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )

    return {
        "similarity": similarity,
        "match": similarity >= 0.75
    }


@app.post("/liveliness")
async def check_liveliness(files: List[UploadFile] = File(...)):
    """
    Check if faces in the uploaded images appear to be live/real.

    This endpoint accepts multiple image frames and uses silent face anti-spoofing
    techniques to detect:
    - Photos or printed images
    - Screen displays
    - Masks or other spoofing attempts

    The analysis is performed without requiring user interaction.
    Supports both single image and multiple frame analysis for improved accuracy.

    Args:
        files: List of image files to analyze (supports 1-10 frames)

    Returns:
        - overall_is_live: Boolean indicating overall liveliness assessment
        - overall_confidence: Float score (0-1) for overall confidence
        - frame_count: Number of frames analyzed
        - frames: List of per-frame analysis results
        - aggregated_scores: Combined scores across all frames
    """
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one image file is required"
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 frames supported per request"
        )

    frame_results = []
    all_scores = []

    for i, file in enumerate(files):
        try:
            img = read_image(file)
            faces = face_app.get(img)

            if len(faces) == 0:
                frame_result = {
                    "frame_index": i,
                    "face_detected": False,
                    "is_live": False,
                    "confidence": 0.0,
                    "reason": "No face detected in frame",
                    "detailed_scores": {}
                }
            elif len(faces) > 1:
                frame_result = {
                    "frame_index": i,
                    "face_detected": True,
                    "is_live": False,
                    "confidence": 0.0,
                    "reason": "Multiple faces detected in frame",
                    "detailed_scores": {}
                }
            else:
                # Single face detected - perform liveliness analysis
                face = faces[0]
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]

                # Convert to (x, y, width, height) format for liveliness service
                x, y, x2, y2 = bbox
                face_bbox = (x, y, x2 - x, y2 - y)

                # Perform liveliness analysis
                liveliness_result = liveliness_service.analyze_liveliness(img, face_bbox)

                frame_result = {
                    "frame_index": i,
                    "face_detected": True,
                    "is_live": liveliness_result["is_live"],
                    "confidence": liveliness_result["confidence"],
                    "reason": liveliness_result["reason"],
                    "detailed_scores": liveliness_result["scores"]
                }

                # Collect scores for aggregation (only from valid detections)
                all_scores.append(liveliness_result["scores"])

            frame_results.append(frame_result)

        except Exception as e:
            frame_result = {
                "frame_index": i,
                "face_detected": False,
                "is_live": False,
                "confidence": 0.0,
                "reason": f"Error processing frame: {str(e)}",
                "detailed_scores": {}
            }
            frame_results.append(frame_result)

    # Calculate aggregated results across all valid frames
    valid_frames = [f for f in frame_results if f["face_detected"] and len(f["detailed_scores"]) > 0]

    if len(valid_frames) == 0:
        overall_result = {
            "overall_is_live": False,
            "overall_confidence": 0.0,
            "frame_count": len(files),
            "valid_frames": 0,
            "frames": frame_results,
            "aggregated_scores": {},
            "reason": "No valid faces detected in any frame",
            "model": "silent_anti_spoofing_v1"
        }
    else:
        # Aggregate scores across all valid frames
        aggregated_scores = {}
        for score_type in ["texture", "edges", "color", "frequency", "reflection"]:
            scores_for_type = [frame["detailed_scores"][score_type] for frame in valid_frames if score_type in frame["detailed_scores"]]
            if scores_for_type:
                # Use median for robustness against outliers
                aggregated_scores[score_type] = float(np.median(scores_for_type))

        # Calculate overall confidence using the same weighting as individual frames
        weights = {
            'texture': 0.25,
            'edges': 0.20,
            'color': 0.20,
            'frequency': 0.20,
            'reflection': 0.15
        }

        overall_confidence = sum(aggregated_scores.get(test, 0) * weights[test] for test in weights.keys())
        overall_is_live = overall_confidence >= 0.6

        # Additional multi-frame consistency check
        live_count = sum(1 for f in valid_frames if f["is_live"])
        consistency_ratio = live_count / len(valid_frames)

        # Adjust confidence based on consistency across frames
        if consistency_ratio < 0.5:
            # If less than half the frames are live, reduce confidence
            overall_confidence *= 0.7
            overall_is_live = False
        elif consistency_ratio > 0.8:
            # If most frames are consistent, boost confidence slightly
            overall_confidence = min(1.0, overall_confidence * 1.1)

        reason = f"Analyzed {len(valid_frames)} valid frames. "
        if overall_is_live:
            reason += f"Consistent live detection across {consistency_ratio:.1%} of frames."
        else:
            reason += f"Inconsistent or low confidence detection (live in {consistency_ratio:.1%} of frames)."

        overall_result = {
            "overall_is_live": overall_is_live,
            "overall_confidence": round(overall_confidence, 3),
            "frame_count": len(files),
            "valid_frames": len(valid_frames),
            "consistency_ratio": round(consistency_ratio, 3),
            "frames": frame_results,
            "aggregated_scores": aggregated_scores,
            "reason": reason,
            "model": "silent_anti_spoofing_v1"
        }

    return overall_result

