from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from pydantic import BaseModel

app = FastAPI()

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0)

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
