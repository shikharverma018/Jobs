from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..core.auth import get_current_user
from ..services.inference import extract_skills
from ..utils.preprocess import sanitize_input

router = APIRouter(prefix="/api", tags=["Skills"])

class JobDescription(BaseModel):
    job_description: str

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict")
def predict_skills(
    payload: JobDescription,
    user: str = Depends(get_current_user)   # enforce auth
):
    # Sanitize input
    text = sanitize_input(payload.job_description)
    if not text:
        raise HTTPException(status_code=400, detail="Invalid input")

    # Extract skills
    result = extract_skills(text)
    return result