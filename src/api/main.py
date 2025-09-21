from fastapi import FastAPI
from .routers import skills
from .routers import auth

app = FastAPI(
    title="Job Skill Extractor API",
    description="Extracts technical and soft skills from job descriptions.",
    version="1.0.0"
)

# Register routers
app.include_router(skills.router)
app.include_router(auth.router)

@app.get("/")
def root():
    return {"message": "Job Skill Extractor API is running"}