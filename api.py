from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import predict_comments
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Schizophrenia Detector API")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the HTML frontend
frontend_path = Path(__file__).parent.parent / "frontend.html"

@app.get("/", response_class=HTMLResponse)
def read_frontend():
    """
    Serve the frontend HTML file.
    """
    try:
        return frontend_path.read_text()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend file not found")

class TextInput(BaseModel):
    comments: list[str]

@app.post("/predict")
def predict(input_data: TextInput):
    """
    Predict the likelihood of schizophrenia based on input comments.
    """
    try:
        if not input_data.comments:
            raise HTTPException(status_code=400, detail="Input list cannot be empty")
        return predict_comments(input_data.comments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
