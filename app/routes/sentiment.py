from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.sentiment_chain import classify_text
from src.auth.api_key import verify_api_key

router = APIRouter()

class SentimentRequest(BaseModel):
    title: str
    body: str

class SentimentResponse(BaseModel):
    label: str
    confidence: float
    rationale: str

@router.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(
    req: SentimentRequest,
    api_key: str = Depends(verify_api_key),
):
    result = classify_text(req.title, req.body)
    return result
