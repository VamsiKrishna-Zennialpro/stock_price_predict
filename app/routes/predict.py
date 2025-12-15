from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.predict_price import predict_next_day
from src.auth.api_key import verify_api_key

router = APIRouter()

class PredictRequest(BaseModel):
    ticker: str

class PredictResponse(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    date: str

@router.post("/predict", response_model=PredictResponse)
def predict_price(
    req: PredictRequest,
    api_key: str = Depends(verify_api_key),
):
    return predict_next_day(req.ticker)
