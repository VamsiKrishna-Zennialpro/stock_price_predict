# app/main.py
from fastapi import FastAPI
from app.routes import health, sentiment, predict, retrieve
from src.observability.langfuse_client import get_langfuse
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from src.security.rate_limit import limiter
from slowapi.middleware import SlowAPIMiddleware

app = FastAPI(
    title="AI Stock Sentiment & Price Prediction API",
    version="1.0.0"
)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(sentiment.router, prefix="/api/v1", tags=["Sentiment"])
app.include_router(retrieve.router, prefix="/api/v1", tags=["Retriever"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

@app.on_event("startup")
def startup_event():
    get_langfuse()
    print("Langfuse initialized")


app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
    )
