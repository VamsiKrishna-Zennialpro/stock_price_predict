from fastapi import APIRouter, Depends, Query
from src.retriever import get_default_retriever
from src.auth.api_key import verify_api_key

router = APIRouter()

@router.get("/retrieve")
def retrieve_news(
    query: str = Query(...),
    ticker: str | None = None,
    k: int = 5,
    api_key: str = Depends(verify_api_key),
):
    retriever = get_default_retriever(k=k)
    filters = {"ticker": ticker} if ticker else None
    docs = retriever.get_relevant_documents(query, k=k, filters=filters)
    return {"results": docs}
