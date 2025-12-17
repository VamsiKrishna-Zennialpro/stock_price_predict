# scripts/build_vectorstore.py
"""
Builds / updates Chroma vector store using Gemini embeddings.
"""

from sqlalchemy.orm import Session
from src.db import SessionLocal
from src.schema import CleanNews
from src.vectorstore import get_collection

BATCH_SIZE = 100


def build_vectorstore():
    session: Session = SessionLocal()
    collection = get_collection(name="news")

    offset = 0
    total_added = 0

    while True:
        rows = (
            session.query(CleanNews)
            .order_by(CleanNews.id)
            .offset(offset)
            .limit(BATCH_SIZE)
            .all()
        )

        if not rows:
            break

        ids = []
        documents = []
        metadatas = []

        for r in rows:
            doc_id = f"clean:{r.id}"

            text = f"{r.title}\n\n{r.body}".strip()
            if not text:
                continue

            ids.append(doc_id)
            documents.append(text)
            metadatas.append(
                {
                    "clean_id": r.id,
                    "ticker": r.ticker,
                    "source": r.source,
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                    "title": r.title,
                }
            )

        if documents:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(documents)

        offset += BATCH_SIZE

    session.close()
    print(f"Vectorstore build complete. Documents added: {total_added}")


if __name__ == "__main__":
    build_vectorstore()
