# src/vectorstore.py
from chromadb import Client
from chromadb.config import Settings
from src.embeddings import embed_texts

client = Client(
    Settings(
        persist_directory="./data/chroma",
        anonymized_telemetry=False,
    )
)

def get_collection(name="news"):
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_texts,
    )
