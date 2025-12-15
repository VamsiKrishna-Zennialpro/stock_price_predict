# scripts/langfuse_healthcheck.py
from src.observability.langfuse_client import get_langfuse

if __name__ == "__main__":
    lf = get_langfuse()
    with lf.trace(name="healthcheck") as trace:
        trace.output = {"status": "ok"}
    print("Langfuse healthcheck successful")
