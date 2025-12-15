# scripts/query_retriever.py
"""
Quick CLI to query the retriever (semantic search) and print top-k results.
"""

import argparse
from src.retriever import get_default_retriever
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--query", dest="query", required=True)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--ticker", dest="ticker", default=None, help="Optional ticker filter (exact match e.g. RELIANCE.NS)")
    args = parser.parse_args()

    retriever = get_default_retriever(k=args.k)
    filters = {"ticker": args.ticker} if args.ticker else None
    docs = retriever.get_relevant_documents(args.query, k=args.k, filters=filters)
    print(f"Found {len(docs)} documents (query='{args.query}', ticker={args.ticker})\n")
    for i, d in enumerate(docs, start=1):
        print(f"--- RESULT {i} ---")
        print("ID:", d.get("id"))
        print("DIST:", d.get("distance"))
        print("META:", d.get("metadata"))
        print("TEXT SNIPPET:")
        text = d.get("text") or ""
        print(text[:800].strip().replace("\n", " "))
        print("\n")

if __name__ == "__main__":
    main()
