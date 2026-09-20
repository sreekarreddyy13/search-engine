"""
fetch_arxiv_candidates.py

Given a search query, fetches the top N matching papers from arXiv,
returning title, authors, abstract, arXiv ID, and publish date for each.

This is the "Stage 1: candidate fetching" piece of the dynamic pipeline —
it replaces the fixed pre-downloaded corpus with a fresh, query-specific
batch of papers fetched live.

Usage:
    pip install requests
    python fetch_arxiv_candidates.py
"""

import requests
import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# arXiv's XML uses "namespaces" - a way of avoiding tag-name clashes between
# different XML formats. We need to declare this to find tags correctly.
NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}



def fetch_candidates(q: str, max_results: int = 20):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": q,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,externalIds"
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    papers = []
    for p in data:
        papers.append({
            "id": p.get("paperId", ""),
            "title": p.get("title", ""),
            "authors": [a.get("name", "") for a in p.get("authors", [])],
            "abstract": p.get("abstract") or "",
            "published": str(p.get("year", "")),
            "citationCount": p.get("citationCount", 0)
        })

    return {"query": q, "count": len(papers), "papers": papers}


def main():
    query = input("Enter a search topic: ")
    papers = fetch_candidates(query, max_results=10)

    print(f"\nFound {len(papers)} papers:\n")
    for i, paper in enumerate(papers, start=1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Published: {paper['published']}")
        print(f"   Abstract: {paper['abstract'][:150]}...")
        print()


if __name__ == "__main__":
    main()
