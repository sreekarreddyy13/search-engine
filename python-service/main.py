"""
main.py - Python/FastAPI microservice

Handles the ML/Python-ecosystem pieces of the search engine:
- /fetch-candidates: query arXiv, return matching paper metadata
- (embeddings and RAG answer generation will be added here later)

Usage:
    pip install fastapi uvicorn requests
    uvicorn main:app --reload --port 5000

Then test in a browser:
    http://localhost:5000/fetch-candidates?q=transformer+compression
"""

from fastapi import FastAPI
import requests
import xml.etree.ElementTree as ET
import cohere
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and loads its values into the environment



app = FastAPI()

co = cohere.Client(os.environ["COHERE_API_KEY"])
class Excerpt(BaseModel):
    title: str
    text: str

class AskRequest(BaseModel):
    question: str
    excerpts: List[Excerpt]

@app.post("/generate-answer")
def generate_answer(request: AskRequest):
    # Build a numbered context block from the retrieved excerpts
    context = ""
    for i, ex in enumerate(request.excerpts, start=1):
        context += f"[{i}] {ex.title}\n{ex.text}\n\n"

    prompt = f"""Answer the question using ONLY the excerpts below. Cite sources using [1], [2], etc. matching the excerpt numbers. If the excerpts don't contain enough information, say so.

Excerpts:
{context}

Question: {request.question}

Answer:"""

    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )

    return {"answer": response.text}

  # this object holds all your registered routes

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_from_arxiv(query: str, max_results: int = 50):
    
    """Same logic as the standalone script, now used inside the service."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    resp = requests.get(ARXIV_API_URL, params=params, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", NAMESPACE):
        title = entry.find("atom:title", NAMESPACE).text.strip()
        abstract = entry.find("atom:summary", NAMESPACE).text.strip()
        published = entry.find("atom:published", NAMESPACE).text.strip()
        arxiv_id = entry.find("atom:id", NAMESPACE).text.strip()

        authors = [
            author.find("atom:name", NAMESPACE).text
            for author in entry.findall("atom:author", NAMESPACE)
        ]

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published,
        })

    return papers


# This decorator registers the function below as the handler for
# GET requests to /fetch-candidates - same role as svr.Get(...) in httplib.
@app.get("/fetch-candidates")
@app.get("/fetch-candidates")
def fetch_candidates(q: str, max_results: int = 20):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": q,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,externalIds"
    }
    headers = {"x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except requests.exceptions.HTTPError as e:
        return {"query": q, "count": 0, "papers": [], "error": f"Semantic Scholar API error: {e}"}

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



# Loaded once at startup, not per-request (same "build once" principle as your C++ index)
  # small, fast, good enough for this scale



class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed-batch")
def embed_batch(request: EmbedRequest):
    response = co.embed(texts=request.texts, model="embed-english-v3.0", input_type="search_document")
    return {"embeddings": response.embeddings}