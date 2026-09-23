"""
main.py - Python/FastAPI microservice

Handles the ML/Python-ecosystem pieces of the search engine:
- /fetch-candidates: fetch candidate papers (Semantic Scholar, arXiv fallback)
- /embed-batch: Cohere embeddings
- /generate-answer: Cohere RAG answer generation

Usage:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 5000

Then test in a browser:
    http://localhost:5000/fetch-candidates?q=transformer+compression
"""

from fastapi import FastAPI, HTTPException
import requests
import xml.etree.ElementTree as ET
import cohere
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import time

load_dotenv()  # reads the .env file and loads its values into the environment



app = FastAPI()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
class Excerpt(BaseModel):
    title: str
    text: str

class AskRequest(BaseModel):
    question: str
    excerpts: List[Excerpt]

@app.post("/generate-answer")
def generate_answer(request: AskRequest):
    if co is None:
        raise HTTPException(status_code=503, detail="COHERE_API_KEY not configured; /generate-answer is unavailable.")

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


class RewriteRequest(BaseModel):
    question: str

QUERY_REWRITE_PREAMBLE = """Extract a short search query from the user question for an academic paper search engine.
Rules:
- Use only the specific nouns and named entities already present in the question.
- Do not add generic qualifier words (e.g. "analysis", "techniques", "mechanisms", "overview") unless the user used them.
- Do not add words that are not implied by the question.
- 2 to 6 words. Output ONLY the query text, no quotes, no punctuation, no explanation."""

@app.post("/rewrite-query")
def rewrite_query(request: RewriteRequest):
    # This is a best-effort retrieval aid for /ask: on any failure (missing key, API
    # error, junk output) fall back to the original question rather than erroring out,
    # since a caller can always search with the raw question anyway.
    if co is not None:
        try:
            response = co.chat(
                model="command-a-03-2025",
                message=request.question,
                preamble=QUERY_REWRITE_PREAMBLE,
                temperature=0.3,
                max_tokens=30,
            )
            rewritten = response.text.strip()
            if 0 < len(rewritten) <= 100 and any(c.isalpha() for c in rewritten):
                return {"query": rewritten}
        except Exception as e:
            print(f"Query rewrite failed, falling back to original question: {e}", flush=True)

    return {"query": request.question}


ARXIV_API_URL = "http://export.arxiv.org/api/query"
NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_from_arxiv(query: str, max_results: int = 50):
    """Fallback candidate source used when Semantic Scholar is unavailable."""
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




_cache = {}          # {(query, max_results): (timestamp, papers)}
CACHE_TTL = 3600     # 1 hour

def fetch_from_semantic_scholar(q: str, max_results: int):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": q,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,externalIds",
    }
    headers = {"x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]}

    for attempt in range(3):
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        print(f"S2 status: {resp.status_code} (attempt {attempt + 1})", flush=True)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(min(wait, 5))
            continue
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [{
            "id": p.get("paperId", ""),
            "title": p.get("title", ""),
            "authors": [a.get("name", "") for a in p.get("authors", [])],
            "abstract": p.get("abstract") or "",
            "published": str(p.get("year", "")),
            "citationCount": p.get("citationCount", 0),
        } for p in data]

    raise RuntimeError("Semantic Scholar rate limit (429) after retries")


@app.get("/fetch-candidates")
def fetch_candidates(q: str, max_results: int = 20):
    key = (q.lower().strip(), max_results)

    # 1. cache
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        papers = _cache[key][1]
        return {"query": q, "count": len(papers), "papers": papers, "source": "cache"}

    # 2. Semantic Scholar, 3. fall back to arXiv
    try:
        papers = fetch_from_semantic_scholar(q, max_results)
        source = "semantic_scholar"
    except Exception as e:
        print(f"S2 failed, falling back to arXiv: {e}", flush=True)
        try:
            papers = fetch_from_arxiv(q, max_results)
            for p in papers:
                p.setdefault("citationCount", 0)
            source = "arxiv_fallback"
        except Exception as e2:
            print(f"arXiv also failed: {e2}", flush=True)
            return {"query": q, "count": 0, "papers": [], "error": str(e2)}

    if papers:
        _cache[key] = (time.time(), papers)
    return {"query": q, "count": len(papers), "papers": papers, "source": source}


class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed-batch")
def embed_batch(request: EmbedRequest):
    if co is None:
        raise HTTPException(status_code=503, detail="COHERE_API_KEY not configured; /embed-batch is unavailable.")

    response = co.embed(texts=request.texts, model="embed-english-v3.0", input_type="search_document")
    return {"embeddings": response.embeddings}