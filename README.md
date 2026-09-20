# AI-Powered Research Paper Search Engine

A hybrid retrieval system that combines a custom-built C++ search engine (trie,
inverted index, TF-IDF) with semantic embeddings and a retrieval-augmented
generation (RAG) pipeline, to search live research papers and answer natural-
language questions with cited sources.

**Live demo:**
- Frontend: open `index.html` locally in a browser (see Setup below)
- C++ API: https://search-engine-1-qtwg.onrender.com
- Python API: https://search-engine-pt99.onrender.com

> Note: both services are on Render's free tier and spin down after
> inactivity — the first request after a period of inactivity can take
> 30-60 seconds to respond while the service wakes up.

---

## Architecture

This project is a **polyglot, two-service system**:

```
┌─────────────┐        ┌──────────────────┐        ┌────────────────────┐
│  Frontend   │───────▶│   C++ Service     │───────▶│  Python/FastAPI    │
│ (index.html)│  HTTP  │  (cpp-httplib)    │  HTTP  │  Service           │
└─────────────┘        │                   │        │                    │
                        │ - Trie            │        │ - arXiv/Semantic   │
                        │ - Inverted Index  │        │   Scholar fetch    │
                        │ - TF-IDF          │        │ - Cohere embeddings│
                        │ - Reciprocal Rank │        │ - Cohere RAG       │
                        │   Fusion          │        │   generation       │
                        └───────────────────┘        └────────────────────┘
```

**Why two services, two languages:** the C++ service handles the
performance-critical retrieval and ranking logic (data structures, scoring
algorithms) — the part that benefits from being close to the metal and where
the DSA work actually lives. The Python service handles everything that
depends on external ML/AI APIs (embeddings, generation), where Python's
ecosystem is the natural fit. They communicate over a REST API, the same way
independent microservices would in a larger system.

---

## Features

- **Static corpus search** (`/search`, `/autocomplete`) — a small pre-loaded
  Wikipedia corpus, indexed once at startup with a custom inverted index and
  TF-IDF ranking, plus trie-based, frequency-ranked autocomplete.
- **Dynamic hybrid search** (`/hybrid-search`) — fetches live candidate papers
  from the Semantic Scholar API for any query, then re-ranks them using a
  combination of TF-IDF (lexical relevance) and sentence embeddings (semantic
  relevance), merged via **Reciprocal Rank Fusion**.
- **RAG-based Q&A** (`/ask`) — runs hybrid search, retrieves the top-ranked
  paper excerpts, and passes them to Cohere's `command-a-03-2025` model to
  generate a natural-language answer with inline citations back to the
  source papers.

---

## Tech Stack

- **C++17**, `cpp-httplib` (HTTP server + client), `nlohmann/json` (JSON
  parsing)
- **Python 3**, FastAPI, `requests`, `cohere` (embeddings + generation)
- **Semantic Scholar API** (candidate paper fetching)
- **Docker** (C++ service containerization), **Render** (deployment)
- Vanilla **HTML/CSS/JS** frontend (no framework)

---

## API Endpoints

### C++ service
| Endpoint | Description |
|---|---|
| `GET /search?q=` | TF-IDF search over the static corpus |
| `GET /autocomplete?prefix=` | Trie-based, frequency-ranked prefix suggestions |
| `GET /hybrid-search?q=` | Dynamic hybrid (TF-IDF + semantic) search over live-fetched papers |
| `GET /ask?q=` | RAG pipeline: retrieves relevant excerpts, generates a cited answer |

### Python service
| Endpoint | Description |
|---|---|
| `GET /fetch-candidates?q=` | Fetches candidate papers from Semantic Scholar |
| `POST /embed-batch` | Returns embeddings for a batch of texts (Cohere) |
| `POST /generate-answer` | Generates a cited answer from a question + excerpts (Cohere) |

---

## Setup (local development)

**Python service:**
```
cd python-service
pip install -r requirements.txt
```
Create a `.env` file in `python-service/` with:
```
COHERE_API_KEY=your_key
SEMANTIC_SCHOLAR_API_KEY=your_key
```
Run:
```
uvicorn main:app --reload --port 5000
```

**C++ service:**
```
cd cpp-service
g++ -std=c++17 newwithapi.cpp -o server -lws2_32   # Windows
./server
```
(Requires `corpus/`, `httplib.h`, and `json.hpp` in the same folder — already included.)

**Frontend:**
Open `index.html` directly in a browser. Edit the `API_BASE` constant near
the top of the `<script>` block to point at `http://localhost:8080` for
local testing, or the deployed C++ service URL for production.

---

## Design Notes & Honest Limitations

A few things worth knowing about how this system behaves, found through
actual testing rather than assumed:

- **Retrieve-then-rerank is only as good as the first-pass retrieval.**
  Hybrid reranking (TF-IDF + embeddings) can only improve results *within*
  whatever candidates the first-pass fetch returns. Early testing with
  arXiv's search API showed that heavily paraphrased queries (e.g.
  "shrinking neural networks to run faster") often returned irrelevant
  candidates, which no amount of reranking could fix. Switching to the
  Semantic Scholar API (which does its own relevance-aware ranking)
  meaningfully improved first-pass recall.
- **Hybrid reranking's effect is more visible with a larger, more diverse
  candidate pool.** With a small candidate set (~5-10 papers) that's already
  well-matched to the query, TF-IDF-only and hybrid rankings often converge
  on the same top results, just reordered — the benefit of semantic
  reranking becomes clearer as the candidate pool grows.
- **The RAG pipeline correctly refuses to answer when retrieval finds
  nothing relevant**, rather than hallucinating a plausible-sounding answer
  — a deliberate and confirmed property of grounding the generation step
  strictly in retrieved excerpts.
- **Full natural-language questions can underperform keyword-style queries
  at the retrieval stage**, since the search APIs used here are not
  optimized for question-form input. A query simplification/keyword
  extraction step would likely improve `/ask`'s retrieval quality.
- **Third-party search APIs introduce some result non-determinism** — small
  differences in query phrasing (e.g. a hyphen) can change which candidates
  come back, since parsing happens inside a black-box service outside this
  system's control.

---

## Possible Future Improvements

- Citation-count-weighted ranking (Semantic Scholar returns this data
  already; not yet incorporated into scoring)
- Query simplification for the `/ask` endpoint
- Splitting the C++ source into multiple files for better organization
- Proper cleanup of trie node memory (currently leaked, harmless for a
  short-lived process but not ideal for a long-running server)
