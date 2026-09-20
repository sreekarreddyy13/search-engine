"""
fetch_wikipedia_corpus.py

Downloads plain-text content for a list of Wikipedia article titles and
saves each one as a separate .txt file in ./corpus/ — this becomes the
corpus your C++ search engine will read.

Usage:
    pip install requests
    python fetch_wikipedia_corpus.py

Each file is named doc_<n>_<title>.txt so your C++ program can loop over
the folder and treat each file as one document.
"""

import os
import re
import requests
import time

# A mix of general-topic articles — feel free to edit this list.
# Diversity of topics matters more than a huge count for a first pass.
ARTICLE_TITLES = [
    "Artificial intelligence", "Internet", "Python (programming language)",
    "Automobile", "Brake", "Formula SAE", "Mechanical engineering",
    "Electric vehicle", "Search engine", "Database", "Operating system",
    "Computer network", "Algorithm", "Data structure", "Machine learning",
    "Solar energy", "Climate change", "World War II", "Ancient Rome",
    "Photosynthesis", "Quantum mechanics", "Evolution", "Football",
    "Basketball", "Olympic Games", "Renaissance", "Space exploration",
    "Mars", "Black hole", "Human brain", "Genetics", "Vaccine",
    "Economics", "Stock market", "Cryptocurrency", "Blockchain",
    "India", "United States", "China", "European Union", "Democracy",
    "Shakespeare", "Music theory", "Jazz", "Film", "Television",
    "Chess", "Mathematics", "Calculus", "Statistics", "Physics",
]

OUTPUT_DIR = "corpus"
API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "SearchEngineCorpusBuilder/1.0 (student project; contact: reddy@example.com)"
}


def sanitize_filename(title: str) -> str:
    """Turn an article title into a safe filename."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_")
    return safe.lower()


def fetch_plaintext(title: str) -> str | None:
    """Fetch the plain-text extract of a Wikipedia article via the API."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "")
        if text.strip():
            return text
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = 0

    for i, title in enumerate(ARTICLE_TITLES, start=1):
        print(f"[{i}/{len(ARTICLE_TITLES)}] Fetching: {title}")
        try:
            text = fetch_plaintext(title)
        except requests.RequestException as e:
            print(f"  Failed to fetch '{title}': {e}")
            continue

        if not text:
            print(f"  No content found for '{title}', skipping.")
            continue

        filename = f"doc_{i}_{sanitize_filename(title)}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        saved += 1
        time.sleep(0.2)  # be polite to the API

    print(f"\nDone. Saved {saved} documents to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
