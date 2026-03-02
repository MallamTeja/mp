import json
import time
from pathlib import Path
from typing import List, Dict, Set
import urllib.parse

import requests
import arxiv
from arxiv2text import arxiv_to_text


PNOS_PATH = Path("papernumbers.json")
DATASET_PATH = Path("dataset.json")

client = arxiv.Client()

CROSSREF_BASE = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

def clean_arxiv_id(arxiv_id: str) -> str:
    
    return arxiv_id.split("v")[0]


def load_paper_ids() -> List[str]:
    if not PNOS_PATH.exists():
        raise FileNotFoundError(f"{PNOS_PATH} not found")
    with open(PNOS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [pid.strip() for pid in data.get("paper_ids", []) if pid.strip()]


def load_existing_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)


def get_existing_ids(dataset: List[Dict]) -> Set[str]:
    return {item.get("arxiv_id") for item in dataset if "arxiv_id" in item}


def fetch_metadata_for_ids(ids: List[str]) -> List[arxiv.Result]:
    if not ids:
        return []
    search = arxiv.Search(id_list=ids)
    return list(client.results(search))


def get_citation_count_crossref(doi: str) -> int:
    if not doi:
        return 0
    url = f"{CROSSREF_BASE}/{doi}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        return int(msg.get("is-referenced-by-count", 0))
    except Exception as e:
        print(f"[WARN] CrossRef citation lookup failed for DOI {doi}: {e}")
        return 0


def get_citation_count_semantic_scholar(arxiv_id: str) -> int:
    """Search Semantic Scholar for citation count using paper title."""
    try:
        
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(client.results(search))
        title = result.title
        
        
        search_url = f"{SEMANTIC_SCHOLAR_BASE}?fields=citationCount&query={urllib.parse.quote_plus(title)}&max=100"
        
        headers = {
            "accept": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        
                
        print(f"DEBUG: Semantic Scholar API URL: {search_url}")
        params = {} 
        print(f"DEBUG: Semantic Scholar params: {params}")
        r = requests.get(search_url, params=params, headers=headers, timeout=15)
        print(f"DEBUG: Semantic Scholar status code: {r.status_code}")
        print(f"DEBUG: Semantic Scholar response headers: {dict(r.headers)}")
        if r.status_code == 200:
            print(f"DEBUG: Semantic Scholar response data: {r.text[:500]}...") 
            r.raise_for_status()
            data = r.json()
            print(f"DEBUG: Semantic Scholar JSON keys: {list(data.keys())}")
            if data and "data" in data and isinstance(data["data"], list):
                papers = data["data"]
                if papers:
                    print(f"DEBUG: Number of papers found: {len(papers)}")
                    print(f"DEBUG: First paper keys: {list(papers[0].keys()) if papers else 'None'}")
                    citation_count = int(papers[0].get("citationCount", {}).get("value", 0))
                    print(f"DEBUG: Citation count extracted: {citation_count}")
                    return citation_count
            else:
                print("DEBUG: No 'data' field in Semantic Scholar response")
        else:
            print("DEBUG: Unexpected JSON structure from Semantic Scholar")
        
        return 0
        
        
        if data and "data" in data and isinstance(data["data"], list):
            papers = data["data"]
            if papers:
                return int(papers[0].get("citationCount", {}).get("value", 0))
        
        return 0
    except Exception as e:
        print(f"[WARN] Semantic Scholar lookup failed for {arxiv_id}: {e}")
        return 0


def get_citation_count(arxiv_id: str) -> int:
    """Try multiple sources for citation count in order of reliability."""
    
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))
    doi = getattr(result, 'doi', None)
    
    if doi:
        crossref_count = get_citation_count_crossref(doi)
        if crossref_count > 0:
            return crossref_count
    
    
    semantic_count = get_citation_count_semantic_scholar(arxiv_id)
    if semantic_count > 0:
        return semantic_count
    
    print(f"[WARN] No citation count found for {arxiv_id}")
    return 0


def fetch_plain_text_from_pdf(pdf_url: str) -> str:
    try:
        text = arxiv_to_text(pdf_url)
        if text is None:
            return ""
        
        if len(text) > 2_000_000:
            print(f"[INFO] Truncating very long text from {pdf_url}")
            text = text[:2_000_000]
        return text
    except Exception as e:
        print(f"[WARN] PDF to text failed for {pdf_url}: {e}")
        return ""


def result_to_obj(r: arxiv.Result, full_text: str) -> Dict:
    if not full_text.strip():
        full_text = r.summary

    arxiv_id = r.get_short_id()
    doi = getattr(r, "doi", None)
    citation_count = get_citation_count(arxiv_id) if doi else 0
    time.sleep(0.2)

    return {
        "arxiv_id": arxiv_id,
        "title": r.title,
        "authors": [a.name for a in r.authors],
        "summary": r.summary,
        "primary_category": r.primary_category,
        "categories": list(r.categories),
        "published": r.published.isoformat() if r.published else None,
        "updated": r.updated.isoformat() if r.updated else None,
        "entry_id": r.entry_id,
        "pdf_url": r.pdf_url,
        "doi": doi,
        "text": full_text,
        "citation_count": citation_count,
    }


def main():
    paper_ids = load_paper_ids()
    dataset = load_existing_dataset()
    existing_ids = get_existing_ids(dataset)

    
    import sys
    limit = 2  
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Invalid limit argument: {sys.argv[1]}. Using default limit of 2.")
            limit = 2

    new_ids = [pid for pid in paper_ids if pid not in existing_ids][:limit]
    if not new_ids:
        print("No new paper IDs found. Nothing to do.")
        return

    print(f"Processing {len(new_ids)} papers (limited to {limit})...")
    results = fetch_metadata_for_ids(new_ids)
    if not results:
        print("No results returned from arXiv for the new IDs.")
        return

    new_entries: List[Dict] = []
    for i, r in enumerate(results):
        sid = r.get_short_id()
        print(f"[{i+1}/{len(results)}] {sid} - {r.title}")

        full_text = fetch_plain_text_from_pdf(r.pdf_url)
        obj = result_to_obj(r, full_text)
        new_entries.append(obj)

    if not new_entries:
        print("No new entries to add.")
        return

    dataset.extend(new_entries)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Appended {len(new_entries)} new papers. Total in dataset: {len(dataset)}")


if __name__ == "__main__":
    main()

