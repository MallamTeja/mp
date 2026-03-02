import json
import time
from pathlib import Path
from typing import List, Dict, Set, Optional
import urllib.parse
from datetime import datetime, timezone

import requests
import arxiv
from arxiv2text import arxiv_to_text


PNOS_PATH = Path("papernumbers.json")
DATASET_PATH = Path("dataset.json")

client = arxiv.Client()

CROSSREF_BASE = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_BASE = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)

MAX_TEXT_CHARS = 2_000_000


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


def get_citation_count_semantic_scholar_by_title(title: str) -> int:
    if not title:
        return 0

    try:
        query = urllib.parse.quote_plus(title)
        search_url = (
            f"{SEMANTIC_SCHOLAR_BASE}"
            f"?fields=citationCount&query={query}&limit=1"
        )
        headers = {
            "accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }

        r = requests.get(search_url, headers=headers, timeout=15)
        print(f"[DEBUG] Semantic Scholar status code: {r.status_code}")
        if r.status_code != 200:
            print(f"[WARN] Semantic Scholar non-200: {r.text[:300]}")
            return 0

        data = r.json()
        # Expect format: {"data": [ { "citationCount": int, ... }, ... ] }
        papers = data.get("data") or []
        if not isinstance(papers, list) or not papers:
            print("[DEBUG] Semantic Scholar: no papers in data")
            return 0

        first = papers[0]
        # Citation count may be an int or nested; handle both defensively
        cc = first.get("citationCount", 0)
        if isinstance(cc, dict):
            cc = cc.get("value", 0)

        try:
            cc_int = int(cc)
        except Exception:
            cc_int = 0

        print(f"[DEBUG] Semantic Scholar citationCount: {cc_int}")
        return max(cc_int, 0)
    except Exception as e:
        print(f"[WARN] Semantic Scholar lookup failed for title '{title}': {e}")
        return 0


def get_citation_count(arxiv_result: arxiv.Result) -> int:
    """
    Try multiple sources for citation count in order:
    1) CrossRef via DOI (if present)
    2) Semantic Scholar via title
    """
    doi: Optional[str] = getattr(arxiv_result, "doi", None)
    title: str = arxiv_result.title

    # 1) CrossRef using DOI
    if doi:
        crossref_count = get_citation_count_crossref(doi)
        if crossref_count > 0:
            return crossref_count

    # 2) Semantic Scholar using title
    semantic_count = get_citation_count_semantic_scholar_by_title(title)
    if semantic_count > 0:
        return semantic_count

    print(f"[WARN] No citation count found for {arxiv_result.get_short_id()}")
    return 0


def fetch_plain_text_from_pdf(pdf_url: str) -> str:
    try:
        text = arxiv_to_text(pdf_url)
        if text is None:
            return ""
        if len(text) > MAX_TEXT_CHARS:
            print(f"[INFO] Truncating very long text from {pdf_url}")
            text = text[:MAX_TEXT_CHARS]
        return text
    except Exception as e:
        print(f"[WARN] PDF to text failed for {pdf_url}: {e}")
        return ""


def compute_years_since_published(published_dt: Optional[datetime]) -> float:
    if not published_dt:
        return 0.0
    now = datetime.now(timezone.utc)
    delta_years = (now - published_dt).total_seconds() / (365.25 * 24 * 3600)
    return max(delta_years, 0.0)


def result_to_obj(r: arxiv.Result, full_text: str) -> Dict:
    # Fallback: if full_text is empty, use the abstract (summary)
    if not full_text.strip():
        full_text = r.summary

    arxiv_id = r.get_short_id()
    doi = getattr(r, "doi", None)

    citation_count = get_citation_count(r)
    time.sleep(0.2)  # basic rate limiting

    # author_count instead of full author list
    author_count = len(r.authors) if r.authors else 0

    published_dt = r.published if isinstance(r.published, datetime) else None
    years_since_published = compute_years_since_published(published_dt)
    citations_per_year = (
        citation_count / years_since_published if years_since_published > 0 else 0.0
    )

    return {
        "arxiv_id": arxiv_id,
        "title": r.title,
        "author_count": author_count,
        "summary": r.summary,
        "primary_category": r.primary_category,
        "categories": list(r.categories),
        "published": r.published.isoformat() if r.published else None,
        "updated": r.updated.isoformat() if r.updated else None,
        "pdf_url": r.pdf_url,
        "doi": doi,
        "has_doi": bool(doi),
        "text": full_text,
        "citation_count": citation_count,
        "years_since_published": years_since_published,
        "citations_per_year": citations_per_year,
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
            print(f"Invalid limit argument: {sys.argv[1]}. Using default limit of {limit}.")

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
