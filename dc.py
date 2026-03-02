import json
import time
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime, timezone

import arxiv
from arxiv2text import arxiv_to_text


PNOS_PATH = Path("papernumbers.json")
DATASET_PATH = Path("dataset.json")

client = arxiv.Client()

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
    if not full_text.strip():
        full_text = r.summary

    arxiv_id = r.get_short_id()
    doi = getattr(r, "doi", None)

    author_count = len(r.authors) if r.authors else 0
    is_author_count_large = author_count > 7

    published_dt = r.published if isinstance(r.published, datetime) else None
    years_since_published = compute_years_since_published(published_dt)

    summary = r.summary or ""
    summary_len = len(summary)
    text_len = len(full_text)

    return {
        "arxiv_id": arxiv_id,
        "title": r.title,
        "author_count": author_count,
        "is_author_count_large": is_author_count_large,
        "summary": summary,
        "summary_len": summary_len,
        "primary_category": r.primary_category,
        "categories": list(r.categories),
        "published": r.published.isoformat() if r.published else None,
        "updated": r.updated.isoformat() if r.updated else None,
        "years_since_published": years_since_published,
        "pdf_url": r.pdf_url,
        "doi": doi,
        "has_doi": bool(doi),
        "text": full_text,
        "text_len": text_len,
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

        time.sleep(0.2)  # basic politeness to arXiv

    if not new_entries:
        print("No new entries to add.")
        return

    dataset.extend(new_entries)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Appended {len(new_entries)} new papers. Total in dataset: {len(dataset)}")


if __name__ == "__main__":
    main()
