import json
from pathlib import Path
from typing import List, Dict, Set
import time
import requests
import arxiv
from arxiv2text import arxiv_to_text

PNOS_PATH = Path("papernumbers.json")
DATASET_PATH = Path("dataset.json")

client = arxiv.Client()

OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_PARAMS = {"mailto": "tejamallam1233@gmail.com"}


def clean_arxiv_id(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def load_paper_ids() -> List[str]:1
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


def arxiv_to_openalex_id(arxiv_id: str) -> str:
    clean_id = clean_arxiv_id(arxiv_id)
    params = {**OPENALEX_PARAMS, "filter": f"ids.arxiv:{clean_id}"}
    try:
        r = requests.get(OPENALEX_BASE, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        return results[0]["id"] if results else None
    except Exception:
        return None


def get_citation_count_openalex_by_arxiv(arxiv_id: str) -> int:
    work_id = arxiv_to_openalex_id(arxiv_id)
    if not work_id:
        return 0
    try:
        r = requests.get(work_id, params=OPENALEX_PARAMS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("cited_by_count", 0)
    except Exception:
        return 0


def fetch_plain_text_from_pdf(pdf_url: str) -> str:
    try:
        text = arxiv_to_text(pdf_url)
        if text is None:
            return ""
        return text
    except Exception:
        return ""


def result_to_obj(r: arxiv.Result, full_text: str) -> Dict:
    if not full_text.strip():
        full_text = r.summary

    arxiv_id = r.get_short_id()
    citation_count = get_citation_count_openalex_by_arxiv(arxiv_id)
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
        "text": full_text,
        "citation_count": citation_count,
    }


def main():
    paper_ids = load_paper_ids()
    dataset = load_existing_dataset()
    existing_ids = get_existing_ids(dataset)

    new_ids = [pid for pid in paper_ids if pid not in existing_ids]
    if not new_ids:
        print("No new paper IDs found. Nothing to do.")
        return

    print(f"New IDs to fetch: {new_ids}")

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
