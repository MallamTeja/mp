import json
from pathlib import Path
from typing import List, Dict, Set

import arxiv
from arxiv2text import arxiv_to_text


PNOS_PATH = Path("pnos.json")
DATASET_PATH = Path("dataset.json")

client = arxiv.Client()


def load_paper_ids() -> List[str]:
    if not PNOS_PATH.exists():
        raise FileNotFoundError(f"{PNOS_PATH} not found")
    with open(PNOS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get("paper_ids", [])
    return [pid.strip() for pid in ids if pid.strip()]


def load_existing_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_existing_ids(dataset: List[Dict]) -> Set[str]:
    return {item.get("arxiv_id") for item in dataset if "arxiv_id" in item}


def fetch_metadata_for_ids(ids: List[str]) -> List[arxiv.Result]:
    if not ids:
        return []
    search = arxiv.Search(id_list=ids)
    return list(client.results(search))


def result_to_obj(r: arxiv.Result, full_text: str) -> Dict:
    return {
        "arxiv_id": r.get_short_id(),
        "title": r.title,
        "authors": [a.name for a in r.authors],
        "summary": r.summary,
        "primary_category": r.primary_category,
        "categories": list(r.categories),
        "published": r.published.isoformat() if r.published else None,
        "updated": r.updated.isoformat() if r.updated else None,
        "entry_id": r.entry_id,
        "pdf_url": r.pdf_url,
        "text": full_text
    }


def fetch_plain_text_from_pdf(pdf_url: str) -> str:
    text = arxiv_to_text(pdf_url, output_dir=None)
    return text


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
    for r in results:
        sid = r.get_short_id()
        print(f"Processing {sid} - {r.title}")
        try:
            full_text = fetch_plain_text_from_pdf(r.pdf_url)
        except Exception as e:
            print(f"[WARN] Failed to extract text for {sid}: {e}")
            continue
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
