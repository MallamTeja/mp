import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime, timezone

import arxiv
from arxiv2text import arxiv_to_text


# ── Paths ────────────────────────────────────────────────────────────────────
PNOS1_PATH = Path("papernumbers1.json")
PNOS2_PATH = Path("papernumbers2.json")

DATASET_PATH = Path("dataset.json")


# ── arXiv client ─────────────────────────────────────────────────────────────
client = arxiv.Client()
MAX_TEXT_CHARS = 2_000_000


# ── Paper ID loading ─────────────────────────────────────────────────────────
def load_paper_ids_from_file(path: Path, batch_id: int) -> List[Dict]:
    if not path.exists():
        print(f"[WARN] {path} not found, skipping")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = [pid.strip() for pid in data.get("paper_ids", []) if pid.strip()]
    return [{"arxiv_id": pid, "batch": batch_id} for pid in ids]


def load_all_papers_with_batches() -> List[Dict]:
    papers: List[Dict] = []
    papers.extend(load_paper_ids_from_file(PNOS1_PATH, batch_id=1))
    papers.extend(load_paper_ids_from_file(PNOS2_PATH, batch_id=2))
    # No Tier 3 file for now
    return papers


# ── Dataset helpers ──────────────────────────────────────────────────────────
def load_existing_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)


def get_existing_ids(dataset: List[Dict]) -> Set[str]:
    return {str(item["arxiv_id"]) for item in dataset if "arxiv_id" in item}


# ── arXiv fetching ───────────────────────────────────────────────────────────
def fetch_metadata_for_ids(ids: List[str]):
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


# ── Feature helpers ───────────────────────────────────────────────────────────
def compute_years_since_published(published_dt: Optional[datetime]) -> float:
    if not published_dt:
        return 0.0
    now = datetime.now(timezone.utc)
    delta_years = (now - published_dt).total_seconds() / (365.25 * 24 * 3600)
    return max(delta_years, 0.0)


def build_paper_number_mapping(all_ids: List[str]) -> Dict[str, int]:
    sorted_ids = sorted(set(all_ids))
    return {pid: idx + 1 for idx, pid in enumerate(sorted_ids)}


# ── Record builder ───────────────────────────────────────────────────────────
def result_to_obj(
    arxiv_id: str,
    r: arxiv.Result,
    full_text: str,
    paper_number: Optional[int],
) -> Dict:
    if not full_text.strip():
        full_text = r.summary or ""

    author_count = len(r.authors) if r.authors else 0
    published_dt = r.published if isinstance(r.published, datetime) else None
    years_since_published = compute_years_since_published(published_dt)

    summary = r.summary or ""
    summary_len = len(summary)
    text_len = len(full_text)

    return {
        "arxiv_id": arxiv_id,
        "paper_number": paper_number,
        # scores all start at 0 and will be filled by distillation.py
        "manual_score": 0,
        "novelty_score": 0,
        "rigor_score": 0,
        "impact_score": 0,
        "title": r.title,
        "author_count": author_count,
        "summary": summary,
        "summary_len": summary_len,
        "primary_category": r.primary_category,
        "categories": list(r.categories),
        "years_since_published": years_since_published,
        "text": full_text,
        "text_len": text_len,
    }


# ── Flush to disk ─────────────────────────────────────────────────────────────
def flush_buffer(raw_buffer: List[Dict], dataset_path: Path) -> None:
    if not raw_buffer:
        return

    dataset = load_existing_dataset(dataset_path)
    dataset.extend(raw_buffer)

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(
        f"[FLUSH] Saved {len(raw_buffer)} raw records. "
        f"Total raw: {len(dataset)}"
    )
    raw_buffer.clear()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    paper_records = load_all_papers_with_batches()
    if not paper_records:
        print("No paper IDs found in papernumbers1.json or papernumbers2.json.")
        return

    all_ids = [rec["arxiv_id"] for rec in paper_records]
    paper_number_map = build_paper_number_mapping(all_ids)

    dataset_existing = load_existing_dataset(DATASET_PATH)
    existing_ids = get_existing_ids(dataset_existing)

    to_process: List[Dict] = [
        rec for rec in paper_records if rec["arxiv_id"] not in existing_ids
    ]
    if not to_process:
        print("No new paper IDs found. Nothing to do.")
        return

    limit = len(to_process)
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"[WARN] Invalid limit argument: {sys.argv[1]}, using {limit}")

    to_process = to_process[:limit]
    ids_only = [rec["arxiv_id"] for rec in to_process]

    print(f"Processing {len(to_process)} papers...")
    results = fetch_metadata_for_ids(ids_only)
    if not results:
        print("No results returned from arXiv for the new IDs.")
        return

    result_map: Dict[str, arxiv.Result] = {
        r.entry_id.split("/")[-1]: r for r in results
    }

    raw_buffer: List[Dict] = []
    FLUSH_EVERY = 10

    try:
        for i, rec in enumerate(to_process):
            arxiv_id = rec["arxiv_id"]
            r = result_map.get(arxiv_id)
            if r is None:
                print(f"[WARN] No arxiv result for {arxiv_id}, skipping")
                continue

            print(f"[{i+1}/{len(to_process)}] {arxiv_id} - {r.title}")

            paper_number = paper_number_map.get(arxiv_id)
            full_text = fetch_plain_text_from_pdf(r.pdf_url)

            obj = result_to_obj(
                arxiv_id=arxiv_id,
                r=r,
                full_text=full_text,
                paper_number=paper_number,
            )
            raw_buffer.append(obj)

            time.sleep(0.2)

            if len(raw_buffer) >= FLUSH_EVERY:
                flush_buffer(raw_buffer, DATASET_PATH)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Flushing completed records before exit...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}. Flushing completed records before exit...")
    finally:
        flush_buffer(raw_buffer, DATASET_PATH)
        print("Done (graceful shutdown).")


if __name__ == "__main__":
    main()
