import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = Path("dataset.json")
GEMINI_MODEL  = "gemini-2.5-flash-lite"
MAX_TEXT_CHARS = 300_000
MAX_RETRIES    = 3
RETRY_DELAY    = 70   # seconds between retries (increased for free tier cooldown)
CALL_DELAY     = 10.0  # seconds between successful calls (increased to stay under limits)
FLUSH_EVERY    = 1   # write to disk after every N scored records

# ── Rubric ────────────────────────────────────────────────────────────────────
RUBRIC = """
You are a world-class AI/ML research evaluator. Score the paper below on three dimensions.
Be strict and consistent. Most papers score 30–70. Only landmark work scores above 85.

1. Novelty (0–33)
   28–33: Fundamentally new concept the field has not seen
   18–27: Clear novel contribution or meaningful combination
   9–17:  Incremental improvement, limited originality
   0–8:   Rehash of known ideas, trivially obvious

2. Rigor (0–33)
   28–33: Thorough experiments, strong baselines, rigorous ablations
   18–27: Solid methodology with minor gaps
   9–17:  Weak baselines, limited experiments
   0–8:   No real experiments, flawed methodology

3. Impact (0–34)
   29–34: Has or will fundamentally change how the field works
   19–28: Useful to many researchers, likely to be built upon
   9–18:  Niche usefulness, limited adoption
   0–8:   Unclear benefit, no practical or theoretical value

RULES:
- Return ONLY a valid JSON object. No markdown, no text outside the JSON.
- Integers only. total must equal novelty + rigor + impact exactly.
- reasoning: 2–3 sentences, brutally honest.

FORMAT:
{"novelty": <int>, "rigor": <int>, "impact": <int>, "total": <int>, "reasoning": "<string>"}
""".strip()


# ── Gemini client ─────────────────────────────────────────────────────────────
_client: Optional[genai.Client] = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Dataset I/O ───────────────────────────────────────────────────────────────
def load() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.loads(f.read().strip())

def save(data: List[Dict]) -> None:
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Scoring ───────────────────────────────────────────────────────────────────
def needs_scoring(rec: Dict) -> bool:
    """Only score rows where ALL four score fields are exactly 0."""
    return (
        rec.get("manual_score",  0) == 0 and
        rec.get("novelty_score", 0) == 0 and
        rec.get("rigor_score",   0) == 0 and
        rec.get("impact_score",  0) == 0
    )


def parse_response(raw: str) -> Dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found: {raw[:200]}")

    p = json.loads(match.group())

    for key in ("novelty", "rigor", "impact", "total", "reasoning"):
        if key not in p:
            raise ValueError(f"Missing key: {key}")

    if not (0 <= p["novelty"] <= 33): raise ValueError(f"novelty out of range: {p['novelty']}")
    if not (0 <= p["rigor"]   <= 33): raise ValueError(f"rigor out of range: {p['rigor']}")
    if not (0 <= p["impact"]  <= 34): raise ValueError(f"impact out of range: {p['impact']}")

    # Never trust the model's own arithmetic
    correct_total = p["novelty"] + p["rigor"] + p["impact"]
    if p["total"] != correct_total:
        print(f"  [WARN] total corrected {p['total']} → {correct_total}")
        p["total"] = correct_total

    if not (1 <= p["total"] <= 100):
        raise ValueError(f"total out of range: {p['total']}")

    return p


def call_llm(title: str, summary: str, text: str) -> Optional[Dict]:
    payload = (text.strip() or summary.strip())[:MAX_TEXT_CHARS]
    if not payload:
        print("  [SKIP] No text or summary available")
        return None

    prompt = f"{RUBRIC}\n\nTITLE: {title}\n\nCONTENT:\n{payload}"
    client = get_client()
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            result = parse_response(resp.text)
            print(
                f"  Score: {result['total']} "
                f"(N={result['novelty']} R={result['rigor']} I={result['impact']})"
            )
            return result
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] Parse error: {e}")
        except Exception as e:
            last_err = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] API error: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    print(f"  [FAIL] All retries exhausted: {last_err}")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Pre-flight: validate API key before loading anything
    get_client()

    data    = load()
    pending = [i for i, r in enumerate(data) if needs_scoring(r)]

    if not pending:
        print("Nothing to score — all records already have scores.")
        return

    print(f"dataset.json: {len(data)} total | {len(pending)} to score\n")

    scored  = 0
    failed  = 0
    unsaved = 0  # counts scored records not yet flushed

    try:
        for pos, idx in enumerate(pending, 1):
            rec = data[idx]
            print(f"[{pos}/{len(pending)}] {rec.get('arxiv_id', idx)} — {rec.get('title','')[:70]}")

            result = call_llm(
                title   = rec.get("title",   ""),
                summary = rec.get("summary", ""),
                text    = rec.get("text",    ""),
            )

            if result is None:
                failed += 1
                time.sleep(CALL_DELAY)
                continue

            # Write scores directly into the record
            data[idx]["manual_score"]  = result["total"]
            data[idx]["novelty_score"] = result["novelty"]
            data[idx]["rigor_score"]   = result["rigor"]
            data[idx]["impact_score"]  = result["impact"]

            scored  += 1
            unsaved += 1
            time.sleep(CALL_DELAY)

            # Flush every FLUSH_EVERY scored records — no partial saves
            if unsaved >= FLUSH_EVERY:
                save(data)
                unsaved = 0
                print(f"  [FLUSH] Saved — scored: {scored} | failed: {failed} | remaining: {len(pending) - pos}")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted. Saving progress...")
        if unsaved > 0:
            save(data)
        print(f"Saved. Scored {scored} this run. Re-run to continue.")
        return

    except Exception as e:
        print(f"\n[ERROR] {e}. Saving progress...")
        if unsaved > 0:
            save(data)
        print(f"Saved. Scored {scored} this run. Re-run to continue.")
        return

    # Final flush for whatever is left
    if unsaved > 0:
        save(data)

    print(f"\nDone. Scored: {scored} | Failed: {failed}")
    if failed:
        print(f"{failed} papers still at 0 — re-run to retry them.")


if __name__ == "__main__":
    main()