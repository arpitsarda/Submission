#!/usr/bin/env python3
"""Task 3 — reads Task 2's customer_segments.json (plus Task 1's raw
signals, for numeric grounding) and produces: the identified primary
gap/opportunity, one named product recommendation, and a 2-sentence hook.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MODEL = "gemini-3.5-flash-lite"
REQUIRED_KEYS = {"primary_signal", "product_recommendation", "reasoning", "hook"}

SYSTEM_PROMPT = '''You are a senior relationship-banking strategist. You will receive ONLY a customer's segmentation profile: three keys - "behavioral" (spending/cash-flow patterns: retained liquidity, supply chain concentration, labor intensity, lease reliance, cash-flow timing), "demographic" (industry, location, corporate structure - not derived from transactions), and "psychographic" (inferred traits: cost resilience, growth orientation, digital maturity - framed as inference, not certainty). Each item in behavioral/psychographic is a short tagged phrase naming a metric and its value; treat these as your complete evidence base - there is no separate raw-data file.

Task:
1. From the behavioral and psychographic items given, identify the single one that represents the most significant, actionable gap or opportunity for this client.
2. Recommend exactly ONE specific, named bank product that follows logically from that item. Do not pick a product first and justify it after - the evidence must come first in your reasoning.
3. Write a 2-sentence hook a Relationship Manager could say on a call or type at the top of an email. Sentence 1 states the observed evidence. Sentence 2 makes the specific ask or offer. Ground every number in the hook in the actual supplied data - never invent a figure.

Do not claim trends, sustained performance, or anything not supported by one month of observed data. Return valid JSON only, with exactly these top-level keys: "primary_signal" (the signal key you selected), "product_recommendation" (the product name), "reasoning" (2-3 sentences connecting signal to product), "hook" (the 2-sentence hook, as one string). No markdown, no extra keys - specifically, do not echo "ranked_by_magnitude" or any other input field back in your response.'''


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}")


def strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def parse_and_validate(text: str) -> dict:
    cleaned = strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: model response was not valid JSON: {exc}\n\nRaw response:\n{cleaned}")

    if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed.keys()):
        raise SystemExit(
            f"ERROR: response must contain at least these keys: {sorted(REQUIRED_KEYS)}; "
            f"received: {sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}"
        )
    extra = set(parsed.keys()) - REQUIRED_KEYS
    if extra:
        print(f"NOTE: model returned extra field(s) {sorted(extra)} — dropped, not saved.")
    return {k: parsed[k] for k in REQUIRED_KEYS}  # keep only what was asked for


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit('ERROR: GEMINI_API_KEY is not set. Run:\nexport GEMINI_API_KEY="your-key-here"')

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("ERROR: google-genai is not installed. Run: pip install -r requirements.txt") from exc

    root = project_root()
    segments = load_json(root / "output" / "customer_segments.json")  # the only input - self-sufficient by design
    output_path = root / "output" / "hook.json"

    user_message = json.dumps(segments, ensure_ascii=False)

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=500,
            ),
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: Gemini API call failed: {exc}") from exc

    response_text = (response.text or "").strip()
    if not response_text:
        raise SystemExit("ERROR: Gemini returned an empty response.")

    parsed = parse_and_validate(response_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(parsed, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
