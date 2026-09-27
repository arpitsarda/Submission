#!/usr/bin/env python3
"""Task 2 — turns Task 1's signals.json into customer_segments.json.

Demographic facts (industry, location, structure) are NOT computed from
transactions — they come from the company's own KYC/CRM record. In a real
system that's a database lookup; here it's a small constant, flagged as
such in the README.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MODEL = "gemini-3.5-flash-lite"
REQUIRED_KEYS = {"behavioral", "demographic", "psychographic"}

# Not derived from transactions — comes from onboarding/KYC data.
COMPANY_PROFILE = {
    "industry": "Construction",
    "location": "Utah",
    "corporate_structure": "LLC",
}

SYSTEM_PROMPT = '''You are a senior relationship-banking and SMB credit analyst producing a compact RAG customer profile for a bank Relationship Manager.

You will receive two inputs: "company_profile" (known facts, not derived from transactions) and "signals" (8 engineered metrics from one month of transaction data).

Do not invent history, trends, customer concentration, margins, liquidity, or efficiency gains not supported by the supplied data. Where a metric's own interpretation already states a limitation (e.g. cannot separate price from volume), preserve that caveat's substance rather than dropping it.

The "behavioral" section must be a compact array with one short tagged item for EACH of these five metrics - do not omit any, do not select a subset: retained_liquidity, supply_chain_concentration, labor_intensity, lease_inefficiency, peak_cash_gap. Each item under 15 words, phrase not full sentence, e.g. "Vendor concentration: 74.7% with Intermountain Lumber" - not a flowing paragraph.

The "demographic" section must use company_profile only - never infer industry, location, or structure from transaction data.

The "psychographic" section must be a compact array with one short tagged item for EACH of these three metrics - do not omit any: resilience_alpha, growth_allocation_rate, modernization_ratio. Same terse phrase style, framed explicitly as inference not certainty, e.g. "Cost resilience: material spend moved opposite the +15% industry direction".

Strict separation: never mention resilience_alpha, growth_allocation_rate, or modernization_ratio anywhere in "behavioral" - they belong to "psychographic" only. Never mention retained_liquidity, supply_chain_concentration, labor_intensity, lease_inefficiency, or peak_cash_gap anywhere in "psychographic" - they belong to "behavioral" only. Each signal appears in exactly one section, never both, and every signal must appear somewhere - none may be silently dropped.

Write in a formal, measured, commercially useful tone. Return valid JSON only, with exactly three top-level keys: "behavioral", "demographic", "psychographic". Each value must be a concise string or compact array. Keep the entire JSON response under 300 tokens total. No filler, no repeated phrasing across sections.'''


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

    if not isinstance(parsed, dict) or set(parsed.keys()) != REQUIRED_KEYS:
        raise SystemExit(
            f"ERROR: response must have exactly these keys: {sorted(REQUIRED_KEYS)}; "
            f"received: {sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}"
        )
    return parsed


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
    signals = load_json(root / "output" / "signals.json")
    output_path = root / "output" / "customer_segments.json"

    user_message = json.dumps(
        {"company_profile": COMPANY_PROFILE, "signals": signals["metrics"]},
        ensure_ascii=False,
    )

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=450,  # headroom above the 300-token target; not a hard truncation point
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

    actual_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
    print(json.dumps(parsed, indent=2))
    print(f"\nActual token count (Gemini's own tokenizer): {actual_tokens}")
    if actual_tokens and actual_tokens > 300:
        print("WARNING: over the 300-token budget — tighten the prompt and rerun.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
