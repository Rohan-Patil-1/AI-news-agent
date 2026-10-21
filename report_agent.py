from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types

from research_models import ResearchReport, StorySynthesis

load_dotenv(override=True)


REPORT_PROMPT = """You are a research report editor.

Create a structured report from the validated story syntheses below.

Rules:
1. Do not introduce facts that are not present in the syntheses.
2. Clearly describe the methodology.
3. Clearly state limitations.
4. Do not claim that all sources are authoritative.
5. Keep the report useful for a technically informed reader.
6. Return only valid JSON.

Return this schema:

{
  "title": "AI News Research Report",
  "executive_summary": "overall summary",
  "methodology": [],
  "limitations": []
}

Story syntheses:
{syntheses_json}
"""


def _clean_json(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def create_report(
    run_id: str,
    syntheses: list[StorySynthesis],
    source_count: int,
    max_attempts: int = 3,
) -> ResearchReport:
    """Create the final validated report metadata."""

    if not syntheses:
        raise ValueError("At least one story synthesis is required.")

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for report generation."
        )

    prompt = REPORT_PROMPT.format(
        syntheses_json=json.dumps(
            [synthesis.model_dump(mode="json") for synthesis in syntheses],
            indent=2,
        )
    )

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=os.environ.get(
                    "GEMINI_REPORT_MODEL",
                    "gemini-3.8-flash",
                ),
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise ValueError("Report generation returned an empty response.")

            raw_result = json.loads(_clean_json(response.text))

            report = ResearchReport(
                report_id=run_id,
                generated_at=datetime.now(timezone.utc),
                title=raw_result["title"],
                executive_summary=raw_result["executive_summary"],
                story_syntheses=syntheses,
                methodology=raw_result["methodology"],
                limitations=raw_result["limitations"],
                source_count=source_count,
            )

            print(f"Final report created on attempt {attempt}.")
            return report

        except Exception as error:
            last_error = error
            print(
                f"Report attempt {attempt}/{max_attempts} failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Report generation failed after {max_attempts} attempts: "
        f"{last_error}"
    ) from last_error