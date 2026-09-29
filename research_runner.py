from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from research_graph import run_research
from research_models import TopStory


REPORTS_DIR = Path("research_reports")


def convert_top5(top5: list[dict[str, Any]]) -> list[TopStory]:
    """Validate and convert v0.3.0 ranking output into research models."""

    if not top5:
        raise ValueError("Cannot research an empty Top 5 list.")

    stories: list[TopStory] = []

    for index, item in enumerate(top5, start=1):
        try:
            stories.append(TopStory.model_validate(item))
        except Exception as error:
            raise ValueError(
                f"Top 5 story {index} failed research validation: {error}"
            ) from error

    return stories


def render_report(result: dict[str, Any]) -> str:
    """Render the current research result as a readable Markdown report."""

    report = result["report"]
    lines = [
        f"# {report.title}",
        "",
        f"**Run ID:** `{result['run_id']}`",
        f"**Status:** `{result['status']}`",
        f"**Generated:** {report.generated_at.isoformat()}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Story Research Status",
        "",
    ]

    for index, synthesis in enumerate(report.story_syntheses, start=1):
        lines.extend(
            [
                f"### {index}. {synthesis.story_title}",
                "",
                synthesis.executive_summary,
                "",
                f"**Confidence:** {synthesis.confidence:.2f}",
                "",
            ]
        )

    lines.extend(["## Methodology", ""])
    lines.extend(f"- {item}" for item in report.methodology)

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)

    lines.extend(
        [
            "",
            "## Research Tasks",
            "",
            f"Total planned tasks: {len(result.get('research_tasks', []))}",
            "",
        ]
    )

    for task in result.get("research_tasks", []):
        lines.extend(
            [
                f"- **{task.task_id}** — {task.objective}",
                "",
            ]
        )

    return "\n".join(lines)


def run_top5_research(top5: list[dict[str, Any]]) -> str:
    """Run research for the v0.3.0 Top 5 and save a Markdown report."""

    stories = convert_top5(top5)
    result = run_research(stories)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / f"{date.today().isoformat()}.md"
    report_path.write_text(render_report(result), encoding="utf-8")

    print(f"Wrote research report to {report_path}")
    return os.fspath(report_path)