from __future__ import annotations
from research_planner import plan_research
from datetime import datetime, timezone
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from source_research_agent import research_task
from collections import defaultdict
from source_comparison_agent import compare_findings
from langgraph.graph import END, START, StateGraph

from research_models import (
    ResearchReport,
    ResearchState,
    StorySynthesis,
    TopStory,
)


def create_run_id() -> str:
    """Create a traceable identifier for one research execution."""

    return f"research-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def planner_node(state: ResearchState) -> dict:
    """Create story-specific investigation tasks using the planner agent."""

    stories = state.get("top_stories", [])

    if not stories:
        raise ValueError("Planner received no top stories.")

    return {
        "research_tasks": plan_research(stories),
        "status": "researching",
    }

def source_research_node(state: ResearchState) -> dict:
    """Run independent source-research tasks concurrently."""

    tasks = state.get("research_tasks", [])

    if not tasks:
        raise ValueError("Source research received no research tasks.")

    findings = []
    errors = []

    with ThreadPoolExecutor(
        max_workers=min(4, len(tasks))
    ) as executor:
        futures = {
            executor.submit(research_task, task): task
            for task in tasks
            if task.agent_type == "source_research"
        }

        for future in as_completed(futures):
            task = futures[future]

            try:
                findings.extend(future.result())
            except Exception as error:
                message = (
                    f"Source research failed for {task.task_id}: "
                    f"{type(error).__name__}: {error}"
                )
                print(message)
                errors.append(message)

                if task.required:
                    raise RuntimeError(message) from error

    return {
        "source_findings": findings,
        "errors": errors,
        "status": "analyzing",
    }

def report_node(state: ResearchState) -> dict:
    """Create a temporary report proving the graph completed successfully."""

    syntheses = [
        StorySynthesis(
            story_title=story.title,
            executive_summary=(
                "Research has been planned for this story. "
                "Detailed source analysis will be added in later phases."
            ),
            confidence=0.0,
        )
        for story in state.get("top_stories", [])
    ]

    report = ResearchReport(
        report_id=state["run_id"],
        generated_at=datetime.now(timezone.utc),
        title="AI News Research Report",
        executive_summary=(
            "This report confirms that the initial LangGraph workflow "
            "completed successfully."
        ),
        story_syntheses=syntheses,
        methodology=[
            "Personalized stories were received from the v0.3.0 pipeline.",
            "A research task was created for each story.",
            "Detailed source retrieval and analysis will be added incrementally.",
            "Free Google News RSS and Hacker News retrieval was used.",
            "Retrieved source content was analyzed into validated findings."
        ],
        limitations=[
            "This initial workflow does not yet retrieve external sources.",
            "This initial workflow does not yet call a language model.",
            "Some publishers may block automated retrieval.",
            "Source coverage depends on free RSS and Hacker News availability."
        ],
        source_count=len(state.get("source_findings", [])),
    )

    return {
        "report": report,
        "status": "completed",
    }

def source_comparison_node(state: ResearchState) -> dict:
    """Compare findings independently for each story."""

    findings_by_story = defaultdict(list)

    for finding in state.get("source_findings", []):
        findings_by_story[finding.story_title].append(finding)

    story_titles = [
        story.title for story in state.get("top_stories", [])
    ]

    comparisons = []

    with ThreadPoolExecutor(
        max_workers=min(4, max(1, len(story_titles)))
    ) as executor:
        futures = {
            executor.submit(
                compare_findings,
                story_title,
                findings_by_story.get(story_title, []),
            ): story_title
            for story_title in story_titles
        }

        for future in as_completed(futures):
            story_title = futures[future]

            try:
                comparisons.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"Source comparison failed for {story_title}: "
                    f"{type(error).__name__}: {error}"
                ) from error

    return {
        "source_comparisons": comparisons,
        "status": "synthesizing",
    }

def build_research_graph():
    """Build and compile the initial research workflow."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("source_research", source_research_node)
    graph.add_node("source_comparison", source_comparison_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "source_research")
    graph.add_edge("source_research", "source_comparison")
    graph.add_edge("source_comparison", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_research(top_stories: list[TopStory]) -> ResearchState:
    """Run the initial workflow for a list of personalized stories."""

    if not top_stories:
        raise ValueError("At least one top story is required.")

    initial_state: ResearchState = {
        "run_id": create_run_id(),
        "started_at": datetime.now(timezone.utc),
        "top_stories": top_stories,
        "errors": [],
        "status": "planning",
    }

    graph = build_research_graph()
    return graph.invoke(initial_state)


if __name__ == "__main__":
    sample_stories = [
        TopStory(
            title="Example AI model release",
            link="https://example.com/model-release",
            description="An example model release for local testing.",
            reasoning="Useful for testing the research workflow.",
        ),
        TopStory(
            title="Example AI research result",
            link="https://example.com/research-result",
            description="An example research story for local testing.",
            reasoning="Useful for testing source analysis.",
        ),
    ]

    result = run_research(sample_stories)

    print(f"Run ID: {result['run_id']}")
    print(f"Status: {result['status']}")
    print(f"Tasks: {len(result['research_tasks'])}")
    print(f"Report: {result['report'].title}")