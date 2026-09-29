from __future__ import annotations
from research_planner import plan_research
from datetime import datetime, timezone
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from source_research_agent import research_task
from collections import defaultdict
from source_comparison_agent import compare_findings
from langgraph.graph import END, START, StateGraph
from technical_analysis_agent import analyze_technical_task
from synthesis_agent import synthesize_story
from report_agent import create_report

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

    with ThreadPoolExecutor(max_workers=1) as executor:
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
    """Create the final report from validated story syntheses."""

    syntheses = state.get("syntheses", [])

    if not syntheses:
        raise ValueError("Report generation received no story syntheses.")

    report = create_report(
        run_id=state["run_id"],
        syntheses=syntheses,
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
        max_workers=1
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

def technical_analysis_node(state: ResearchState) -> dict:
    """Run planned technical-analysis tasks concurrently."""

    tasks = [
        task
        for task in state.get("research_tasks", [])
        if task.agent_type == "technical_analysis"
    ]

    if not tasks:
        return {
            "technical_analyses": [],
            "status": "analyzing",
        }

    stories_by_title = {
        story.title: story
        for story in state.get("top_stories", [])
    }

    findings_by_story = defaultdict(list)

    for finding in state.get("source_findings", []):
        findings_by_story[finding.story_title].append(finding)

    analyses = []

    with ThreadPoolExecutor(
        max_workers=1
    ) as executor:
        futures = {}

        for task in tasks:
            story = stories_by_title.get(task.story_title)

            if story is None:
                raise ValueError(
                    f"No Top 5 story found for task {task.task_id}."
                )

            future = executor.submit(
                analyze_technical_task,
                task,
                story.model_dump(mode="json"),
                findings_by_story.get(task.story_title, []),
            )
            futures[future] = task

        for future in as_completed(futures):
            task = futures[future]

            try:
                analyses.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    f"Technical analysis failed for {task.task_id}: "
                    f"{type(error).__name__}: {error}"
                ) from error

    return {
        "technical_analyses": analyses,
        "status": "analyzing",
    }

def synthesis_node(state: ResearchState) -> dict:
    """Create one evidence-aware synthesis per story."""

    stories = state.get("top_stories", [])

    findings_by_story = defaultdict(list)
    for finding in state.get("source_findings", []):
        findings_by_story[finding.story_title].append(finding)

    comparisons_by_story = {
        comparison.story_title: comparison
        for comparison in state.get("source_comparisons", [])
    }

    technical_by_story = {
        analysis.story_title: analysis
        for analysis in state.get("technical_analyses", [])
    }

    syntheses = []
    errors = []

    with ThreadPoolExecutor(
        max_workers=1
    ) as executor:
        futures = {
            executor.submit(
                synthesize_story,
                story,
                findings_by_story.get(story.title, []),
                comparisons_by_story.get(story.title),
                technical_by_story.get(story.title),
            ): story
            for story in stories
        }

        for future in as_completed(futures):
            story = futures[future]

            try:
                syntheses.append(future.result())
            except Exception as error:
                message = (
                    f"Synthesis failed for {story.title}: "
                    f"{type(error).__name__}: {error}"
                )
                print(message)
                errors.append(message)
                raise RuntimeError(message) from error

    return {
        "syntheses": syntheses,
        "errors": errors,
        "status": "synthesizing",
    }

def build_research_graph():
    """Build and compile the initial research workflow."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("source_research", source_research_node)
    graph.add_node("technical_analysis", technical_analysis_node)
    graph.add_node("source_comparison", source_comparison_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "source_research")
    graph.add_edge("source_research", "technical_analysis")
    graph.add_edge("technical_analysis", "source_comparison")
    graph.add_edge("source_comparison", "synthesis")
    graph.add_edge("synthesis", "report")
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