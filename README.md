# AI-news-agent

# Daily AI News Agent

A small, single-agent system that autonomously researches AI news every day, picks the 5 most significant stories, explains *why* each one made the cut, and writes a markdown digest — running unattended on a daily schedule via GitHub Actions.

**Status:** Live — running daily, no manual trigger required.

## Why this is "agentic" and not just a script

The agent makes an actual judgment call. Given a pool of candidate articles with mixed, partial signals — some carry real engagement data from Hacker News (points, comments), others (RSS) carry none — it has to decide what's genuinely significant rather than following a fixed rule like "sort by points." That reasoning step, combined with running on its own schedule with no human triggering each run, is what makes this agentic rather than a static news aggregator.

## Architecture

1. **Sourcing** — pulls recent articles from AI-focused RSS feeds (TechCrunch, VentureBeat, MIT Technology Review, Ars Technica) and from Hacker News' free Algolia search API. HN stories come with real engagement numbers, which gives the reasoning step something concrete to cite instead of inventing importance.
2. **Dedupe** — collapses the same story when it's reported by multiple outlets.
3. **Reasoning** — sends the candidate pool to Gemini with instructions to pick the top 5 by genuine significance and ground its reasoning in the real signals it was given, not fabricated ones.
4. **Output** — writes a dated markdown digest to `digests/`.
5. **Automation** — a GitHub Actions workflow runs this daily and commits the digest straight back to the repo, so the commit history is itself a running log of the agent's output.

## Example output

Each run produces a file like `digests/2026-09-21.md`:

```markdown
# AI News Digest — 2026-09-21

## 1. [Article title](https://...)
Plain-language summary of what happened.

**Why it made the top 5:** Reasoning grounded in engagement data
and/or significance to the field.
```

Browse the [`digests/`](./digests) folder for the full running history.

## Setup (to run locally)

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `export GEMINI_API_KEY=your_key_here`
5. `python agent.py`

Your digest lands in `digests/YYYY-MM-DD.md`.

## How the automation is wired up

- `.github/workflows/daily.yml` runs on a daily cron schedule (and can also be triggered manually from the Actions tab via `workflow_dispatch`).
- The workflow installs dependencies, runs `agent.py`, and commits any new digest back to the repo using a bot-authored commit.
- The Gemini key is stored as a GitHub Actions repository secret (`GEMINI_API_KEY`) — it's never present in the code or committed to the repo.

## Tech stack

| Piece | Tool | Why |
|---|---|---|
| Reasoning | Gemini (`gemini-flash-latest`), `google-genai` SDK | Free tier, and the `-latest` alias avoids breakage as Google ships new model versions |
| News sourcing | `feedparser` (RSS), Hacker News Algolia API | Both free, no auth required |
| HTTP | `requests` | For the one raw API call (HN) |
| Automation | GitHub Actions (cron + `workflow_dispatch`) | Free scheduled compute, no server to maintain |
| Output | Markdown, committed to the repo | Simple, versioned, human-readable |

## Extending this later

- Swap the flat top-5 picker for a LangGraph loop that can decide to pull more candidates if the first batch looks thin.
- Add a second "critic" agent that checks whether each reasoning claim actually holds up against the source article — early practice for the multi-agent debate pattern used in larger agentic projects.
- Push the digest to Slack or email instead of (or alongside) committing to the repo.
- Swap Gemini for a locally hosted model via Ollama to practice self-hosted inference and compare cost/latency tradeoffs.
- Expose the ranking/reasoning step as an MCP tool so other agents can call it.
