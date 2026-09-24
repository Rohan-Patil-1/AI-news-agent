# AI News Agent

An autonomous AI news research agent that collects recent AI stories, removes duplicates, evaluates their significance using Gemini, generates a concise daily digest, and delivers it automatically by email.

**Status:** Live — runs automatically every day via GitHub Actions.

**Latest Release:** `v0.2.0` — Automated Email Digest

---

## What This Project Does

The AI News Agent continuously monitors recent AI news from multiple sources and produces a daily **Top 5 AI News Digest**.

Instead of simply collecting or sorting articles, the agent evaluates the candidate stories and makes a reasoning-based selection of the stories it considers most significant.

Each selected story includes:

- A plain-language summary
- The original article link
- An explanation of why it was selected
- Real engagement signals when available

The entire pipeline runs automatically without requiring manual execution.

---

## Why This Is an Agentic System

This project goes beyond a traditional news scraper or RSS aggregator.

The agent receives a pool of candidate articles containing **incomplete and heterogeneous information**. Some articles have Hacker News engagement data such as points and comments, while RSS articles may have no engagement information at all.

The reasoning layer must therefore evaluate multiple signals rather than applying a fixed rule such as:

> "Sort articles by popularity."

The system asks Gemini to determine which stories are most significant based on factors such as:

- Relevance to the AI field
- Potential impact
- Novelty
- Source credibility
- Available engagement signals
- Information provided in the candidate pool

The agent then explains its selection.

Combined with autonomous scheduled execution and automated output delivery, this creates a small but practical agentic workflow.
