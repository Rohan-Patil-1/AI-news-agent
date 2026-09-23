import os
import json
import datetime
import time
import feedparser
import requests
from google import genai

RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "https://arstechnica.com/ai/feed/",
]

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_QUERY = "AI OR LLM OR agent OR machine learning"
HOURS_LOOKBACK = 48

OUTPUT_DIR = "digests"

def fetch_rss():
    articles = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source = feed.feed.get("title", url)
        for entry in feed.entries[:8]:
            articles.append({
                "title": entry.get("title", "").strip(),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", "")[:400],
                "source": source,
                "engagement": None,
            })
    return articles

def fetch_hn():
    since = int(
        (datetime.datetime.utcnow() - datetime.timedelta(hours=HOURS_LOOKBACK)).timestamp()
    )

    params = {
        "query": HN_QUERY,
        "tags": "story",
        "numericFilters": f"created_at_i>{since}",
    }
    resp = requests.get(HN_SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    hits = resp.json().get("hits", [])    

    articles = []
    for hit in hits[:20]:
        if not hit.get("url"):
            continue
        articles.append({
            "title": hit.get("title", "").strip(),
            "link": hit.get("url"),
            "summary": "",
            "source": "Hacker News",
            "engagement": f"{hit.get('points', 0)} points, {hit.get('num_comments', 0)} comments",
        })
    return articles

def dedupe(articles):
    seen = set()
    unique = []
    for a in articles:
        key = a["title"].lower()[:60]
        if key in seen or not a["title"]:
            continue
        seen.add(key)
        unique.append(a)
    return unique

RANKING_PROMPT = """You are an AI news analyst. Below is a list of candidate
articles from the last 48 hours, some with real engagement data (points,
comments) and some without.

Pick the TOP 5 by genuine significance to the AI field - not just
popularity. When an article has engagement numbers, weigh them as one
signal among several (novelty, impact, credibility of source), and cite
the actual numbers in your reasoning when you use them. Do not invent
engagement numbers for articles that don't have any.

Return ONLY valid JSON, no markdown fences, matching this schema:
[
  {{
    "title": "...",
    "link": "...",
    "description": "one or two sentence plain-language summary",
    "reasoning": "one or two sentences on why this made the top 5, grounded in what you were given"
  }}
]

Candidate articles:
{articles_json}
"""

def rank_and_reason(articles):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = RANKING_PROMPT.format(
        articles_json=json.dumps(articles, indent=2)[:12000]
    )

    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
    ]

    last_error = None

    for model in models_to_try:
        for attempt in range(4):
            try:
                print(
                    f"Calling Gemini model {model} "
                    f"(attempt {attempt + 1}/4)..."
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                text = response.text.strip()

                # Remove markdown fences if Gemini happens to add them
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]

                if text.endswith("```"):
                    text = text[:-3]

                text = text.strip()

                result = json.loads(text)

                if not isinstance(result, list):
                    raise ValueError("Gemini response was not a JSON list.")

                if len(result) == 0:
                    raise ValueError("Gemini returned an empty result.")

                print(
                    f"Gemini succeeded using {model} "
                    f"on attempt {attempt + 1}."
                )

                return result[:5]

            except Exception as e:
                last_error = e

                print(
                    f"Gemini request failed for {model}: "
                    f"{type(e).__name__}: {e}"
                )

                if attempt < 3:
                    wait_time = 5 * (2 ** attempt)
                    print(
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

        print(
            f"Model {model} failed after 4 attempts. "
            f"Trying next model..."
        )

    raise RuntimeError(
        f"Gemini failed after all model/ retry attempts: {last_error}"
    )

def write_digest(top5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = os.path.join(OUTPUT_DIR, f"{today}.md")

    lines = [f"# AI News Digest — {today}\n"]
    for i, item in enumerate(top5, start=1):
        lines.append(f"## {i}. [{item['title']}]({item['link']})\n")
        lines.append(f"{item['description']}\n")
        lines.append(f"**Why it made the top 5:** {item['reasoning']}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote digest to {path}")
    return path

def main():
    rss_articles = fetch_rss()
    hn_articles = fetch_hn()
    candidates = dedupe(rss_articles + hn_articles)

    if not candidates:
        print("No candidate articles found — check feed URLs / network access.")
        return

    top5 = rank_and_reason(candidates)
    write_digest(top5)


if __name__ == "__main__":
    main()
