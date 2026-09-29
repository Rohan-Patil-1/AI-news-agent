from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse
from dotenv import load_dotenv
import feedparser
import requests
from bs4 import BeautifulSoup

load_dotenv()

USER_AGENT = (
    "AI-News-Agent/0.4 research source reader; "
    "contact the repository maintainer before automated access"
)

REQUEST_TIMEOUT_SECONDS = 15
MAX_ARTICLE_CHARS = 12000
MAX_RESULTS_PER_QUERY = 5

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

HACKER_NEWS_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


@dataclass(frozen=True)
class SourceCandidate:
    """A source discovered by the free retrieval layer."""

    title: str
    url: str
    publisher: str
    published_at: datetime | None
    source_type: str
    snippet: str
    content: str


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _publisher_from_url(url: str) -> str:
    hostname = urlparse(url).netloc.lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname or "unknown"


def _extract_article_text(response: requests.Response) -> str:
    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside"]
    ):
        element.decompose()

    paragraphs = [
        _clean_text(paragraph.get_text(" ", strip=True))
        for paragraph in soup.find_all("p")
    ]

    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= 40
    ]

    return "\n\n".join(paragraphs)[:MAX_ARTICLE_CHARS]


def fetch_url_content(url: str) -> str:
    """Fetch readable text from one URL with bounded network behavior."""

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "text/html" not in content_type:
        return ""

    return _extract_article_text(response)


def _parse_feed_date(entry: object) -> datetime | None:
    parsed_time = getattr(entry, "published_parsed", None)

    if not parsed_time:
        return None

    return datetime(
        parsed_time.tm_year,
        parsed_time.tm_mon,
        parsed_time.tm_mday,
        parsed_time.tm_hour,
        parsed_time.tm_min,
        parsed_time.tm_sec,
        tzinfo=timezone.utc,
    )


def search_google_news(query: str) -> list[SourceCandidate]:
    """Search Google News RSS without requiring an API key."""

    url = GOOGLE_NEWS_RSS_URL.format(query=quote_plus(query))
    feed = feedparser.parse(url)
    candidates: list[SourceCandidate] = []

    for entry in feed.entries[:MAX_RESULTS_PER_QUERY]:
        article_url = entry.get("link", "").strip()

        if not article_url:
            continue

        candidates.append(
            SourceCandidate(
                title=_clean_text(entry.get("title", "")),
                url=article_url,
                publisher=_publisher_from_url(article_url),
                published_at=_parse_feed_date(entry),
                source_type="news",
                snippet=_clean_text(entry.get("summary", ""))[:1000],
                content="",
            )
        )

    return candidates


def search_hacker_news(query: str) -> list[SourceCandidate]:
    """Search Hacker News discussions without requiring an API key."""

    response = requests.get(
        HACKER_NEWS_SEARCH_URL,
        params={
            "query": query,
            "tags": "story",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    candidates: list[SourceCandidate] = []

    for hit in response.json().get("hits", [])[:MAX_RESULTS_PER_QUERY]:
        article_url = (hit.get("url") or "").strip()

        if not article_url:
            continue

        candidates.append(
            SourceCandidate(
                title=_clean_text(hit.get("title", "")),
                url=article_url,
                publisher=_publisher_from_url(article_url),
                published_at=None,
                source_type="community",
                snippet=(
                    f"{hit.get('points', 0)} points, "
                    f"{hit.get('num_comments', 0)} comments"
                ),
                content="",
            )
        )

    return candidates


def discover_sources(
    story_title: str,
    search_queries: list[str],
) -> list[SourceCandidate]:
    """Discover and enrich sources for one research task."""

    candidates: list[SourceCandidate] = []
    seen_urls: set[str] = set()

    queries = search_queries or [story_title]

    for query in queries:
        discovered = search_google_news(query)

        try:
            discovered.extend(search_hacker_news(query))
        except requests.RequestException as error:
            print(f"Hacker News search failed for {query!r}: {error}")

        for candidate in discovered:
            if candidate.url in seen_urls:
                continue

            seen_urls.add(candidate.url)

            try:
                content = fetch_url_content(candidate.url)
            except requests.RequestException as error:
                print(f"Source fetch failed for {candidate.url}: {error}")
                content = ""

            candidates.append(
                SourceCandidate(
                    title=candidate.title,
                    url=candidate.url,
                    publisher=candidate.publisher,
                    published_at=candidate.published_at,
                    source_type=candidate.source_type,
                    snippet=candidate.snippet,
                    content=content,
                )
            )

            if len(candidates) >= MAX_RESULTS_PER_QUERY:
                return candidates

    return candidates


if __name__ == "__main__":
    results = discover_sources(
        story_title="AI model release",
        search_queries=["AI model release"],
    )

    print(f"Discovered {len(results)} source(s).")

    for source in results:
        print(
            f"- {source.publisher}: {source.title} "
            f"({len(source.content)} content characters)"
        )