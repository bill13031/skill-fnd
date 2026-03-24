from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class DuckDuckGoSearcher:
    def __init__(self) -> None:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ImportError(
                "duckduckgo-search is required for DuckDuckGoSearcher. Install it first to use the web_search skill."
            ) from exc
        self._ddgs_cls = DDGS

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not query.strip():
            return []
        with self._ddgs_cls() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        normalized: List[SearchResult] = []
        for item in results:
            normalized.append(
                SearchResult(
                    title=str(item.get("title", "")).strip(),
                    url=str(item.get("href", "")).strip(),
                    snippet=str(item.get("body", "")).strip(),
                )
            )
        return normalized

    @staticmethod
    def format_results(results: List[SearchResult]) -> str:
        if not results:
            return "No DuckDuckGo results."
        lines: List[str] = []
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result.title}")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   Snippet: {result.snippet}")
        return "\n".join(lines)
