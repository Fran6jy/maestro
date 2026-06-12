"""
maestro/data/connectors/newsapi.py
===================================
NewsAPI connector delivering Forex-relevant headlines
pre-processed and ready for FinBERT embedding.

Pipeline per article:
  raw JSON → clean text → dedup → score (TextBlob polarity) → store

The full FinBERT / GPT-4o NLP pipeline lives in agents/sentiment/.
This connector is purely responsible for retrieval and basic cleaning.

Dependencies: requests, pandas, textblob
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from maestro.config.config import get, get_secret

logger = logging.getLogger(__name__)


class NewsConnector:
    """
    Fetches financial news headlines from NewsAPI and returns
    a clean DataFrame ready for NLP processing.

    Usage
    -----
    >>> news = NewsConnector()
    >>> df = news.fetch_recent(lookback_days=7)
    >>> df.columns
    Index(['published_at', 'source', 'headline', 'description',
           'url', 'query', 'text_hash', 'polarity'])
    """

    def __init__(self) -> None:
        self.base_url    = get("data_sources.newsapi.base_url")
        self.api_key     = get_secret(get("data_sources.newsapi.token_env_var", "NEWSAPI_KEY"))
        self.queries     = get("data_sources.newsapi.queries", [])
        self.page_size   = get("data_sources.newsapi.page_size", 100)
        self.lookback    = get("data_sources.newsapi.lookback_days", 30)
        self._seen_hashes: set[str] = set()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    def fetch_recent(
        self,
        lookback_days: int | None = None,
        queries: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch recent articles for all configured Forex queries.

        Returns
        -------
        pd.DataFrame sorted by published_at descending, deduplicated by content hash.
        """
        days = lookback_days or self.lookback
        q_list = queries or self.queries
        from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        all_articles: list[dict] = []

        for query in q_list:
            try:
                articles = self._fetch_query(query, from_date)
                all_articles.extend(articles)
                logger.info("  Query '%s': %d articles", query, len(articles))
                time.sleep(0.5)
            except NewsError as exc:
                logger.error("Failed query '%s': %s", query, exc)

        if not all_articles:
            return pd.DataFrame()

        df = self._to_dataframe(all_articles)
        df = self._deduplicate(df)
        df = self._add_polarity(df)

        logger.info(
            "News dataset: %d unique articles (%d queries, last %d days)",
            len(df), len(q_list), days
        )
        return df

    def _fetch_query(self, query: str, from_date: str) -> list[dict]:
        """Fetch a single page of articles for one query.
        Uses top-headlines endpoint (free tier) with fallback to everything.
        """
        # Free tier only supports top-headlines; everything requires paid plan
        url = f"{self.base_url}/top-headlines"
        params = {
            "q":        query,
            "language": "en",
            "pageSize": min(self.page_size, 100),
            "apiKey":   self.api_key,
        }
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "ok":
                    raise NewsError(f"API error: {data.get('message', 'unknown')}")
                return data.get("articles", [])
            except requests.RequestException as exc:
                if attempt == 2:
                    raise NewsError(str(exc)) from exc
                time.sleep(2 ** attempt)
        return []

    # ── Processing ────────────────────────────────────────────────────────────
    def _to_dataframe(self, articles: list[dict]) -> pd.DataFrame:
        records = []
        for art in articles:
            headline = (art.get("title") or "").strip()
            desc     = (art.get("description") or "").strip()
            if not headline or headline == "[Removed]":
                continue
            records.append({
                "published_at": pd.Timestamp(art.get("publishedAt"), tz="UTC"),
                "source":       art.get("source", {}).get("name", ""),
                "headline":     headline,
                "description":  desc,
                "url":          art.get("url", ""),
                "full_text":    f"{headline}. {desc}".strip(". "),
            })
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["text_hash"] = df["full_text"].apply(
            lambda t: hashlib.md5(t.lower().encode()).hexdigest()
        )
        return df.sort_values("published_at", ascending=False).reset_index(drop=True)

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles by content hash."""
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["text_hash"])
        new_mask = ~df["text_hash"].isin(self._seen_hashes)
        self._seen_hashes.update(df["text_hash"].tolist())
        return df[new_mask].reset_index(drop=True)

    @staticmethod
    def _add_polarity(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a fast TextBlob polarity score as a lightweight pre-filter.
        Articles with polarity ≈ 0 and short length are likely not market-moving.
        Full FinBERT scoring happens downstream in agents/sentiment/.
        """
        if df.empty:
            return df
        try:
            from textblob import TextBlob
            df["polarity"] = df["full_text"].apply(
                lambda t: TextBlob(t).sentiment.polarity if t else 0.0
            )
        except ImportError:
            logger.warning("textblob not installed — skipping polarity pre-scoring")
            df["polarity"] = 0.0
        return df


class NewsError(Exception):
    """Raised when NewsAPI calls fail."""
    pass
