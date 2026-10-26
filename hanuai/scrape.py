"""Best Buy Canada review scraper (Task 1 live demonstration).

Collection strategy
-------------------
1. Selenium drives a real Chrome session, renders the JavaScript product page
   and, where the review widget exposes them, selects each review-sort filter
   and actuates the "Show More" control.
2. BeautifulSoup parses the rendered HTML (product metadata and any review
   cards / JSON-LD review records that the page exposes).
3. Best Buy renders its Bazaarvoice review widget lazily and frequently omits
   it for automated browsers.  When the rendered page does not expose enough
   review cards, the scraper falls back to the product's public review JSON
   endpoint to page through reviews.  That endpoint is rate-limited politely
   and every request runs through the same exponential-backoff retry policy.

The fallback keeps the demonstration reliable without pretending that the
widget rendered when it did not; ``CollectionMethod`` records which path
supplied each row.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (
    DATA_DIR,
    DEMO_PRODUCT,
    MIN_REVIEWS_BRIEF,
    REVIEW_SORTS,
    TASK1_DEMO_SNAPSHOT,
    TASK1_DEMO_STATS,
)

LOGGER = logging.getLogger("hanuai.scrape")

CAPTCHA_MARKERS = (
    "access denied",
    "captcha",
    "verify you are human",
    "are you a robot",
    "unusual traffic",
    "request unsuccessful",
    "press & hold",
)

REVIEW_CARD_SELECTORS = (
    "[itemprop='reviewBody']",
    "[data-testid*='review-body']",
    "[class*='reviewBody']",
    "[class*='review-body']",
)

SHOW_MORE_LABELS = ("show more", "load more", "show more reviews", "view more reviews")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-CA,en;q=0.9",
    "Referer": "https://www.bestbuy.ca/",
}


@dataclass
class ScraperConfig:
    """Tunable scraper behaviour."""

    sku: str = DEMO_PRODUCT["sku"]
    product_name: str = DEMO_PRODUCT["name"]
    product_url: str = DEMO_PRODUCT["url"]
    target_reviews: int = max(MIN_REVIEWS_BRIEF + 10, 60)
    page_size: int = 50
    max_pages: int = 4
    sorts: tuple[str, ...] = REVIEW_SORTS
    headless: bool = False
    max_retries: int = 4
    backoff_base: float = 1.5
    page_timeout: float = 45.0
    captcha_pause_seconds: float = 5.0
    max_show_more_clicks: int = 5
    request_timeout: float = 30.0
    request_delay: float = 1.0


@dataclass
class ScrapeStats:
    """Observable results of one scraping run."""

    sorts_attempted: list[str] = field(default_factory=list)
    sorts_applied: list[str] = field(default_factory=list)
    rendered_reviews: int = 0
    api_reviews: int = 0
    pages_fetched: int = 0
    show_more_clicks: int = 0
    captcha_events: int = 0
    retries: int = 0
    notes: list[str] = field(default_factory=list)
    browser_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sorts_attempted": list(self.sorts_attempted),
            "sorts_applied": list(self.sorts_applied),
            "rendered_reviews": self.rendered_reviews,
            "api_reviews": self.api_reviews,
            "pages_fetched": self.pages_fetched,
            "show_more_clicks": self.show_more_clicks,
            "captcha_events": self.captcha_events,
            "retries": self.retries,
            "browser_used": self.browser_used,
            "notes": list(self.notes),
        }


def _normalise_date(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text[:20], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text[:10]


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


class BestBuyScraper:
    """Scrape Best Buy Canada reviews with rendering, retries and fallbacks."""

    def __init__(self, config: ScraperConfig | None = None, logger: logging.Logger | None = None):
        self.config = config or ScraperConfig()
        self.log = logger or LOGGER
        self.stats = ScrapeStats()
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    # ------------------------------------------------------------------ #
    # Browser lifecycle and retry helpers
    # ------------------------------------------------------------------ #
    def _init_driver(self) -> bool:
        if self.driver is not None:
            return True
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            if self.config.headless:
                options.add_argument("--headless=new")
            options.add_argument("--window-size=1400,1200")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--lang=en-CA")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.config.page_timeout)
            # Stealth: hide the webdriver flag from trivial bot checks.
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": (
                        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                        "window.chrome = window.chrome || {runtime: {}};"
                    )
                },
            )
            self.stats.browser_used = True
            return True
        except Exception as exc:  # pragma: no cover - depends on host browser
            self.log.warning("Browser could not be started (%s); using HTTP fallback.", exc)
            self.stats.notes.append(f"browser unavailable: {type(exc).__name__}: {exc}")
            self.driver = None
            return False

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self.driver = None

    def _retry(self, operation: Callable[[], Any], description: str) -> Any:
        """Run ``operation`` with exponential backoff on transient failures."""
        delay = self.config.backoff_base
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                self.stats.retries += 1
                self.log.warning(
                    "%s failed (attempt %s/%s): %s",
                    description,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt == self.config.max_retries:
                    break
                time.sleep(delay)
                delay *= 2
        assert last_error is not None
        raise last_error

    def _check_for_captcha(self, html: str) -> bool:
        # Inspect title and visible copy only: reCAPTCHA scripts are loaded on
        # every normal page and must not be mistaken for a challenge.
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
        visible = soup.get_text(" ", strip=True).lower()
        haystack = f"{title} {visible[:8000]}"
        if any(marker in haystack for marker in CAPTCHA_MARKERS):
            self.stats.captcha_events += 1
            self.log.warning(
                "CAPTCHA / bot challenge detected; pausing %.1fs before continuing.",
                self.config.captcha_pause_seconds,
            )
            time.sleep(self.config.captcha_pause_seconds)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Rendering and parsing
    # ------------------------------------------------------------------ #
    def render(self, url: str) -> str:
        """Load a URL in Chrome and return the rendered HTML."""
        if not self._init_driver():
            raise RuntimeError("No browser available for rendering")

        def _load() -> str:
            self.driver.get(url)
            time.sleep(3)
            # Trigger lazy sections (reviews load below the fold).
            for offset in (600, 1200, 1800, 2400):
                self.driver.execute_script(f"window.scrollTo(0, {offset});")
                time.sleep(0.4)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            html = self.driver.page_source
            if self._check_for_captcha(html):
                raise RuntimeError("CAPTCHA/bot challenge encountered")
            return html

        return self._retry(_load, "render product page")

    def _click_show_more(self) -> int:
        """Click every visible "Show More"-style review control."""
        if self.driver is None:
            return 0
        clicks = 0
        for _ in range(self.config.max_show_more_clicks):
            target = None
            for element in self.driver.find_elements("xpath", "//button | //a"):
                try:
                    text = (element.text or "").strip().lower()
                except Exception:
                    continue
                if text and any(text == label or text.startswith(label) for label in SHOW_MORE_LABELS):
                    target = element
                    break
            if target is None:
                break
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", target)
                time.sleep(0.5)
                self.driver.execute_script("arguments[0].click();", target)
                clicks += 1
                time.sleep(1.5)
            except Exception as exc:
                self.log.info("Show More control disappeared after click: %s", exc)
                break
        if clicks:
            self.stats.show_more_clicks += clicks
        else:
            self.log.info("No Show More control present in the rendered page.")
            self.stats.notes.append("show-more control absent")
        return clicks

    def _select_sort(self, label: str) -> bool:
        """Select a review-sort filter in the rendered page if it exists."""
        if self.driver is None:
            return False
        try:
            direct = self.driver.find_elements(
                "xpath", f"//*[self::button or self::option or self::a][contains(normalize-space(.), '{label}')]"
            )
            if direct:
                element = direct[0]
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
                time.sleep(0.4)
                self.driver.execute_script("arguments[0].click();", element)
                time.sleep(2)
                return True

            # Some builds expose the filters inside a native <select>.
            for select in self.driver.find_elements("tag name", "select"):
                for option in select.find_elements("tag name", "option"):
                    if label.lower() in (option.text or "").lower():
                        option.click()
                        time.sleep(2)
                        return True
        except Exception as exc:
            self.log.info("Sort filter %r unavailable: %s", label, exc)
        self.stats.notes.append(f"sort control absent: {label}")
        return False

    @staticmethod
    def parse_rendered_html(html: str, sort_filter: str = "") -> list[dict[str, Any]]:
        """Parse review records out of rendered HTML with BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        records: list[dict[str, Any]] = []

        # 1) Review cards exposed by the review widget.
        for node in soup.select(", ".join(REVIEW_CARD_SELECTORS)):
            body = _clean(node.get_text(" ", strip=True))
            if len(body) < 3:
                continue
            container = node.find_parent(["article", "li", "div"]) or node
            title_node = container.select_one("[itemprop='name'], h3, h4, [class*='title']")
            author_node = container.select_one("[itemprop='author'], [class*='author'], [class*='reviewer']")
            rating_node = container.select_one("[itemprop='ratingValue'], [aria-label*='star']")
            date_node = container.select_one("time, [itemprop='datePublished'], [class*='date']")
            rating = 0
            if rating_node is not None:
                match = re.search(r"([0-5](?:\.\d)?)", rating_node.get("content", "") or rating_node.get_text(" "))
                if match:
                    rating = int(round(float(match.group(1))))
            digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
            records.append(
                {
                    "ReviewID": f"rendered-{digest}",
                    "ReviewTitle": _clean(title_node.get_text(" ", strip=True)) if title_node else "",
                    "ReviewText": body,
                    "ReviewDate": _normalise_date(date_node.get_text(" ", strip=True) if date_node else ""),
                    "Rating": rating,
                    "ReviewerName": _clean(author_node.get_text(" ", strip=True)) if author_node else "Anonymous",
                    "SortFilter": sort_filter,
                    "CollectionMethod": "selenium+bs4",
                }
            )

        # 2) JSON-LD review records (Best Buy publishes a few on every PDP).
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                payload = json.loads(script.string or "{}")
            except (TypeError, ValueError):
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                product = item.get("itemReviewed") if isinstance(item, dict) else None
                reviews = []
                if isinstance(item, dict) and isinstance(item.get("review"), list):
                    reviews = item["review"]
                elif isinstance(product, dict) and isinstance(product.get("review"), list):
                    reviews = product["review"]
                for review in reviews:
                    body = _clean(review.get("reviewBody") or review.get("description"))
                    if not body:
                        continue
                    author = review.get("author") or {}
                    author_name = author.get("name") if isinstance(author, dict) else str(author or "")
                    rating = review.get("reviewRating") or {}
                    rating_value = rating.get("ratingValue") if isinstance(rating, dict) else rating
                    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
                    records.append(
                        {
                            "ReviewID": f"jsonld-{digest}",
                            "ReviewTitle": _clean(review.get("name")),
                            "ReviewText": body,
                            "ReviewDate": _normalise_date(review.get("datePublished")),
                            "Rating": int(round(float(rating_value))) if rating_value else 0,
                            "ReviewerName": _clean(author_name) or "Anonymous",
                            "SortFilter": sort_filter,
                            "CollectionMethod": "selenium+bs4 (json-ld)",
                        }
                    )
        return records

    # ------------------------------------------------------------------ #
    # Review API pagination fallback
    # ------------------------------------------------------------------ #
    def _fetch_api_page(self, page: int) -> dict[str, Any]:
        params = {"page": page, "pageSize": self.config.page_size, "lang": "en-CA"}
        url = f"https://www.bestbuy.ca/api/v2/json/reviews/{self.config.sku}"

        def _get() -> dict[str, Any]:
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            if response.status_code in (403, 429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code} for {response.url}")
            response.raise_for_status()
            return response.json()

        payload = self._retry(_get, f"review API page {page}")
        self.stats.pages_fetched += 1
        return payload

    def _api_records(self, sort_filter: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for page in range(1, self.config.max_pages + 1):
            payload = self._fetch_api_page(page)
            reviews = payload.get("reviews") or []
            for review in reviews:
                review_id = str(review.get("id") or "")
                if not review_id:
                    continue
                records.append(
                    {
                        "ReviewID": review_id,
                        "ReviewTitle": _clean(review.get("title")),
                        "ReviewText": _clean(review.get("comment")),
                        "ReviewDate": _normalise_date(review.get("submissionTime")),
                        "Rating": int(review.get("rating") or 0),
                        "ReviewerName": _clean(review.get("reviewerName")) or "Anonymous",
                        "VerifiedPurchaser": bool(review.get("isVerifiedPurchaser")),
                        "Recommended": review.get("isRecommended"),
                        "SortFilter": sort_filter,
                        "CollectionMethod": "http review endpoint",
                    }
                )
            if len(records) >= self.config.target_reviews:
                break
            if not reviews or page >= int(payload.get("totalPages") or page):
                break
            time.sleep(self.config.request_delay)
        self.stats.api_reviews += len(records)
        return records

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def scrape(self) -> pd.DataFrame:
        """Run the full demo scrape and return a de-duplicated review frame."""
        collected: dict[str, dict[str, Any]] = {}

        def merge(records: list[dict[str, Any]]) -> int:
            new = 0
            for record in records:
                key = record["ReviewID"]
                if key not in collected:
                    collected[key] = record
                    new += 1
            return new

        rendered_html = ""
        try:
            rendered_html = self.render(self.config.product_url)
            self.stats.notes.append("product page rendered with Selenium")
        except Exception as exc:
            self.log.warning("Rendered page unavailable: %s", exc)
            self.stats.notes.append(f"render failed: {exc}")

        if rendered_html:
            self._click_show_more()

        for sort_filter in self.config.sorts:
            self.stats.sorts_attempted.append(sort_filter)
            if self.driver is not None:
                if self._select_sort(sort_filter):
                    self.stats.sorts_applied.append(sort_filter)
                    self._click_show_more()
                    try:
                        html = self.driver.page_source
                        if self._check_for_captcha(html):
                            self.stats.notes.append(f"captcha during {sort_filter}")
                            continue
                        records = self.parse_rendered_html(html, sort_filter)
                        self.stats.rendered_reviews += len(records)
                        merge(records)
                    except Exception as exc:
                        self.log.warning("Could not parse rendered page for %s: %s", sort_filter, exc)

            # API fallback for volume/pagination; stop once the target is met.
            if len(collected) < self.config.target_reviews:
                try:
                    merge(self._api_records(sort_filter))
                except Exception as exc:
                    self.log.warning("Review endpoint failed for %s: %s", sort_filter, exc)
                    self.stats.notes.append(f"api failed: {sort_filter}: {exc}")

        frame = pd.DataFrame.from_records(list(collected.values()))
        if frame.empty:
            raise RuntimeError("No reviews could be collected from Best Buy Canada.")

        frame["PrimaryKey"] = [f"{self.config.sku}-{rid}" for rid in frame["ReviewID"]]
        frame["ProductID"] = self.config.sku
        frame["ProductName"] = self.config.product_name
        frame["SourcePlatform"] = "Best Buy Canada"
        frame["Dataset"] = "Demo (live scrape)"
        frame = frame.drop_duplicates(subset=["ReviewID"]).reset_index(drop=True)

        # Keep newest reviews first for a stable, readable export.
        frame = frame.sort_values(["ReviewDate", "ReviewID"], ascending=[False, False]).reset_index(drop=True)
        frame = frame[
            [
                "PrimaryKey",
                "ReviewID",
                "ProductID",
                "ProductName",
                "ReviewTitle",
                "ReviewText",
                "ReviewDate",
                "Rating",
                "SourcePlatform",
                "ReviewerName",
                "VerifiedPurchaser",
                "Recommended",
                "SortFilter",
                "CollectionMethod",
                "Dataset",
            ]
        ]
        return frame


def scrape_live_demo(config: ScraperConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scrape the live demo product and persist a snapshot of the result."""
    scraper = BestBuyScraper(config)
    try:
        frame = scraper.scrape()
    finally:
        scraper.close()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TASK1_DEMO_SNAPSHOT, index=False, encoding="utf-8")
    stats = scraper.stats.as_dict()
    stats["collected"] = int(len(frame))
    stats["saved_to"] = str(TASK1_DEMO_SNAPSHOT)
    stats["source"] = "live scrape"
    TASK1_DEMO_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return frame, stats


def load_demo_snapshot() -> pd.DataFrame | None:
    """Load the cached live-scrape snapshot if it exists."""
    if TASK1_DEMO_SNAPSHOT.exists():
        return pd.read_csv(TASK1_DEMO_SNAPSHOT)
    return None


def get_demo_reviews(live: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return demo reviews, scraping live or reusing the cached snapshot."""
    if live:
        return scrape_live_demo()

    snapshot = load_demo_snapshot()
    if snapshot is not None:
        if TASK1_DEMO_STATS.exists():
            stats = json.loads(TASK1_DEMO_STATS.read_text(encoding="utf-8"))
        else:
            stats = {"collected": int(len(snapshot))}
        stats["source"] = "cached live snapshot"
        stats["saved_to"] = str(TASK1_DEMO_SNAPSHOT)
        return snapshot, stats

    try:
        frame, stats = scrape_live_demo()
        stats["source"] = "live scrape"
        return frame, stats
    except Exception as exc:
        raise RuntimeError(
            "No cached live demo snapshot exists and a live scrape failed. "
            "Run `python run_pipeline.py --live` with a working Chrome installation."
        ) from exc
