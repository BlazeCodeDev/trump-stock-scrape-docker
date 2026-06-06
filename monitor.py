"""Background monitor: polls RSS/API, classifies with a LOCAL AI model, fires ntfy.

Classification uses any OpenAI-compatible /v1/chat/completions endpoint:
Ollama, LM Studio, llama.cpp server, vLLM, etc. No cloud API required.
"""

import feedparser
import requests
import json
import re
import time
import base64
import calendar
import logging
from datetime import datetime, timezone

import db
import ollama_client

log = logging.getLogger(__name__)

# Simple status dict — GIL makes string assignments safe across threads
_status: dict = {"last_run": "never", "running": False, "ai_ok": None}


def get_status() -> dict:
    return dict(_status)


# ── Classification prompt ─────────────────────────────────────────────────────

_SYSTEM = """\
You are a financial analyst specialising in how political statements move markets.
Analyse Trump's Truth Social posts and decide whether they have meaningful stock
market implications.

Stock-RELEVANT topics:
- Tariffs, trade policy, sanctions, import/export rules
- Tax cuts, deregulation, subsidies, trade deals
- Specific companies, CEOs, or industries named positively or negatively
- Energy policy: oil, gas, coal, renewables, pipelines
- Defence spending and defence contractors
- Technology regulation, AI policy, social-media law
- Federal Reserve comments, interest rates, dollar strength
- Infrastructure or government spending programmes
- Cryptocurrency / digital-asset policy
- Key trade partners: China, EU, Mexico, Canada, Japan

NOT relevant:
- Personal attacks on political opponents with no market angle
- Sports or entertainment commentary with no business link
- Pure social/cultural grievances
- Vague patriotism with no policy content

Posts often include attached images — screenshots of articles/headlines, charts,
photos, or memes. When image(s) are provided, read them and factor their content
into your analysis (e.g. a screenshot announcing tariffs, a chart of a stock, a
named company/product). The market signal may live entirely in the image.

Be concise and actionable. If uncertain, lean toward relevant with low urgency."""

_JSON_INSTRUCTION = """\
Respond with ONLY a single JSON object, no prose, no markdown fences. Exact keys:
{
  "relevant": true or false,
  "summary": "one sentence: what the post says and why it matters (or doesn't)",
  "affected_assets": ["ticker or asset class", ...]  (empty array if none, e.g. ["TSLA","Oil","DXY"]),
  "direction": "bullish" | "bearish" | "mixed" | "watch",
  "tip": "actionable 1-2 sentence trading tip with caveats",
  "urgency": "high" | "medium" | "low"
}"""

_DIRECTIONS = {"bullish", "bearish", "mixed", "watch"}
_URGENCIES = {"high", "medium", "low"}

# ── Feed fetching ─────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


_IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.I)


def _dedupe(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _rss_images(entry) -> list[str]:
    """Image URLs from an RSS entry: <img> tags in the HTML + media_content."""
    urls = _IMG_RE.findall(entry.get("summary") or "")
    for mc in (entry.get("media_content") or []):
        if mc.get("url"):
            urls.append(mc["url"])
    return _dedupe(urls)


def _api_images(status: dict) -> list[str]:
    """Image URLs from a Truth Social (Mastodon) status' media_attachments."""
    urls = []
    for src in (status, status.get("reblog") or {}):
        for m in (src.get("media_attachments") or []):
            if m.get("type") == "image":
                u = m.get("url") or m.get("preview_url")
                if u:
                    urls.append(u)
    return _dedupe(urls)


def _page_images(page_url: str) -> list[str]:
    """Scrape post media from a trumpstruth.org status page.

    The default RSS feed strips images, so when a vision model is active we fetch
    the post's page and pull the attached media (ignoring avatars/logos). Only
    called for new posts, so it's a handful of requests per cycle.
    """
    try:
        r = requests.get(page_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        raw = [
            u for u in _IMG_RE.findall(r.text)
            if ("media_attachments" in u or "/attachments/" in u) and "/avatars/" not in u
        ]
        # Dedupe by filename to collapse CDN/archive variants of the same image;
        # first occurrence wins (the full-resolution archive copy).
        seen, urls = set(), []
        for u in raw:
            key = u.rsplit("/", 1)[-1].split("?")[0]
            if key not in seen:
                seen.add(key)
                urls.append(u)
        return urls[:MAX_IMAGES]
    except Exception as exc:
        log.warning("Page image fetch failed (%s): %s", page_url, exc)
        return []


def _rss_ts(entry) -> float:
    """Epoch seconds from a feedparser entry's parsed time (UTC), or 0."""
    pp = entry.get("published_parsed") or entry.get("updated_parsed")
    if pp:
        try:
            return float(calendar.timegm(pp))
        except Exception:
            return 0.0
    return 0.0


def _fetch_rss(url: str) -> list[dict]:
    feedparser.USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
    )
    feed = feedparser.parse(url)
    if not feed.entries and feed.get("status") == 403:
        log.error(
            "RSS returned 403 (Cloudflare blocking this IP). "
            "Switch to trumpstruth.org feed in settings, or set a TS token."
        )
    return [
        {
            "id":           e.get("id") or e.get("link") or "",
            "text":         _strip_html(e.get("summary") or e.get("title") or ""),
            "published":    e.get("published", ""),
            "published_ts": _rss_ts(e),
            "link":         e.get("link", ""),
            "images":       _rss_images(e),
        }
        for e in feed.entries
    ]


def _fetch_api(account_id: str, token: str) -> list[dict]:
    resp = requests.get(
        f"https://truthsocial.com/api/v1/accounts/{account_id}/statuses",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "TrumpStockMonitor/2.0"},
        params={"limit": 20, "exclude_replies": "true"},
        timeout=15,
    )
    resp.raise_for_status()
    posts = []
    for s in resp.json():
        text = _strip_html(s.get("content", ""))
        if not text and s.get("reblog"):
            text = _strip_html(s["reblog"].get("content", ""))
        created = s.get("created_at", "")
        posts.append({
            "id":           s["id"],
            "text":         text,
            "published":    created,
            "published_ts": db.parse_published(created),
            "link":         s.get("url") or s.get("uri", ""),
            "images":       _api_images(s),
        })
    return posts

# ── Local AI classification ───────────────────────────────────────────────────

def _extract_json(content: str) -> dict:
    """Pull a JSON object out of a model response, tolerating fences / stray text."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start:end + 1]
    return json.loads(content)


def _normalise(data: dict) -> dict:
    direction = str(data.get("direction", "watch")).lower()
    urgency   = str(data.get("urgency", "low")).lower()
    assets    = data.get("affected_assets") or []
    if not isinstance(assets, list):
        assets = [str(assets)]
    return {
        "relevant":        bool(data.get("relevant", False)),
        "summary":         str(data.get("summary", "")),
        "affected_assets": [str(a) for a in assets][:8],
        "direction":       direction if direction in _DIRECTIONS else "watch",
        "tip":             str(data.get("tip", "")),
        "urgency":         urgency if urgency in _URGENCIES else "low",
    }


MAX_IMAGES = 3           # cap images sent per post
MAX_IMAGE_BYTES = 4_000_000


def _image_data_uri(url: str) -> str | None:
    """Download an image and return a base64 data URI, or None on failure.

    Downloading here (rather than passing the URL to the model) keeps the local
    model offline and avoids the Ollama container needing internet access.
    """
    try:
        r = requests.get(
            url, timeout=15, stream=True,
            headers={"User-Agent": "TrumpStockMonitor/2.0"},
        )
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type", "").split(";")[0]).strip().lower()
        if not ctype.startswith("image/"):
            return None
        chunks, total = [], 0
        for chunk in r.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                log.warning("Image too large, skipping: %s", url)
                return None
            chunks.append(chunk)
        b64 = base64.b64encode(b"".join(chunks)).decode()
        return f"data:{ctype};base64,{b64}"
    except Exception as exc:
        log.warning("Image fetch failed (%s): %s", url, exc)
        return None


def _classify(text: str, settings: dict, images: list[str] | None = None) -> dict:
    url   = settings.get("model_url", "")
    name  = settings.get("model_name", "")
    key   = settings.get("model_key", "")

    # Build the user message. For vision models with attached images, send a
    # multimodal content list (text + image_url blocks); otherwise plain text.
    text_part = f"Post text:\n\n{text}" if text else "This post has no text — analyse the attached image(s)."
    blocks = [{"type": "text", "text": text_part}]
    if images and ollama_client.is_vision(name):
        for img_url in images[:MAX_IMAGES]:
            uri = _image_data_uri(img_url)
            if uri:
                blocks.append({"type": "image_url", "image_url": {"url": uri}})
        if len(blocks) > 1:
            log.info("  attaching %d image(s) for vision analysis", len(blocks) - 1)

    user_content = blocks if len(blocks) > 1 else text_part

    payload = {
        "model": name,
        "messages": [
            {"role": "system", "content": _SYSTEM + "\n\n" + _JSON_INSTRUCTION},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "stream": False,
        # Most OpenAI-compatible servers (Ollama, LM Studio, vLLM) honour this.
        # Ignored gracefully by servers that don't; _extract_json still copes.
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    resp = requests.post(url, json=payload, headers=headers, timeout=240)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    return _normalise(_extract_json(content))

# ── ntfy notification ─────────────────────────────────────────────────────────

_PRIO = {"high": "urgent", "medium": "high", "low": "default"}


def send_test(ntfy_url: str) -> None:
    """Send a one-off test notification to verify the ntfy configuration."""
    requests.post(
        ntfy_url,
        data=b"Test notification from Trump Stock Monitor. "
             b"If you can read this, ntfy is configured correctly.",
        headers={"Title": "Test notification", "Priority": "default"},
        timeout=10,
    ).raise_for_status()


def _header_safe(s: str) -> str:
    """HTTP/ntfy headers must be latin-1 — drop anything that isn't."""
    return s.encode("latin-1", "ignore").decode("latin-1").strip() or "Trump market signal"


def _send_ntfy(ntfy_url: str, post: dict, c: dict) -> None:
    direction = (c.get("direction") or "watch").capitalize()
    urgency   = (c.get("urgency") or "low").capitalize()
    assets    = ", ".join(c["affected_assets"]) if c.get("affected_assets") else "Broad market"

    # Title: the scannable headline — what and which way.
    title = _header_safe(f"{assets}: {direction}")

    # Body: short, structured, easy to read at a glance. No raw URL clutter —
    # the post opens on tap (Click) and via the action button instead.
    meta = f"{direction} • {urgency} urgency • {assets}"
    parts = []
    if c.get("summary"):
        parts.append(c["summary"].strip())
    parts.append(meta)
    if c.get("tip"):
        parts.append(f"Tip: {c['tip'].strip()}")
    body = "\n\n".join(parts)

    headers = {
        "Title":    title,
        "Priority": _PRIO.get(c.get("urgency", "low"), "default"),
    }
    link = post.get("link")
    if link:
        headers["Click"]   = link                       # tap notification → open post
        headers["Actions"] = f"view, View post, {link}"  # explicit button

    requests.post(
        ntfy_url, data=body.encode("utf-8"), headers=headers, timeout=10
    ).raise_for_status()
    log.info("ntfy sent: %s", title)

# ── Main loop ─────────────────────────────────────────────────────────────────

def _run_once(settings: dict) -> None:
    posts = (
        _fetch_api(settings.get("ts_account_id", "107780257626128497"), settings["ts_token"])
        if settings.get("ts_token")
        else _fetch_rss(settings.get("feed_url", "https://www.trumpstruth.org/feed"))
    )
    log.info("Feed: %d posts total", len(posts))

    # First run: baseline the existing feed so we never classify or notify the
    # pre-existing backlog. Mark everything currently in the feed as seen (no
    # classification, no notifications). Only posts that appear AFTER this are
    # treated as new. This runs even before a model is downloaded.
    if db.count_posts() == 0:
        seeded = 0
        for p in posts:
            if p["id"]:
                db.save_post(p, None)  # seen-only; hidden from the UI, no ntfy
                seeded += 1
        log.info("First run: baselined %d existing posts; only newer posts from now on.", seeded)
        return

    # A post is genuinely "new" only if it was published AFTER the newest post we
    # already know about. This is the key guard: it prevents classifying old
    # posts even if dedup-by-id misses them. Anything older (or undatable) is
    # baselined (marked seen, hidden) and never classified.
    high_water = db.max_published_ts()
    new_posts = []
    for p in posts:
        if not p["id"] or db.is_seen(p["id"]):
            continue
        ts = p.get("published_ts", 0) or 0
        if ts and ts > high_water:
            new_posts.append(p)
        else:
            db.save_post(p, None)  # older than high-water or undatable → skip

    # Process oldest → newest so notifications arrive in posting order.
    new_posts.sort(key=lambda p: p.get("published_ts", 0))
    log.info("New: %d posts to classify (oldest first)", len(new_posts))

    if not new_posts:
        return

    # We have genuinely-new posts. Now we need a downloaded model to classify
    # them. If the selected model isn't installed yet, skip WITHOUT consuming the
    # new posts — they stay unseen and get classified once a model is downloaded.
    if not settings.get("model_url") or not settings.get("model_name"):
        log.warning("No local model configured — %d new post(s) waiting.", len(new_posts))
        return
    base = ollama_client.base_url(settings["model_url"])
    installed = ollama_client.list_installed(base)
    if installed is not None and settings["model_name"] not in installed:
        _status["ai_ok"] = False
        log.warning(
            "Model '%s' is not downloaded — %d new post(s) waiting. Download it in "
            "Settings → AI Model.",
            settings["model_name"], len(new_posts),
        )
        return

    # If a vision model is active, enrich new posts that have no images yet by
    # scraping the post page (the default RSS feed strips images). Only the new
    # posts, so it's cheap.
    if ollama_client.is_vision(settings["model_name"]):
        for p in new_posts:
            if not p.get("images") and p.get("link"):
                p["images"] = _page_images(p["link"])

    ntfy = settings.get("ntfy_url", "")

    for post in new_posts:
        has_text   = bool(post.get("text"))
        has_images = bool(post.get("images")) and ollama_client.is_vision(settings.get("model_name", ""))
        if not has_text and not has_images:
            # Nothing to analyse (no text, and either no images or a text-only model)
            db.save_post(post, None)
            continue
        try:
            log.info("Classifying: %s", (post["text"][:80] + "…") if has_text else "[image-only post]")
            result = _classify(post["text"], settings, images=post.get("images"))
            db.save_post(post, result)
            _status["ai_ok"] = True
            log.info(
                "  relevant=%-5s  dir=%-8s  urgency=%s",
                result["relevant"], result.get("direction"), result.get("urgency"),
            )
            if result["relevant"] and ntfy:
                _send_ntfy(ntfy, post, result)
        except requests.exceptions.ConnectionError:
            _status["ai_ok"] = False
            log.error(
                "Cannot reach local model at %s — is Ollama/LM Studio running? "
                "In Docker use host.docker.internal, not localhost.",
                settings.get("model_url"),
            )
            return  # stop this cycle; post stays unseen so we retry next time
        except Exception as exc:
            _status["ai_ok"] = False
            log.error("Error on post %s: %s", post["id"][:40], exc)
            db.save_post(post, None)


def run_forever() -> None:
    log.info("Monitor thread started")
    while True:
        settings = db.get_all_settings()
        interval = max(60, int(settings.get("check_interval") or 300))
        _status["running"] = True
        try:
            _run_once(settings)
        except Exception as exc:
            log.error("Monitor cycle error: %s", exc)
        finally:
            _status["running"] = False
            _status["last_run"] = datetime.now(timezone.utc).isoformat()
        log.info("Sleeping %ds until next check", interval)
        time.sleep(interval)
