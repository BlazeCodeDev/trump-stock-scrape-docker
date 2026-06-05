"""Background monitor: polls RSS/API, classifies with a LOCAL AI model, fires ntfy.

Classification uses any OpenAI-compatible /v1/chat/completions endpoint:
Ollama, LM Studio, llama.cpp server, vLLM, etc. No cloud API required.
"""

import feedparser
import requests
import json
import re
import time
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
            "id":        e.get("id") or e.get("link") or "",
            "text":      _strip_html(e.get("summary") or e.get("title") or ""),
            "published": e.get("published", ""),
            "link":      e.get("link", ""),
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
        posts.append({
            "id":        s["id"],
            "text":      text,
            "published": s.get("created_at", ""),
            "link":      s.get("url") or s.get("uri", ""),
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


def _classify(text: str, settings: dict) -> dict:
    url   = settings.get("model_url", "")
    name  = settings.get("model_name", "")
    key   = settings.get("model_key", "")

    payload = {
        "model": name,
        "messages": [
            {"role": "system", "content": _SYSTEM + "\n\n" + _JSON_INSTRUCTION},
            {"role": "user", "content": f"Post text:\n\n{text}"},
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

    resp = requests.post(url, json=payload, headers=headers, timeout=180)
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


def _send_ntfy(ntfy_url: str, post: dict, c: dict) -> None:
    direction = c.get("direction", "watch")
    assets    = ", ".join(c["affected_assets"]) if c["affected_assets"] else "Markets"
    title = f"Trump: {assets} ({direction.upper()})"
    body  = (
        f"{c['summary']}\n\n"
        f"Trading tip: {c['tip']}\n\n"
        f"Source: {post['link']}"
    )
    requests.post(
        ntfy_url,
        data=body.encode("utf-8"),
        headers={
            "Title":    title,
            "Priority": _PRIO.get(c.get("urgency", "low"), "default"),
        },
        timeout=10,
    ).raise_for_status()
    log.info("ntfy sent: %s", title)

# ── Main loop ─────────────────────────────────────────────────────────────────

def _run_once(settings: dict) -> None:
    if not settings.get("model_url") or not settings.get("model_name"):
        log.warning("No local model configured — open the web UI to choose one.")
        return

    # Nothing is downloaded automatically. If the bundled Ollama is reachable but
    # the selected model isn't installed yet, skip the cycle WITHOUT consuming
    # posts — they'll be classified once the user downloads a model in the UI.
    base = ollama_client.base_url(settings["model_url"])
    installed = ollama_client.list_installed(base)
    if installed is not None and settings["model_name"] not in installed:
        _status["ai_ok"] = False
        log.warning(
            "Model '%s' is not downloaded — open Settings → AI Model in the web UI "
            "to download it. Skipping this cycle.",
            settings["model_name"],
        )
        return

    posts = (
        _fetch_api(settings.get("ts_account_id", "107780257626128497"), settings["ts_token"])
        if settings.get("ts_token")
        else _fetch_rss(settings.get("feed_url", "https://www.trumpstruth.org/feed"))
    )
    log.info("Feed: %d posts total", len(posts))

    new_posts = [p for p in posts if p["id"] and not db.is_seen(p["id"])]
    log.info("New: %d posts to classify", len(new_posts))

    if not new_posts:
        return

    ntfy = settings.get("ntfy_url", "")

    for post in new_posts:
        if not post["text"]:
            db.save_post(post, None)
            continue
        try:
            log.info("Classifying: %s…", post["text"][:80])
            result = _classify(post["text"], settings)
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
