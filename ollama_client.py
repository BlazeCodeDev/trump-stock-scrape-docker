"""Talks to the bundled Ollama container: list/pull models, classify hardware fit.

Exposes a curated model catalogue. Each entry is rated for:
  • task quality  — how reliably it classifies posts + emits valid JSON
  • hardware fit  — computed against the detected RAM of this machine
"""

import json
import threading
import logging
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

# ── Curated catalogue ─────────────────────────────────────────────────────────
# size_gb : download size
# min_ram : RAM (GB) recommended to run it comfortably
# quality : suitability for THIS task (classification + JSON) — basic|good|excellent|best
# speed   : relative inference speed on CPU
# vision=True models can read images attached to posts. Text-only models ignore
# images (the post is classified on its text alone).
CATALOG: list[dict] = [
    # ── Text-only ──────────────────────────────────────────────────────────
    {"name": "llama3.2:1b",       "size_gb": 1.3, "min_ram": 4,  "quality": "basic",     "speed": "very fast", "vision": False},
    {"name": "qwen2.5:3b",        "size_gb": 1.9, "min_ram": 5,  "quality": "good",      "speed": "fast",      "vision": False},
    {"name": "llama3.2:3b",       "size_gb": 2.0, "min_ram": 5,  "quality": "good",      "speed": "fast",      "vision": False},
    {"name": "mistral:7b",        "size_gb": 4.1, "min_ram": 8,  "quality": "good",      "speed": "medium",    "vision": False},
    {"name": "qwen2.5:7b",        "size_gb": 4.7, "min_ram": 9,  "quality": "excellent", "speed": "medium",    "vision": False},
    {"name": "llama3.1:8b",       "size_gb": 4.9, "min_ram": 10, "quality": "excellent", "speed": "medium",    "vision": False},
    {"name": "gemma2:9b",         "size_gb": 5.4, "min_ram": 12, "quality": "excellent", "speed": "medium",    "vision": False},
    {"name": "qwen2.5:14b",       "size_gb": 9.0, "min_ram": 18, "quality": "best",      "speed": "slow",      "vision": False},
    # ── Vision (can read images in posts) ─────────────────────────────────
    {"name": "moondream",         "size_gb": 1.7, "min_ram": 4,  "quality": "basic",     "speed": "fast",      "vision": True},
    {"name": "llava:7b",          "size_gb": 4.7, "min_ram": 8,  "quality": "good",      "speed": "medium",    "vision": True},
    {"name": "llava-llama3:8b",   "size_gb": 5.5, "min_ram": 10, "quality": "good",      "speed": "medium",    "vision": True},
    {"name": "minicpm-v:8b",      "size_gb": 5.5, "min_ram": 10, "quality": "excellent", "speed": "medium",    "vision": True},
    {"name": "llama3.2-vision:11b","size_gb": 7.9, "min_ram": 12, "quality": "excellent", "speed": "slow",      "vision": True},
]


def is_vision(name: str) -> bool:
    """True if the named model can interpret images."""
    return any(m["name"] == name and m.get("vision") for m in CATALOG)

# ── Pull progress tracking (shared across threads) ────────────────────────────
_pull_state: dict[str, dict] = {}   # name -> {status, percent, done, error}
_pull_lock = threading.Lock()


def base_url(model_url: str) -> str:
    """Derive the Ollama base (scheme://host:port) from the chat-completions URL."""
    p = urlparse(model_url or "")
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}"
    return "http://ollama:11434"


def detect_ram_gb() -> float | None:
    """Best-effort total RAM in GB, honouring a cgroup memory limit if smaller."""
    ram = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            v = f.read().strip()
            if v.isdigit():
                lim = int(v) / 1024 / 1024 / 1024
                ram = min(ram, lim) if ram else lim
    except Exception:
        pass
    return round(ram, 1) if ram else None


def hardware_fit(min_ram: int, ram: float | None) -> str:
    """good | tight | heavy | unknown — how well this model fits the machine."""
    if ram is None:
        return "unknown"
    if ram >= min_ram + 2:
        return "good"
    if ram >= min_ram:
        return "tight"
    return "heavy"


def list_installed(base: str) -> list[str] | None:
    """Installed model names, or None if Ollama is unreachable."""
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def get_catalog(settings: dict) -> dict:
    base      = base_url(settings.get("model_url", ""))
    installed = list_installed(base)
    ram       = detect_ram_gb()
    inst      = set(installed or [])

    models = []
    for m in CATALOG:
        with _pull_lock:
            pull = _pull_state.get(m["name"])
        models.append({
            **m,
            "installed": m["name"] in inst,
            "fit":       hardware_fit(m["min_ram"], ram),
            "pull":      pull,
        })
    return {
        "reachable": installed is not None,
        "ram_gb":    ram,
        "selected":  settings.get("model_name", ""),
        "models":    models,
    }


def _do_pull(base: str, name: str) -> None:
    with _pull_lock:
        _pull_state[name] = {"status": "starting", "percent": 0, "done": False, "error": None}
    try:
        with requests.post(
            f"{base}/api/pull",
            json={"model": name, "stream": True},
            stream=True,
            timeout=(10, 300),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                status = d.get("status", "")
                total, done = d.get("total") or 0, d.get("completed") or 0
                with _pull_lock:
                    prev = _pull_state.get(name, {}).get("percent", 0)
                    pct = int(done * 100 / total) if total else prev
                    _pull_state[name] = {"status": status, "percent": pct, "done": False, "error": None}
                if status == "success":
                    break
        with _pull_lock:
            _pull_state[name] = {"status": "installed", "percent": 100, "done": True, "error": None}
        log.info("Model pulled: %s", name)
    except Exception as exc:
        with _pull_lock:
            _pull_state[name] = {"status": "error", "percent": 0, "done": True, "error": str(exc)}
        log.error("Pull failed for %s: %s", name, exc)


def start_pull(base: str, name: str) -> None:
    """Kick off a background pull unless one is already in flight for this model."""
    with _pull_lock:
        cur = _pull_state.get(name)
        if cur and not cur.get("done"):
            return
    threading.Thread(target=_do_pull, args=(base, name), daemon=True, name=f"pull-{name}").start()
