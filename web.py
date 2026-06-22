"""Flask web application — serves the UI and JSON API."""

from flask import Flask, jsonify, request, render_template
import db
import monitor
import ollama_client

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/posts")
def api_posts():
    limit         = min(int(request.args.get("limit", 50)), 200)
    relevant_only = request.args.get("relevant_only", "false").lower() == "true"
    return jsonify(db.get_posts(limit=limit, relevant_only=relevant_only))


@app.get("/api/settings")
def api_get_settings():
    s = db.get_all_settings()
    # Never expose secrets to the browser — replace with a boolean "is set" flag
    for key in db.SECRET_KEYS:
        s[f"{key}_set"] = bool(s.get(key))
        s[key] = ""
    return jsonify(s)


@app.post("/api/settings")
def api_save_settings():
    data = request.get_json(force=True) or {}
    safe_keys = ["feed_url", "model_url", "model_name", "check_interval", "max_post_age_min", "ntfy_url", "ts_account_id"]
    for key in safe_keys:
        if key in data:
            db.set_setting(key, str(data[key]))
    # Secrets: empty string = keep existing value
    for key in db.SECRET_KEYS:
        if data.get(key):
            db.set_setting(key, str(data[key]))
    return jsonify({"ok": True})


@app.get("/api/status")
def api_status():
    return jsonify(monitor.get_status())


@app.post("/api/ntfy/test")
def api_ntfy_test():
    """Send a test notification to the supplied URL (or the saved one)."""
    data = request.get_json(force=True) or {}
    url = (data.get("ntfy_url") or db.get_setting("ntfy_url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "No ntfy URL set"}), 400
    try:
        monitor.send_test(url)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.get("/api/models")
def api_models():
    """Catalogue with installed state, hardware fit, and live pull progress."""
    return jsonify(ollama_client.get_catalog(db.get_all_settings()))


@app.post("/api/models/pull")
def api_models_pull():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    base = ollama_client.base_url(db.get_setting("model_url"))
    ollama_client.start_pull(base, name)
    return jsonify({"ok": True})
