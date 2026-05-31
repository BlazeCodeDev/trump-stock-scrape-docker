"""Entry point: initialises DB, runs monitor + web server."""

import threading
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

import db
import monitor
from web import app


def main() -> None:
    db.init_db()

    threading.Thread(target=monitor.run_forever, daemon=True, name="monitor").start()

    port = int(os.getenv("PORT", "1122"))
    log.info("Web UI → http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
