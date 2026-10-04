"""Local hand-labeling app for the Phase 1 detector validation.

Serves LABEL_DIR/manifest.json and clips, and appends labels to LABEL_DIR/labels.jsonl.
Detector outputs are never sent to the page.

Usage:
    python -m curation.label_server LABEL_DIR [--port 8765]
then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGES = {"full": Path(__file__).with_name("label_page.html"), "idle": Path(__file__).with_name("label_idle.html"),
         "completion": Path(__file__).with_name("label_completion.html")}


def make_handler(label_dir: Path, page: Path = PAGES["full"]):
    labels_path = label_dir / "labels.jsonl"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(label_dir), **kw)

        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/labels":
                latest = {}
                if labels_path.exists():
                    for line in labels_path.read_text().splitlines():
                        if line.strip():
                            row = json.loads(line)
                            latest[row["label_id"]] = row
                body = json.dumps(latest).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/labels":
                self.send_error(404)
                return
            row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            row["labeled_at"] = datetime.now(timezone.utc).isoformat()
            with labels_path.open("a") as f:  # append-only: last write per label_id wins
                f.write(json.dumps(row) + "\n")
            self.send_response(204)
            self.end_headers()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page", choices=sorted(PAGES), default="full")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.label_dir, PAGES[args.page]))
    print(f"Labeling app: http://localhost:{args.port}  (labels -> {args.label_dir / 'labels.jsonl'})")
    server.serve_forever()


if __name__ == "__main__":
    main()
