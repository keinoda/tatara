#!/usr/bin/env python3
"""Basic認証付きでmonitorの2ファイルだけを配信するHTTP server。"""

from __future__ import annotations

import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path


class MonitorHandler(BaseHTTPRequestHandler):
    root: Path
    expected_authorization: str

    def do_GET(self) -> None:  # noqa: N802
        authorization = self.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization, self.expected_authorization):
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Tatara monitor"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        route = self.path.split("?", 1)[0]
        names = {"/": ("index.html", "text/html; charset=utf-8"), "/status.json": ("status.json", "application/json")}
        if route not in names:
            self.send_error(404)
            return
        filename, content_type = names[route]
        path = self.root / filename
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(503, "monitor snapshot is not ready")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6001)
    args = parser.parse_args()
    user = os.environ.get("MONITOR_USER")
    password = os.environ.get("MONITOR_PASSWORD")
    if not user or not password:
        parser.error("MONITOR_USER and MONITOR_PASSWORD are required")
    token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    MonitorHandler.root = args.root.resolve()
    MonitorHandler.expected_authorization = f"Basic {token}"
    server = ThreadingHTTPServer((args.bind, args.port), MonitorHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
