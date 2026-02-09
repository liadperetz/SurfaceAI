"""Simple HTTP server for serving static websites.

Usage:
    python -m surfaceai.web.server [--port PORT]
    surfaceai serve [--port PORT]
"""

from __future__ import annotations

import http.server
import json
import socketserver
from pathlib import Path

WEBSITES_DIR = Path(__file__).parent / "websites"
DEFAULT_PORT = 8080


class WebsiteHandler(http.server.SimpleHTTPRequestHandler):
    """Handler that serves files from the websites directory with health check."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBSITES_DIR), **kwargs)

    def do_GET(self):
        """Handle GET requests, with special handling for health check."""
        if self.path == "/_health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(port: int = DEFAULT_PORT) -> None:
    """Start the HTTP server on the specified port.

    Args:
        port: Port to serve on (default: 8080)
    """
    if not WEBSITES_DIR.exists():
        raise FileNotFoundError(f"Websites directory not found: {WEBSITES_DIR}")

    with socketserver.TCPServer(("", port), WebsiteHandler) as httpd:
        print(f"Serving websites from: {WEBSITES_DIR}")
        print(f"Server running at http://localhost:{port}/")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve static websites for web agent testing")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    run_server(args.port)
