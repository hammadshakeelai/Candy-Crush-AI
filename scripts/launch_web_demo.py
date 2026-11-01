#!/usr/bin/env python3
"""
AlphaCandy Interactive Web Playground Launcher
=============================================
Launches a zero-dependency local web server hosting the interactive
AlphaCandy Match-3 neural policy visualizer and player canvas.
"""

import os
import sys
import socket
import webbrowser
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"


def find_free_port(start_port=8080, max_port=8100):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[AlphaCandy Web] {self.address_string()} - {format % args}\n")


def main():
    if not WEB_DIR.exists() or not (WEB_DIR / "index.html").exists():
        print(f"Error: Web directory not found at {WEB_DIR}")
        sys.exit(1)

    port = find_free_port()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, CustomHandler)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("🍬 ALPHACANDY NEURAL PLAYGROUND & LIVE SOLVER DASHBOARD")
    print("=" * 65)
    print(f"🌐 Server active at: {url}")
    print("🎮 Open the link above in your browser to interact with:")
    print("   • Live 9x9 bit-accurate candy physics & gravity cascades")
    print("   • 79.3M AlphaCandy Titan policy evaluation & heatmaps")
    print("   • Dynamic CVaR risk mode tilting (Safe / Normal / Miracle)")
    print("   • Autonomous step & auto-play modes")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[AlphaCandy Web] Server shut down cleanly.")
        httpd.server_close()


if __name__ == "__main__":
    main()
