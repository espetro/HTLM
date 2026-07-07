"""COOP/COEP HTTP server for the wllama browser benchmark harness."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pathlib
import re
import sys
import urllib.parse
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler


class COOPCOEPHandler(BaseHTTPRequestHandler):
    """Static file handler with cross-origin isolation headers + POST /results."""

    server_version = "HTLM-bench/1.0"

    def __init__(
        self,
        request,
        client_address,
        server,
        *,
        harness_file: pathlib.Path,
        models_dir: pathlib.Path,
        wllama_dir: pathlib.Path,
        prompts_file: pathlib.Path,
        results_file: pathlib.Path,
    ):
        self.harness_file = pathlib.Path(harness_file)
        self.models_dir = pathlib.Path(models_dir)
        self.wllama_dir = pathlib.Path(wllama_dir)
        self.prompts_file = pathlib.Path(prompts_file)
        self.results_file = pathlib.Path(results_file)
        super().__init__(request, client_address, server)

    def log_message(self, format: str, *args) -> None:
        print(f"[serve] {format % args}", file=sys.stderr)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _resolve(self, path: str) -> pathlib.Path | None:
        parsed = urllib.parse.urlparse(path)
        p = urllib.parse.unquote(parsed.path)
        if p == "/":
            return self.harness_file
        if p == "/prompts.json":
            return self.prompts_file
        if p.startswith("/models/"):
            rel = p[len("/models/") :].lstrip("/")
            if not rel or ".." in rel.split("/"):
                return None
            return self.models_dir / rel
        if p.startswith("/wllama/"):
            rel = p[len("/wllama/") :].lstrip("/")
            if not rel or ".." in rel.split("/"):
                return None
            return self.wllama_dir / rel
        return None

    def _content_type(self, path: pathlib.Path) -> str:
        if path.suffix == ".wasm":
            return "application/wasm"
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    def _parse_range(self, range_hdr: str, size: int) -> tuple[int, int] | None:
        # Single byte range only: bytes=start-end
        m = re.match(r"bytes=(\d+)-(\d*)", range_hdr.strip())
        if not m:
            return None
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        if start >= size or end < start:
            return None
        end = min(end, size - 1)
        return start, end

    def _send_file(self, path: pathlib.Path) -> None:
        path = path.resolve()
        try:
            size = path.stat().st_size
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        range_hdr = self.headers.get("Range")
        start, end = 0, size - 1
        status = HTTPStatus.OK
        content_length = size
        if range_hdr:
            rng = self._parse_range(range_hdr, size)
            if rng is None:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start, end = rng
            status = HTTPStatus.PARTIAL_CONTENT
            content_length = end - start + 1

        self.send_response(status)
        self.send_header("Content-Type", self._content_type(path))
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()

        if self.command == "HEAD":
            return

        with open(path, "rb") as f:
            if start:
                f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        path = self._resolve(self.path)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        self._send_file(path)

    do_HEAD = do_GET  # noqa: N815

    def do_POST(self) -> None:
        if self.path != "/results":
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Bad Content-Length")
            return
        data = self.rfile.read(length)
        try:
            json.loads(data)  # validate
        except json.JSONDecodeError as e:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {e}")
            return
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.results_file.write_bytes(data)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')


def _make_handler(
    *,
    harness_file: pathlib.Path,
    models_dir: pathlib.Path,
    wllama_dir: pathlib.Path,
    prompts_file: pathlib.Path,
    results_file: pathlib.Path,
):
    def _handler(request, client_address, server):
        return COOPCOEPHandler(
            request,
            client_address,
            server,
            harness_file=harness_file,
            models_dir=models_dir,
            wllama_dir=wllama_dir,
            prompts_file=prompts_file,
            results_file=results_file,
        )

    return _handler


def main() -> None:
    p = argparse.ArgumentParser(description="COOP/COEP server for the wllama harness")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--harness-file", required=True, type=pathlib.Path)
    p.add_argument("--models-dir", required=True, type=pathlib.Path)
    p.add_argument("--wllama-dir", required=True, type=pathlib.Path)
    p.add_argument("--prompts-file", required=True, type=pathlib.Path)
    p.add_argument("--results-file", required=True, type=pathlib.Path)
    args = p.parse_args()

    handler = _make_handler(
        harness_file=args.harness_file,
        models_dir=args.models_dir,
        wllama_dir=args.wllama_dir,
        prompts_file=args.prompts_file,
        results_file=args.results_file,
    )
    server = HTTPServer(("", args.port), handler)
    server.allow_reuse_address = True
    print(f"[serve] listening on http://localhost:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
