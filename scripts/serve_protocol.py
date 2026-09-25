#!/usr/bin/env python3
"""Serve the optional FastAPI protocol surface (read-only).

Requires: pip install 'glue-lp[api]'  (fastapi, uvicorn)
Core tests do NOT require this.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="GLUE-LP optional protocol API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn not installed. pip install 'glue-lp[api]' or: pip install fastapi uvicorn"
        ) from exc
    uvicorn.run("glue_lp.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
