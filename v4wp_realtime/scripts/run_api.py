#!/usr/bin/env python3
"""V4_wP Signal API 서버 실행

Usage:
    python -m v4wp_realtime.scripts.run_api          # 기본 (0.0.0.0:8000)
    python -m v4wp_realtime.scripts.run_api --port 3000
    python v4wp_realtime/scripts/run_api.py
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시 모듈 import 보장)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="V4_wP Signal API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "v4wp_realtime.api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
