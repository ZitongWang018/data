from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default=os.environ.get("MODELSCOPE_CACHE", "/root/autodl-tmp/modelscope_cache"))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not str(cache_dir).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to download outside /root/autodl-tmp: {cache_dir}")

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(args.model, cache_dir=str(cache_dir))
    print(path)


if __name__ == "__main__":
    main()

