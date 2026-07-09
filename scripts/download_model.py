from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache"))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not str(cache_dir).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to download outside /root/autodl-tmp: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model,
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(path)


if __name__ == "__main__":
    main()

