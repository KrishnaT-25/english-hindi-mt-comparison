"""Check that every pinned model's files are available to this account."""
from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

from src.common import RESULTS, write_json


def main():
    environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    result = {}
    for name, entry in environment["models"].items():
        try:
            hf_hub_download(entry["id"], "config.json", revision=entry["revision"],
                            force_download=True)
            result[name] = {"accessible": True, "revision": entry["revision"]}
        except Exception as exc:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            result[name] = {"accessible": False, "revision": entry["revision"],
                            "error": type(cause).__name__ + ": " + str(cause).splitlines()[0]}
    write_json(RESULTS / "model_access.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

