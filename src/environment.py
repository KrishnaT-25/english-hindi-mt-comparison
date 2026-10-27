"""Record hardware, package versions, and accessible model revisions."""
from __future__ import annotations

import importlib.metadata
import platform

import torch
from huggingface_hub import HfApi

from src.common import RESULTS, config, write_json


def main():
    cfg = config()
    packages = ["torch", "transformers", "datasets", "peft", "bitsandbytes",
                "sacrebleu", "sentencepiece", "accelerate"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        result["cuda_runtime"] = torch.version.cuda
    api = HfApi()
    result["models"] = {}
    for name in ("mt5", "llama", "gemma"):
        model_id = cfg[name]["model_id"]
        try:
            info = api.model_info(model_id)
            result["models"][name] = {"id": model_id, "revision": info.sha}
        except Exception as exc:
            result["models"][name] = {"id": model_id, "error": str(exc)}
    write_json(RESULTS / "environment.json", result)
    print(result)


if __name__ == "__main__":
    main()

