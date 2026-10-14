"""One invocation's durable status and hashes; no scheduling or inference code."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class RuntimeProvenance:
    def __init__(self, output, action, config_path, *, command, torch_module):
        self.output, self.config_path = Path(output), Path(config_path)
        self.torch = torch_module
        self.started = time.monotonic()
        self.result = None
        versions = {}
        for name in ("numpy", "transformers", "peft", "safetensors"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.state = dict(schema_version="scale_interface_run/v1", status="running", phase="configuration",
            action=action, started_utc=now(), pid=os.getpid(), command=command, seed=None,
            environment=dict(python=sys.version, platform=platform.platform(),
                torch=str(torch_module.__version__), cuda=torch_module.version.cuda, packages=versions),
            gpu=dict(visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), initialized=False),
            inputs={}, input_roles={}, code_sha256={})

    def save(self):
        temporary = self.output / "status.tmp"
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.output / "status.json")

    def __enter__(self):
        self.save()
        return self

    def phase(self, phase):
        self.state["phase"] = phase
        self.save()

    def input(self, role, filename):
        # Retain lexical paths: cluster /home symlinks must not change input keys.
        path = Path(filename).absolute()
        name = str(path)
        self.state["input_roles"][role] = name
        if name not in self.state["inputs"]:
            self.state["hashing_input"] = name
            self.save()
            self.state["inputs"][name] = sha256(path)

    def configure(self, config):
        effective = deepcopy(config)
        effective.setdefault("seed", 42)
        effective.setdefault("device", "cuda")
        effective.setdefault("run", {})
        if self.state["action"] == "generate-scene":
            for key, value in dict(channel_intervention="original", max_new_tokens=32,
                                   maximum_length=2502, placement="question_tail", constant_cm=100).items():
                effective["run"].setdefault(key, value)
        with (self.output / "effective_config.json").open("x", encoding="utf-8") as stream:
            json.dump(effective, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self.state.update(seed=effective["seed"], effective_config_sha256=sha256(self.output / "effective_config.json"))
        self.state["gpu"]["requested_device"] = effective["device"]
        self.phase("hashing_inputs")
        self.input("configuration", self.config_path)
        package = Path(__file__).parent
        self.state["code_sha256"] = {path.relative_to(package).as_posix(): sha256(path)
            for path in sorted(package.rglob("*.py")) if "tests" not in path.relative_to(package).parts}
        for key in ("features", "radii", "attributes", "rows", "images", "relations", "full_manifest"):
            value = config.get("data", {}).get(key)
            if value is not None:
                if key == "rows" and isinstance(value, list):
                    self.state["inline_rows_sha256"] = hashlib.sha256(
                        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                else:
                    self.input("data." + key, value)
        host = config.get("host", {})
        for key in ("host_checkpoint", "deployment"):
            if host.get(key):
                self.input("host." + key, host[key])
        if (host.get("aqt") or {}).get("checkpoint"):
            self.input("host.aqt.checkpoint", host["aqt"]["checkpoint"])
        if config.get("run", {}).get("measurement_source"):
            self.input("run.measurement_source", config["run"]["measurement_source"])
        model_path = host.get("model", {}).get("llama_model_path")
        if model_path:
            directory = Path(model_path)
            if not directory.is_dir():
                raise ValueError("base LLM provenance requires a local model directory")
            weights = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin")))
            if not weights:
                raise ValueError("no local base LLM weights found for provenance")
            metadata = {p for pattern in ("*.json", "tokenizer*", "vocab*", "merges.txt", "*.model")
                        for p in directory.glob(pattern) if p.is_file()}
            for path in sorted(set(weights) | metadata):
                self.input("base_llm." + path.name, path)
        self.state.pop("hashing_input", None)
        self.save()

    def record_gpu(self):
        # Called after model loading, preserving seed/data/host initialization order.
        if self.torch.cuda.is_initialized():
            self.state["gpu"].update(initialized=True, devices=[dict(logical_index=index,
                name=self.torch.cuda.get_device_properties(index).name,
                total_memory_bytes=self.torch.cuda.get_device_properties(index).total_memory)
                for index in range(self.torch.cuda.device_count())])
        self.save()

    def __exit__(self, kind, error, tb):
        self.state.update(status="failed" if kind else "complete", finished_utc=now(),
                          elapsed_seconds=time.monotonic() - self.started)
        if error is not None:
            self.state["error"] = type(error).__name__ + ": " + str(error)
        try:
            self.state["artifacts"] = {path.name: sha256(path) for path in sorted(self.output.iterdir())
                if path.is_file() and path.name not in ("status.json", "status.tmp")}
            for name in ("predictions", "result"):
                filename = name + (".jsonl" if name == "predictions" else ".json")
                if filename in self.state["artifacts"]:
                    self.state[name + "_sha256"] = self.state["artifacts"][filename]
            if isinstance(self.result, dict) and "completed_predictions" in self.result:
                self.state["completed_predictions"] = self.result["completed_predictions"]
        except BaseException as seal_error:
            self.state.update(status="failed", seal_error=type(seal_error).__name__ + ": " + str(seal_error))
            self.save()
            if kind is None:
                raise
        self.save()
        return False
