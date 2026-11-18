"""Run logging: git hash, config, seed, timestamps, package versions."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "configs" / "data.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "httpx",
        "datasets",
        "sklearn",
        "yaml",
        "tqdm",
    ]
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in names:
        mod_name = "sklearn" if name == "sklearn" else ("yaml" if name == "yaml" else name)
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", None)
            if ver is None and name == "yaml":
                ver = getattr(mod, "version", "unknown")
            out[name] = str(ver) if ver is not None else "unknown"
        except ImportError:
            out[name] = "not-installed"
    return out


def start_run_log(task: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    log = {
        "task": task,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "seed": cfg.get("seed"),
        "config": cfg,
        "package_versions": package_versions(),
        "events": [],
    }
    return log


def log_event(run_log: dict[str, Any], message: str, **kwargs: Any) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "message": message,
        **kwargs,
    }
    run_log.setdefault("events", []).append(entry)


def finish_run_log(run_log: dict[str, Any], out_path: Path | str) -> Path:
    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_log, indent=2, default=str))
    return path


def ensure_dirs() -> None:
    for rel in [
        "data/names",
        "data/templates",
        "data/corpus",
        "data/prompts",
        "data/mix",
        "data/assignments",
        "reports/run_logs",
        "RESULTS",
        "artifacts",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
