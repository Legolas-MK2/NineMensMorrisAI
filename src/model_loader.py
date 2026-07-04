"""
Shared model-discovery and model-loading helpers for the webui /
claude_plays / training_dashboard entry points.

Pulled out of webui.py (which had a duplicate-scanning bug — it walked
src/checkpoints twice) and claude_plays.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import Config
from model import ActorCritic


def discover_models(
    repo_root: Path,
    max_bytes: int = Config.max_model_file_bytes,
) -> List[Dict[str, str]]:
    """Scan the canonical model/checkpoint directories under `repo_root`.

    Skips any `.pt` file larger than `max_bytes` (likely incomplete or
    corrupted). Returns dicts of the form:
        {"name": <display name>, "path": <absolute path>, "type": "src"}
    """
    repo_root = Path(repo_root)
    candidates = [
        ("src/", repo_root / "src" / "models"),
        ("src/checkpoints/", repo_root / "src" / "checkpoints"),
        ("models/", repo_root / "models"),
        ("checkpoints/", repo_root / "checkpoints"),
    ]
    out: List[Dict[str, str]] = []
    seen: set = set()
    for prefix, d in candidates:
        if not d.exists():
            continue
        for pt in sorted(d.glob("*.pt"), reverse=True):
            try:
                size = pt.stat().st_size
            except OSError:
                continue
            if size >= max_bytes:
                continue
            key = str(pt.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": f"{prefix}{pt.name}", "path": str(pt), "type": "src"})
    return out


def load_actor_critic(
    path: str,
    obs_size: int,
    num_actions: int,
    device: torch.device,
    cache: Optional[Dict[str, Any]] = None,
) -> ActorCritic:
    """Load an ActorCritic model from a checkpoint file.

    If `cache` is provided, results are memoized by absolute path.
    Handles both raw `state_dict` files and full checkpoint dicts that
    wrap one under `model_state_dict`.
    """
    if cache is not None and path in cache:
        return cache[path]

    cfg = Config()
    model = ActorCritic(obs_size, num_actions, cfg).to(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    if cache is not None:
        cache[path] = model
    return model
