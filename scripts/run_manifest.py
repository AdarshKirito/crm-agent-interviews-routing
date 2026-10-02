"""Write immutable benchmark pins, or reject an incompatible resume before startup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root: Path) -> str:
    paths = subprocess.check_output([
        "git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
        "agent/app", "agent/pyproject.toml", "agent/uv.lock", "mcp-salesforce/src", "mcp-salesforce/package-lock.json",
        "search/src", "search/pyproject.toml", "search/uv.lock", "scripts", "patches", "Dockerfile", "deploy/start.sh",
    ]).decode().split("\0")
    hasher = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        path = root / name
        hasher.update(name.encode() + b"\0")
        hasher.update(path.read_bytes() if path.is_file() else b"<deleted>")
        hasher.update(b"\0")
    return hasher.hexdigest()


def ensure_manifest(path: Path, pins: dict) -> str:
    """A resume cannot silently merge results from different measured systems."""
    fingerprint = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != fingerprint:
            changed = sorted(key for key in pins if previous.get(key) != pins[key])
            raise ValueError(f"Run configuration changed ({', '.join(changed) or 'legacy manifest'}). "
                             "Preserve the existing run and select a new OUT directory.")
        return fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents concurrent streams with the same output pins
    # from silently replacing one another's manifest.
    payload = {**pins, "run_fingerprint": fingerprint,
               "started": datetime.now(timezone.utc).isoformat()}
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "system", "split", "task-ids", "eval-mode", "orgs", "modes", "big-model",
                 "small-model", "policy-model", "judge-model", "judge-provider", "user-model", "user-provider",
                 "thinking-level", "backend", "agent-runtime", "image", "image-id", "max-user-turns", "max-turns"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    task_ids = Path(args.task_ids)
    pins = {
        "schema_version": 2, "system": args.system, "split": args.split,
        "task_ids": str(task_ids.resolve()), "task_ids_sha256": digest(task_ids),
        "eval_mode": args.eval_mode, "orgs": args.orgs, "modes": args.modes,
        "big_model": args.big_model, "small_model": args.small_model, "policy_model": args.policy_model,
        "judge_model": args.judge_model, "judge_provider": args.judge_provider,
        "user_sim_model": args.user_model, "user_sim_provider": args.user_provider,
        "thinking_level": args.thinking_level, "gemini_backend": args.backend,
        "google_cloud_location": os.getenv("GOOGLE_CLOUD_LOCATION", ""),
        "agent_runtime": args.agent_runtime, "image": args.image, "image_id": args.image_id,
        "max_user_turns": int(args.max_user_turns), "max_turns": int(args.max_turns),
        "source_sha256": source_digest(root),
        "routing_sha256": digest(root / "agent/app/data/routing.yaml"),
        "harness_patch_sha256": digest(root / "patches/crmarena-crmroute.patch"),
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "fallback": False,
    }
    print(ensure_manifest(Path(args.output), pins))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
