"""Offline checks that resumes cannot change a measured system's configuration."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_manifest import ensure_manifest


def test_matching_resume_preserves_original_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    pins = {"judge_provider": "ollama_chat", "thinking_level": "low", "task_ids_sha256": "fixed"}
    fingerprint = ensure_manifest(path, pins)
    original = path.read_bytes()
    assert ensure_manifest(path, pins) == fingerprint
    assert path.read_bytes() == original


@pytest.mark.parametrize("pin", ["task_ids_sha256", "routing_sha256", "thinking_level", "user_sim_provider",
                                "max_user_turns", "gemini_backend", "image_id", "source_sha256"])
def test_changed_pin_is_rejected_without_overwriting_evidence(tmp_path, pin):
    path = tmp_path / "manifest.json"
    ensure_manifest(path, {pin: "original"})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        ensure_manifest(path, {pin: "changed"})
    assert path.read_bytes() == original


def test_legacy_manifest_requires_a_new_output_directory(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"system": "full"}))
    with pytest.raises(ValueError, match="new OUT"):
        ensure_manifest(path, {"system": "full"})
