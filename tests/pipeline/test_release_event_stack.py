"""Crash-recovery contracts for the sole active event-stack release command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _release_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    shutil.copytree("models/event_stack/035644cf0889a5dd", root / "models/event_stack/035644cf0889a5dd")
    shutil.copytree("dist/event_stack", root / "dist/event_stack")
    registry = json.loads(Path("release/event_stack_incumbent.json").read_text(encoding="utf-8"))
    (root / "release").mkdir()
    (root / "release/event_stack_incumbent.json").write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return root, registry


def test_recovery_after_prepared_phase_retains_old_verified_release(tmp_path: Path):
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    journal = {
        "schema_version": 1,
        "phase": "prepared",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": None,
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_recovery_after_dist_phase_finishes_a_verified_new_pair(tmp_path: Path):
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, _tree_hash, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    journal = {
        "schema_version": 1,
        "phase": "dist_replaced",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": _tree_hash(root / "dist/event_stack"),
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry
