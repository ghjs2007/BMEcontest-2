"""Crash-recovery contracts for the sole active event-stack release command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


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
        "backup_dist_path": None,
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
        "backup_dist_path": None,
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_recovery_with_tampered_replaced_dist_retains_journal_for_safe_repair(tmp_path: Path):
    """A corrupt active package must never be silently paired with old metadata."""

    from src.pipeline.artifacts import PromotionContractError
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, _tree_hash, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    journal = {
        "schema_version": 1,
        "phase": "dist_replaced",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": _tree_hash(root / "dist/event_stack"),
        "backup_dist_path": None,
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))
    (root / "dist/event_stack/requirements.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(PromotionContractError, match="does not match"):
        recover_release_transaction(root)

    assert _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_recovery_after_post_copy_rollback_keeps_old_pair(tmp_path: Path):
    """A failed package can leave a callback journal but must not advance registry."""

    from scripts.release_event_stack import _journal_path, _stable_json_bytes, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1,
        "phase": "dist_replaced",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": None,
        "backup_dist_path": None,
    }))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry
