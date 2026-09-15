"""Crash-recovery contracts for the sole active event-stack release command."""

from __future__ import annotations

import json
import shutil
import hashlib
import subprocess
import sys
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


def _distinct_candidate(root: Path) -> dict[str, object]:
    """Create a separately attested run and copied package for interruption tests."""
    from src.pipeline.artifacts import _stable_json_bytes, incumbent_registry_payload

    old_key = "035644cf0889a5dd"
    new_key = "candidate-run-1"
    candidate_root = root / "models/event_stack" / new_key
    shutil.copytree(root / "models/event_stack" / old_key, candidate_root)
    manifest_path = candidate_root / "deployment/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git_sha"] = "candidate"
    manifest_path.write_bytes(_stable_json_bytes(manifest))
    summary_path = candidate_root / "promotion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment_key"] = new_key
    summary_path.write_bytes(_stable_json_bytes(summary))
    bundles = {}
    for name in [*(f"outer-fold-{fold}" for fold in range(5)), "deployment"]:
        entry = {"role": "deployment" if name == "deployment" else "outer-fold-evidence",
                 "manifest_sha256": hashlib.sha256((candidate_root / name / "manifest.json").read_bytes()).hexdigest()}
        if name != "deployment":
            entry["diagnostics_sha256"] = hashlib.sha256((candidate_root / name / "diagnostics.json").read_bytes()).hexdigest()
        bundles[name] = entry
    attestation = json.loads((candidate_root / "promotion_attestation.json").read_text(encoding="utf-8"))
    attestation["run_key"] = new_key
    attestation["aggregate_summary"]["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    attestation["bundles"] = bundles
    (candidate_root / "promotion_attestation.json").write_bytes(_stable_json_bytes(attestation))
    candidate_dist = root / "dist/event_stack"
    shutil.rmtree(candidate_dist)
    shutil.copytree(root / "models/event_stack" / new_key / "deployment", candidate_dist / "bundle")
    # Rebuild the runtime package by borrowing non-bundle runtime files from source dist.
    source_dist = Path("dist/event_stack")
    for name in ("predict_event_stack.py", "requirements.txt"):
        shutil.copy2(source_dist / name, candidate_dist / name)
    runtime = json.loads((source_dist / "runtime_manifest.json").read_text(encoding="utf-8"))
    runtime["bundle_manifest_sha256"] = hashlib.sha256((candidate_dist / "bundle/manifest.json").read_bytes()).hexdigest()
    runtime["files"]["bundle/manifest.json"] = runtime["bundle_manifest_sha256"]
    (candidate_dist / "runtime_manifest.json").write_text(json.dumps(runtime, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return incumbent_registry_payload(candidate_root)


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


def test_active_dist_candidate_match_reads_runtime_manifest(tmp_path: Path):
    from scripts.release_event_stack import _active_dist_matches_candidate

    root, registry = _release_fixture(tmp_path)
    assert _active_dist_matches_candidate(root, registry) is True


def test_new_process_recovers_package_swap_before_tree_hash_with_distinct_pair(tmp_path: Path):
    """The post-swap/pre-hash interruption must finish the new, not mixed, pair."""

    from scripts.release_event_stack import _journal_path, _stable_json_bytes

    root, old_registry = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)
    assert candidate["run_key"] != old_registry["run_key"]
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1,
        "phase": "prepared",
        "previous_registry": old_registry,
        "candidate_registry": candidate,
        "candidate_dist_sha256": None,
        "backup_dist_path": None,
    }))

    completed = subprocess.run(
        [sys.executable, "-c", f"from scripts.release_event_stack import recover_release_transaction; recover_release_transaction(r'{root}')"],
        cwd=Path.cwd(), text=True, capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == candidate


@pytest.mark.parametrize("phase", ["dist_backup", "dist_replaced", "registry_replaced", "verified"])
def test_new_process_recovers_each_later_candidate_phase(tmp_path: Path, phase: str):
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, _tree_hash

    root, old_registry = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1, "phase": phase, "previous_registry": old_registry,
        "candidate_registry": candidate,
        "candidate_dist_sha256": None if phase == "dist_backup" else _tree_hash(root / "dist/event_stack"),
        "backup_dist_path": None,
    }))

    completed = subprocess.run(
        [sys.executable, "-c", f"from scripts.release_event_stack import recover_release_transaction; recover_release_transaction(r'{root}')"],
        cwd=Path.cwd(), text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == candidate


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
