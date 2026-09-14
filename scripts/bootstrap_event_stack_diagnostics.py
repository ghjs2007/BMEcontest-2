"""One-time, guarded diagnostics bootstrap for the frozen incumbent run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as project_config
from src.pipeline.artifacts import (
    EventStackBundle,
    PromotionContractError,
    _write_promotion_attestation,
    load_event_stack_bundle,
    verify_promotion_attestation,
)
from src.pipeline.diagnostics import (
    aggregate_subject_diagnostics,
    canonical_diagnostics_bytes,
)
from src.pipeline.runner import (
    FilesystemDataSource,
    FoldResult,
    RunConfig,
    cache_key,
    expected_feature_dimensions,
    run_outer_fold,
)
from scripts.promote_event_stack import _config_from_summary


INCUMBENT_RUN_KEY = "035644cf0889a5dd"
INCUMBENT_METRICS = {"n_tp": 109, "n_true": 153, "n_pred": 237, "f1": 0.558974358974359}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_verified_incumbent(
    run_root: Path, output_directory: Path, run_key: str
) -> tuple[dict[str, object], tuple[RunConfig, ...], tuple[dict[str, object], ...]]:
    """Verify the predecessor's summary/manifest binding before any replay."""

    try:
        summary_path = run_root / "promotion_summary.json"
        attestation = json.loads((run_root / "promotion_attestation.json").read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionContractError("incumbent promotion evidence cannot be read") from exc
    if not isinstance(attestation, dict) or not isinstance(summary, dict):
        raise PromotionContractError("incumbent promotion evidence must be JSON objects")
    aggregate = attestation.get("aggregate_summary")
    if (
        attestation.get("run_key") != run_key
        or summary.get("experiment_key") != run_key
        or not isinstance(aggregate, Mapping)
        or aggregate.get("sha256") != _sha256(summary_path)
    ):
        raise PromotionContractError("incumbent promotion summary attestation is invalid")
    entries = attestation.get("bundles")
    if not isinstance(entries, Mapping):
        raise PromotionContractError("incumbent promotion bundles are not attested")
    for fold in range(5):
        key = f"outer-fold-{fold}"
        entry = entries.get(key)
        manifest = run_root / key / "manifest.json"
        if not isinstance(entry, Mapping) or entry.get("manifest_sha256") != _sha256(manifest):
            raise PromotionContractError("incumbent outer-fold manifest attestation is invalid")
    raw_configs = summary.get("run_configs")
    folds = summary.get("folds")
    if not isinstance(raw_configs, list) or not isinstance(folds, list) or len(raw_configs) != 5 or len(folds) != 5:
        raise PromotionContractError("incumbent summary lacks five frozen folds")
    configs = tuple(sorted((_config_from_summary(item) for item in raw_configs if isinstance(item, Mapping)), key=lambda item: item.outer_fold))
    if tuple(item.outer_fold for item in configs) != (0, 1, 2, 3, 4):
        raise PromotionContractError("incumbent configurations are not folds zero through four")
    records: list[dict[str, object]] = []
    by_fold = {config.outer_fold: config for config in configs}
    for item in sorted(folds, key=lambda value: int(value["outer_fold"])):
        if not isinstance(item, Mapping) or not isinstance(item.get("config_hash"), str):
            raise PromotionContractError("incumbent fold record is malformed")
        fold = item.get("outer_fold")
        if fold not in by_fold:
            raise PromotionContractError("incumbent fold does not match a frozen config")
        path = output_directory / "crossfit" / f"fold{fold}_{item['config_hash']}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromotionContractError("incumbent cached fold evidence cannot be read") from exc
        if not isinstance(record, dict) or record.get("config_hash") != item["config_hash"]:
            raise PromotionContractError("incumbent cached fold evidence does not match summary")
        expected_config = asdict(by_fold[fold])
        # The frozen incumbent predates the explicit Context-v1 null field.
        # Missing is accepted only for a field whose registered value is null.
        if isinstance(record.get("run_config"), Mapping):
            expected_config = {
                key: value for key, value in expected_config.items()
                if not (value is None and key not in record["run_config"])
            }
        if json.dumps(record.get("run_config"), sort_keys=True) != json.dumps(expected_config, sort_keys=True):
            raise PromotionContractError("incumbent cached fold configuration changed")
        records.append(record)
    return summary, configs, tuple(records)


def _assert_incumbent_metrics(results: Sequence[FoldResult]) -> None:
    totals = {
        "n_tp": sum(item.outer_metrics.n_tp for item in results),
        "n_true": sum(item.outer_metrics.n_true for item in results),
        "n_pred": sum(item.outer_metrics.n_pred for item in results),
    }
    totals["f1"] = 2 * totals["n_tp"] / (totals["n_pred"] + totals["n_true"])
    if totals != INCUMBENT_METRICS:
        raise PromotionContractError(
            "incumbent replay metrics differ from frozen baseline: " + repr(totals)
        )


def _validate_incumbent_input_provenance(
    configs: Sequence[RunConfig], records: Sequence[Mapping[str, object]], source: object,
) -> None:
    """Reproduce the incumbent's v5 cache identity, including its absent null field."""

    from src.pipeline import runner

    input_files = getattr(source, "input_files", None)
    if not callable(input_files):
        raise PromotionContractError("registered promotion source cannot enumerate input files")
    models = {
        "window": runner.WINDOW_MODEL_PARAMETERS,
        "micro_window": runner.MICRO_WINDOW_MODEL_PARAMETERS,
        "verifier": runner.VERIFIER_MODEL_PARAMETERS,
        "verifier_lgbm": runner.VERIFIER_LGBM_PARAMETERS,
    }
    for config, record in zip(configs, records):
        frozen_config = asdict(config)
        frozen_config.pop("context_features_version", None)
        try:
            payload = {
                "schema_version": runner.RUNNER_SCHEMA_VERSION,
                "config": frozen_config,
                "feature_dimensions": list(expected_feature_dimensions(config)),
                "model_parameters": models,
                "input_files": [runner._file_signature(Path(path)) for path in input_files(config)],
            }
            current_hash = hashlib.sha256(
                json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
        except (OSError, TypeError, ValueError) as exc:
            raise PromotionContractError("registered promotion inputs cannot be fingerprinted") from exc
        if record.get("config_hash") != current_hash:
            raise PromotionContractError("registered fold evidence does not bind the current files and feature schema")


def bootstrap_incumbent_diagnostics(
    run_key: str,
    *,
    models_root: Path = project_config.MODEL_DIR,
    output_directory: Path = project_config.OUTPUT_DIR,
    source_factory: Callable[[Path], object] = FilesystemDataSource,
    replay_fold: Callable[[RunConfig, object], FoldResult] | None = None,
) -> tuple[Path, ...]:
    """Replay and install diagnostics only after every immutable guard passes."""

    if run_key != INCUMBENT_RUN_KEY:
        raise PromotionContractError("diagnostic bootstrap is permanently restricted to the incumbent run key")
    root = Path(models_root) / "event_stack" / run_key
    installed_diagnostics = root / "diagnostics"
    if installed_diagnostics.is_dir():
        problems = verify_promotion_attestation(root, expected_run_key=run_key)
        paths = tuple(sorted(installed_diagnostics.glob("fold*.diagnostics.json")))
        if not problems and len(paths) == 5:
            return paths
        raise PromotionContractError(
            "an incomplete diagnostics bootstrap already exists; refusing to overwrite immutable evidence"
        )
    summary, configs, records = _read_verified_incumbent(root, Path(output_directory), run_key)
    source = source_factory(project_config.ROOT_DIR)
    _validate_incumbent_input_provenance(configs, records, source)
    replay = replay_fold or (
        lambda config, current_source: run_outer_fold(
            config, data_source=current_source.load_outer_fold(config)
        )
    )
    results = tuple(replay(config, source) for config in configs)
    _assert_incumbent_metrics(results)
    diagnostics = tuple(result.subject_diagnostics for result in results)
    if any(not item for item in diagnostics):
        raise PromotionContractError("incumbent replay did not produce subject diagnostics")

    # No directory is touched above this point: all provenance and replay gates
    # have already passed.
    diagnostic_directory = root / "diagnostics"
    diagnostic_directory.mkdir(parents=False, exist_ok=False)
    try:
        written: list[Path] = []
        bundles: dict[str, EventStackBundle] = {}
        for config, record, diagnostic in zip(configs, records, diagnostics):
            filename = f"fold{config.outer_fold}_{record['config_hash']}.diagnostics.json"
            path = diagnostic_directory / filename
            payload = dict(diagnostic)
            path.write_bytes(canonical_diagnostics_bytes(payload))
            written.append(path)
            bundle_path = root / f"outer-fold-{config.outer_fold}"
            bundle_diagnostic = bundle_path / "diagnostics.json"
            bundle_diagnostic.write_bytes(canonical_diagnostics_bytes(payload))
            manifest_path = bundle_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["diagnostics.json"] = _sha256(bundle_diagnostic)
            manifest_path.write_bytes(_stable_json_bytes(manifest))
            bundles[f"outer-fold-{config.outer_fold}"] = load_event_stack_bundle(
                bundle_path, expected_role="outer-fold-evidence"
            )
        deployment = root / "deployment"
        bundles["deployment"] = load_event_stack_bundle(deployment, expected_role="deployment")
        summary["subject_diagnostics"] = aggregate_subject_diagnostics(diagnostics)
        summary_path = root / "promotion_summary.json"
        summary_path.write_bytes(_stable_json_bytes(summary))
        _write_promotion_attestation(root, summary, bundles, expected_run_key=run_key)
        problems = verify_promotion_attestation(root, expected_run_key=run_key)
        if problems:
            raise PromotionContractError("bootstrapped attestation failed verification: " + "; ".join(problems))
        return tuple(written)
    except Exception:
        # The guard promise is for failures before writes. Once installation
        # begins, leave evidence for forensic inspection instead of deleting it.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap immutable incumbent diagnostics.")
    parser.add_argument("--run-key", required=True)
    args = parser.parse_args(argv)
    try:
        paths = bootstrap_incumbent_diagnostics(args.run_key)
    except PromotionContractError as exc:
        print(f"bootstrap refused: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
