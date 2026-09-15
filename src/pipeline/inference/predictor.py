"""Canonical raw-data inference for a verified frozen event-stack bundle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from src.pipeline.artifacts import EventStackBundle, load_event_stack_bundle
from src.pipeline.candidate_control import CandidateAdmissionConfig, admit_candidates
from src.pipeline.event_stack import (
    DensityConfig, EventRef, MicroCandidateConfig, apply_event_policy,
    density_candidates, micro_candidates, multiscale_verifier_features, union_candidates,
)
from src.pipeline.features.macro import MacroFeatureConfig, add_time_prior, extract_macro_windows
from src.pipeline.features.micro import extract_micro_windows
from src.pipeline.imu_features import MicroFeatureConfig
from src.pipeline.io.raw_session import RawSessionSource, discover_raw_sessions, load_raw_session
from src.pipeline.preprocessing.timeline import valid_imu_spans

from .legacy_payload import resolve_device
from .schema import make_prediction_result, validate_prediction


@dataclass(frozen=True)
class PredictionOptions:
    include_timeline: bool = False
    include_candidates: bool = False
    device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"


def _positive_probability(model: object, features: np.ndarray, width: int, name: str) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} model features must have width {width}")
    if not len(values):
        return np.empty(0, dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = np.asarray(model.classes_)
    column = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(values), 2) or len(column) != 1:
        raise ValueError(f"{name} model returned incompatible probabilities")
    return probabilities[:, int(column[0])]


def _by_session(windows: Sequence[EventRef], scores: np.ndarray) -> dict[str, list[tuple[int, int, float]]]:
    if len(windows) != len(scores):
        raise ValueError("window scores must align with windows")
    result: dict[str, list[tuple[int, int, float]]] = {}
    for window, score in zip(windows, scores):
        result.setdefault(window.sid, []).append((window.start_ms, window.end_ms, float(score)))
    return result


def _config(raw: object, cls):
    if not isinstance(raw, dict):
        raise ValueError("frozen run config is malformed")
    return cls(**raw)


class Predictor:
    """Inference-only facade; it never reads caches, splits, labels, or training data."""

    def __init__(self, bundle: EventStackBundle, *, bundle_path: Path, run_key: str, device: str = "auto") -> None:
        self._bundle = bundle
        self._bundle_path = Path(bundle_path)
        self._run_key = run_key
        self._device = resolve_device(device)
        schema = bundle.feature_schema
        widths = schema["widths"] if "widths" in schema else schema
        if dict(widths) != {"macro": 63, "micro": 47, "verifier": 116}:
            raise ValueError("raw Predictor supports only the frozen 63/47/116 release schema")
        configs = bundle.run_config.get("fold_configs")
        if not isinstance(configs, list) or not configs or not isinstance(configs[0], dict):
            raise ValueError("bundle has no frozen fold configuration")
        frozen = configs[0]
        self._density = _config(frozen.get("density"), DensityConfig)
        self._micro_candidate = _config(frozen.get("micro_candidate"), MicroCandidateConfig)
        self._macro_config = MacroFeatureConfig(
            window_ms=self._density.window_ms, stride_ms=self._density.stride_ms,
            coverage_min=self._density.coverage_min,
        )
        self._micro_config = MicroFeatureConfig(
            window_ms=self._micro_candidate.window_ms, stride_ms=self._micro_candidate.stride_ms,
            coverage_min=self._density.coverage_min, gravity_align=bool(frozen.get("micro_gravity_align")),
        )

    @classmethod
    def from_bundle(cls, path: Path, *, device: str = "auto") -> "Predictor":
        bundle_path = Path(path)
        try:
            declared_run_key = str(json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))["run_key"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("bundle manifest run key cannot be read") from exc
        bundle = load_event_stack_bundle(bundle_path, expected_role="deployment", expected_run_key=declared_run_key)
        run_key = bundle_path.parent.name
        # Repository use can point at dist/event_stack/bundle; its release key
        # is tracked by the immutable registry, never inferred from features.
        for parent in (bundle_path, *bundle_path.parents):
            registry = parent / "release" / "event_stack_incumbent.json"
            if registry.is_file():
                try:
                    run_key = str(json.loads(registry.read_text(encoding="utf-8"))["run_key"])
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                break
        return cls(bundle, bundle_path=bundle_path, run_key=run_key, device=device)

    @staticmethod
    def options(**kwargs: object) -> PredictionOptions:
        return replace(PredictionOptions(), **kwargs)

    def predict_file(self, path: Path, *, subject_id: str | None = None,
                     options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
        path = Path(path)
        return self.predict_sources((RawSessionSource(path, path.stem, subject_id),), options)

    def predict_folder(self, path: Path, *, subject_id: str | None = None,
                       options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
        sources = tuple(replace(source, subject_id=subject_id) for source in discover_raw_sessions(Path(path)))
        return self.predict_sources(sources, options)

    def predict_sources(self, sources: Sequence[RawSessionSource], options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
        if not sources:
            raise ValueError("at least one raw source is required")
        resolved_device = resolve_device(options.device)
        # A future audited GPU adapter belongs here; cpu is the registered release runtime.
        if resolved_device != self._device and self._device != "cpu":
            raise RuntimeError("Predictor device state is inconsistent")
        macro_windows: list[EventRef] = []
        macro_rows: list[np.ndarray] = []
        micro_windows: list[EventRef] = []
        micro_rows: list[np.ndarray] = []
        bounds: dict[str, tuple[int, int]] = {}
        source_names: list[str] = []
        total_duration = 0.0
        subject_by_sid: dict[str, str] = {}
        for source in sources:
            session = load_raw_session(source)
            sid = source.session_id
            if sid in subject_by_sid:
                raise ValueError("raw session identifiers must be unique")
            spans = valid_imu_spans(session)
            if spans:
                bounds[sid] = (min(span.start_ms for span in spans), max(span.end_ms for span in spans))
                total_duration += (bounds[sid][1] - bounds[sid][0]) / 1000.0
            subject_by_sid[sid] = source.subject_id or sid
            macro = extract_macro_windows(session, session_id=sid, config=self._macro_config)
            micro = extract_micro_windows(session, session_id=sid, config=self._micro_config)
            macro_windows.extend(macro.windows); macro_rows.append(macro.features)
            micro_windows.extend(micro.windows); micro_rows.append(micro.features)
            source_names.append(str(source.path))
        macro_features_62 = np.concatenate(macro_rows, axis=0) if macro_rows else np.empty((0, 62), dtype=np.float32)
        macro_features = add_time_prior(macro_features_62, macro_windows)
        micro_features = np.concatenate(micro_rows, axis=0) if micro_rows else np.empty((0, 47), dtype=np.float32)
        macro_scores = _positive_probability(self._bundle.models["macro"], macro_features, 63, "macro")
        micro_scores = _positive_probability(self._bundle.models["micro"], micro_features, 47, "micro")
        macro_by_sid = _by_session(macro_windows, macro_scores)
        micro_by_sid = _by_session(micro_windows, micro_scores)
        candidates = union_candidates(density_candidates(macro_by_sid, self._density), micro_candidates(micro_by_sid, float(self._bundle.run_config["micro_threshold"]), self._micro_candidate))
        scored: np.ndarray
        if candidates:
            features = multiscale_verifier_features(candidates, macro_by_sid, micro_by_sid, context_features_version="v1", session_bounds_by_sid=bounds)
            if features.shape[1] != 116:
                raise RuntimeError("canonical verifier did not produce frozen 116-D features")
            logistic = _positive_probability(self._bundle.models["verifier_logistic"], features, 116, "verifier")
            lgbm = _positive_probability(self._bundle.models["verifier_lgbm"], features, 116, "verifier")
            weight = float(self._bundle.policy["blend_weight"])
            scored = weight * logistic + (1.0 - weight) * lgbm
        else:
            scored = np.empty(0, dtype=np.float64)
        admission = CandidateAdmissionConfig(float(self._bundle.policy["nms_iou"]), float(self._bundle.policy["admission_threshold"]), self._bundle.policy["max_candidates_per_subject"])
        admitted_indices = admit_candidates(candidates, scored, [subject_by_sid[c.event.sid] for c in candidates], admission)
        admitted = [candidates[index] for index in admitted_indices]
        admitted_scores = scored[list(admitted_indices)] if admitted_indices else np.empty(0, dtype=np.float64)
        final = apply_event_policy([candidate.event for candidate in admitted], admitted_scores, [subject_by_sid[candidate.event.sid] for candidate in admitted], float(self._bundle.policy["threshold"]), self._bundle.policy["max_events_per_group"])
        confidence_by_event = {(candidate.event.sid, candidate.event.start_ms, candidate.event.end_ms): float(score) for candidate, score in zip(admitted, admitted_scores)}
        events = [{"id": index, "session_id": event.sid, "start_ms": event.start_ms, "end_ms": event.end_ms,
                   "duration_s": (event.end_ms - event.start_ms) / 1000.0,
                   "confidence": confidence_by_event[(event.sid, event.start_ms, event.end_ms)]}
                  for index, event in enumerate(final)]
        result = make_prediction_result(run_key=self._run_key, source=source_names[0] if len(source_names) == 1 else str(self._bundle_path), duration_seconds=total_duration, events=events)
        result["diagnostics"] = {"coverage": None, "warnings": [], "resolved_device": resolved_device}
        if options.include_candidates:
            result["candidates"] = [{"session_id": c.event.sid, "start_ms": c.event.start_ms, "end_ms": c.event.end_ms, "score": float(score), "admitted": index in set(admitted_indices)} for index, (c, score) in enumerate(zip(candidates, scored))]
        if options.include_timeline:
            result["timeline"] = {"macro_windows": len(macro_windows), "micro_windows": len(micro_windows)}
        validate_prediction({key: result[key] for key in ("schema_version", "model", "input", "events", "diagnostics")})
        return result
