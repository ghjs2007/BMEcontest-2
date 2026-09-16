"""Build the standalone raw-data inference distribution from canonical source.

The generated ``dist/inference`` tree is deliberately a build product.  Its
``event_stack`` package is copied from, and mechanically import-rewritten from,
the canonical implementation; it is never a hand-maintained second algorithm.

The staging/manifest/probe/atomic-replace core is shared with the competition
submission builder (``scripts/build_submission.py``), which parameterizes the
entrypoint and vendored import closure instead of forking this machinery.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.artifacts import (  # noqa: E402
    load_current_promoted_release,
    verify_bundle_manifest,
)


_RUNTIME_ROOT_MODULES = ("src.pipeline.inference.predictor",)
_PIN_NAMES = (
    ("numpy", "numpy"),
    ("joblib", "joblib"),
    ("scikit_learn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
)
_GENERATED_BASE_FILES = frozenset({"manifest.json", "feature_schema.json", "requirements.txt", "README.md"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _module_path(root: Path, module: str) -> Path | None:
    """Return an existing repository-local source file for an absolute module."""
    parts = module.split(".")
    if not parts or parts[0] != "src":
        return None
    direct = root.joinpath(*parts).with_suffix(".py")
    if direct.is_file():
        return direct
    package = root.joinpath(*parts, "__init__.py")
    return package if package.is_file() else None


def _module_name(root: Path, source: Path) -> str:
    relative = source.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _top_level_local_imports(root: Path, source: Path) -> set[Path]:
    """Resolve only import-time local dependencies of a canonical module.

    Imports nested inside diagnostic-only functions are intentionally excluded:
    they are not part of raw predictor runtime closure and would drag research
    loaders/caches into the distribution.  Every module copied here is still
    discovered by recursively parsing its own import-time imports.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    current = _module_name(root, source).split(".")
    if source.name != "__init__.py":
        current = current[:-1]
    found: set[Path] = set()
    for node in tree.body:
        module: str | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                path = _module_path(root, alias.name)
                if path is not None:
                    found.add(path)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if node.level > len(current):
                raise ValueError(f"relative import escapes canonical source: {source}")
            prefix = current[: len(current) - node.level + 1]
            module = ".".join(prefix + (node.module.split(".") if node.module else []))
        else:
            module = node.module
        if module:
            path = _module_path(root, module)
            if path is not None:
                found.add(path)
            elif module.startswith("src"):
                raise ValueError(f"local import resolves outside runtime closure: {module} from {source}")
    return found


def runtime_source_closure(repository_root: Path, *,
                           roots: Sequence[str] = _RUNTIME_ROOT_MODULES) -> tuple[Path, ...]:
    """Compute the complete import-time canonical source closure for the given roots."""
    root = Path(repository_root).resolve()
    pending = [path for module in roots if (path := _module_path(root, module))]
    if len(pending) != len(roots):
        raise ValueError("canonical Predictor source is missing")
    closure: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in closure:
            continue
        if source.is_symlink():
            raise ValueError(f"canonical runtime source may not be a symlink: {source}")
        closure.add(source)
        pending.extend(sorted(_top_level_local_imports(root, source), key=str))
    # Packages must be explicit files in the generated closure, not accidental
    # namespace packages supplied by a repository parent.
    for source in tuple(closure):
        relative = source.relative_to(root)
        for parent in relative.parents:
            if parent == Path("src"):
                continue
            init = root / parent / "__init__.py"
            if init.is_file():
                closure.add(init)
    return tuple(sorted(closure, key=lambda item: item.relative_to(root).as_posix()))


def _destination_for_source(repository_root: Path, source: Path, runtime_root: Path) -> Path:
    relative = source.relative_to(repository_root)
    if relative.parts[:2] == ("src", "pipeline"):
        return runtime_root / Path(*relative.parts[2:])
    if relative == Path("src/config.py"):
        return runtime_root / "config.py"
    if relative.parts[:2] == ("src", "eval"):
        return runtime_root / "eval" / Path(*relative.parts[2:])
    raise ValueError(f"runtime source is outside permitted canonical namespaces: {relative}")


def _rewrite_imports(text: str) -> str:
    """Mechanically rewrite canonical imports for the vendored package name."""
    return (
        text.replace("from src.pipeline.", "from event_stack.")
        .replace("import src.pipeline.", "import event_stack.")
        .replace("from src.eval.", "from event_stack.eval.")
        .replace("import src.eval.", "import event_stack.eval.")
        .replace("import src.config as config", "from event_stack import config")
        .replace("from src import config", "from event_stack import config")
    )


def copy_canonical_runtime_source(repository_root: Path, destination: Path, *,
                                  roots: Sequence[str] = _RUNTIME_ROOT_MODULES) -> tuple[str, ...]:
    """Vendor exactly the parsed canonical closure under ``event_stack``."""
    root = Path(repository_root).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for source in runtime_source_closure(root, roots=roots):
        target = _destination_for_source(root, source, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_rewrite_imports(source.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        copied.append(target.relative_to(destination.parent).as_posix())
    return tuple(sorted(copied))


def _requirements(bundle_path: Path) -> str:
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    versions = manifest.get("dependency_versions")
    if not isinstance(versions, dict):
        raise ValueError("deployment manifest dependency versions are missing")
    lines: list[str] = []
    for key, distribution in _PIN_NAMES:
        value = versions.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"deployment manifest {key} pin is invalid")
        lines.append(f"{distribution}=={value}")
    return "\n".join(lines) + "\n"


_PREDICT_ENTRYPOINT = '''"""Standalone raw event-stack inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_stack.inference import Predictor

def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="collect_data*.txt file or folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    return parser.parse_args(argv)

def main(argv=None):
    args = _args(argv)
    try:
        manifest = json.loads((_ROOT / "manifest.json").read_text(encoding="utf-8"))
        predictor = Predictor.from_bundle(_ROOT / "models", device=args.device,
                                          run_key=str(manifest["release_run_key"]))
        options = predictor.options(include_timeline=args.include_timeline, include_candidates=args.include_candidates, device=args.device)
        result = predictor.predict_folder(args.input, options=options) if args.input.is_dir() else predictor.predict_file(args.input, options=options)
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"event-stack inference refused: {exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


_README = """# Standalone Event-Stack Inference

This directory is generated from the promoted canonical release. Do not edit
`event_stack/` manually; rebuild it with `python scripts/build_inference_distribution.py`.

Install the exact recorded dependencies, then predict an official raw
`collect_data*.txt` file or a directory containing such files:

```bash
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
```

`--device cpu` is supported. The promoted release has no audited CUDA adapter,
so forcing `--device gpu` or `--device cuda` is rejected rather than silently
falling back to CPU. The JSON result follows `dist/schema/prediction.schema.json`.
"""


def _distribution_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and not path.is_symlink() and path.name != "manifest.json"
    }


def verify_distribution_manifest(package: Path, *, entrypoint: str = "predict.py") -> None:
    """Reject hash drift, symlinks and undeclared runtime files."""
    package = Path(package)
    if package.is_symlink() or not package.is_dir():
        raise ValueError("distribution package must be a real directory")
    generated = _GENERATED_BASE_FILES | {entrypoint}
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    required = {"release_run_key", "model_version", "prediction_schema_version", "feature_schema", "python_version", "dependencies", "model_files", "source_files", "hashes"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("distribution manifest schema is invalid")
    for name in generated | {"event_stack", "models"}:
        if not (package / name).exists():
            raise ValueError(f"distribution file is missing: {name}")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError("distribution may not contain symlinks")
    actual = _distribution_files(package)
    if manifest["hashes"] != actual:
        raise ValueError("distribution manifest file checksums do not match")
    source_files = sorted(path for path in actual if path.startswith("event_stack/"))
    model_files = sorted(path for path in actual if path.startswith("models/"))
    if manifest["source_files"] != source_files or manifest["model_files"] != model_files:
        raise ValueError("distribution manifest source/model closure does not match")
    if set(actual) - generated - set(source_files) - set(model_files):
        raise ValueError("distribution contains undeclared runtime files")


def _atomic_replace(staging: Path, destination: Path) -> Path:
    destination = Path(destination)
    parent = destination.parent
    backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        if destination.exists():
            if destination.is_symlink():
                raise ValueError("distribution destination may not be a symlink")
            os.replace(destination, backup)
            moved = True
        os.replace(staging, destination)
        if moved:
            shutil.rmtree(backup)
        return destination
    except Exception:
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination)
        if moved and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and destination.exists():
            shutil.rmtree(backup)


def build_distribution(*, repository_root: Path, bundle_path: Path, destination: Path,
                       entrypoint: str, entrypoint_text: str, readme_text: str,
                       closure_roots: Sequence[str] = _RUNTIME_ROOT_MODULES) -> Path:
    """Atomically construct a verified distribution package for the active release.

    Shared by the inference distribution (``predict.py``) and the competition
    submission (``main.py`` with the additional adapter closure root).
    """
    root = Path(repository_root).resolve()
    release = load_current_promoted_release(root)
    bundle = Path(bundle_path).resolve()
    expected = root / "models" / "event_stack" / str(release["run_key"]) / "deployment"
    if bundle != expected:
        raise ValueError("distribution must use the current promoted deployment bundle")
    problems = verify_bundle_manifest(bundle, expected_run_key="deployment")
    if problems:
        raise ValueError("deployment bundle manifest verification failed: " + "; ".join(problems))
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.is_symlink():
        raise ValueError("distribution destination may not traverse symlinks")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        copy_canonical_runtime_source(root, staging / "event_stack", roots=closure_roots)
        shutil.copytree(bundle, staging / "models", symlinks=False)
        (staging / entrypoint).write_text(entrypoint_text, encoding="utf-8", newline="\n")
        (staging / "requirements.txt").write_text(_requirements(bundle), encoding="utf-8", newline="\n")
        shutil.copy2(bundle / "feature_schema.json", staging / "feature_schema.json")
        (staging / "README.md").write_text(readme_text, encoding="utf-8", newline="\n")
        model_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest = {
            "release_run_key": str(release["run_key"]),
            "model_version": model_manifest.get("bundle_version"),
            "prediction_schema_version": "1.0",
            "feature_schema": json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8")),
            "python_version": model_manifest["dependency_versions"]["python"],
            "dependencies": {key: model_manifest["dependency_versions"][key] for key, _ in _PIN_NAMES},
            "model_files": [], "source_files": [], "hashes": {},
        }
        manifest["hashes"] = _distribution_files(staging)
        manifest["source_files"] = sorted(path for path in manifest["hashes"] if path.startswith("event_stack/"))
        manifest["model_files"] = sorted(path for path in manifest["hashes"] if path.startswith("models/"))
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        verify_distribution_manifest(staging, entrypoint=entrypoint)
        probe_environment = dict(os.environ)
        probe_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        probe = subprocess.run(
            [sys.executable, "-I", entrypoint, "--help"], cwd=staging,
            capture_output=True, text=True, env=probe_environment,
        )
        if probe.returncode != 0:
            raise RuntimeError("isolated distribution import probe failed: " + probe.stderr.strip())
        # A distribution is source/model only.  The isolated import probe must
        # not make bytecode a generated runtime dependency.
        for cache in staging.rglob("__pycache__"):
            shutil.rmtree(cache)
        verify_distribution_manifest(staging, entrypoint=entrypoint)
        return _atomic_replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_inference_distribution(*, repository_root: Path, bundle_path: Path, destination: Path) -> Path:
    """Atomically construct a verified raw-inference package for the active release."""
    return build_distribution(
        repository_root=repository_root, bundle_path=bundle_path, destination=destination,
        entrypoint="predict.py", entrypoint_text=_PREDICT_ENTRYPOINT, readme_text=_README,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/event_stack/160afaf81debf1ee/deployment")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/inference")
    args = parser.parse_args(argv)
    try:
        built = build_inference_distribution(repository_root=ROOT, bundle_path=args.bundle, destination=args.destination)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"inference distribution build refused: {exc}", file=sys.stderr)
        return 2
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
