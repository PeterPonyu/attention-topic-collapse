#!/usr/bin/env python3
"""Verify distributed analysis inputs without loading models or checkpoints."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "publication-manifest.json").read_text())
    failures, optional_absent = [], []
    verified = {"required_files": 0, "optional_checkpoints": 0}
    for group in verified:
        for relative, expected in manifest[group].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT):
                failures.append({"file": relative, "error": "path leaves the package"})
            elif not path.is_file():
                if group == "optional_checkpoints":
                    optional_absent.append(relative)
                else:
                    failures.append({"file": relative, "error": "required file missing"})
            elif sha256(path) != expected:
                failures.append({"file": relative, "error": "SHA-256 mismatch"})
            else:
                verified[group] += 1
    result = {
        "study": manifest["study"],
        "passed": not failures,
        "verified": verified,
        "optional_checkpoints_absent": optional_absent,
        "external_source_inputs_not_distributed_or_verified": manifest.get("external_source_inputs", {}),
        "failures": failures,
        "scope": "File identity only; no model import, checkpoint deserialization or training. Historical external-source digests are records, not verified local files.",
    }
    destination = ROOT / "validation/publication-check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
