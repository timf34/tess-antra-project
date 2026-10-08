"""Snapshot current local source, including uncommitted fixes, excluding secrets and generated data.

Run from the repo: uv run python scripts/local/package_collection.py
The archive is for upload to /workspace/tess-antra-project on a new experiment pod.
"""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    dest = ROOT / "artifacts/collection_source.tar.gz"
    dest.parent.mkdir(exist_ok=True)
    manifest = {}
    with tarfile.open(dest, "w:gz") as archive:
        for name in sorted(set(filter(None, names))):
            path = ROOT / name
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing symlink in snapshot: {name}")
            if any(
                part in {".git", ".venv", "artifacts", "runs", "__pycache__"}
                for part in path.relative_to(ROOT).parts
            ):
                continue
            if path.name == ".env" or path.name.endswith(".local.env"):
                continue
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
    (dest.parent / "collection_source_manifest.json").write_text(
        json.dumps(
            {"archive_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "files": manifest}, indent=2
        )
    )
    print(f"{dest}: {len(manifest)} source files; credentials and generated data excluded")


if __name__ == "__main__":
    main()
