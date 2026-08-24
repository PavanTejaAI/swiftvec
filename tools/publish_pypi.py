import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = "crates/python/Cargo.toml"
DIST = ROOT / "dist"


def run(cmd):
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=True)


def versions():
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    pyproject = (ROOT / "crates" / "python" / "pyproject.toml").read_text(encoding="utf-8")
    cv = re.search(r'version\s*=\s*"([^"]+)"', cargo).group(1)
    pv = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE).group(1)
    return cv, pv


def main() -> int:
    if not shutil.which("uv"):
        print("error: uv is required (https://docs.astral.sh/uv/)")
        return 1
    if not os.environ.get("UV_PUBLISH_TOKEN"):
        print("error: set UV_PUBLISH_TOKEN to your PyPI token first")
        return 1

    cv, pv = versions()
    if cv != pv:
        print(f"error: version mismatch Cargo.toml {cv} != pyproject.toml {pv}")
        return 1
    print(f"version {cv}: ok")

    if DIST.exists():
        for f in DIST.iterdir():
            f.unlink()

    run(["uvx", "maturin", "build", "--release", "-m", MANIFEST, "-o", "dist"])

    artifacts = sorted(p for p in DIST.iterdir() if p.suffix == ".whl")
    if not artifacts:
        print("error: no wheel or sdist produced")
        return 1

    for a in artifacts:
        print(f"  artifact: {a.name}")

    run(["uv", "publish", *[str(a) for a in artifacts]])
    print(f"published swiftvec {cv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
