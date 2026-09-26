"""One-click reproduction (cross-platform).

Creates an isolated venv, installs the *locked* dependency set, runs the test
suite, then runs the end-to-end demo and writes benchmark.json.

    python scripts/bootstrap.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR = os.path.join(ROOT, ".venv")
LOCK = os.path.join(ROOT, "requirements.lock.txt")


def _py() -> str:
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def _run(args, cwd=ROOT, check=True):
    print(f"\n$ {' '.join(args)}")
    return subprocess.run(args, cwd=cwd, check=check)


def main() -> int:
    print("=== IntentForge · one-click reproduction ===")

    if not os.path.exists(VENV_DIR):
        print("[1/4] creating virtualenv ...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print("[1/4] reusing existing virtualenv")

    print("[2/4] installing locked dependencies ...")
    _run([_py(), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    _run([_py(), "-m", "pip", "install", "-r", LOCK])

    print("[3/4] running unit tests ...")
    r = _run([_py(), "-m", "pytest", "tests", "-q", "-W", "ignore::UserWarning"], check=False)
    if r.returncode != 0:
        print("!! tests failed - aborting")
        return 1

    print("[4/4] running end-to-end demo (writes benchmark.json) ...")
    _run([_py(), os.path.join("intentforge", "examples", "run_demo.py")])

    print("\n=== done ===")
    print("artifacts: benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
