"""Runnable both from a fresh uv environment and the installed package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from kusama.experiment import main
    except ImportError as exc:
        print(f"Missing or incompatible dependency: {exc}. Run `uv sync --locked --dev` with Python 3.11.", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(main())
