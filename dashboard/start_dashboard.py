#!/usr/bin/env python3
"""
Compatibility entrypoint for Render when Root Directory is set to `dashboard/`.

In that configuration, Render copies only the `dashboard/` directory and the
service start command may still be:
  python start_dashboard.py

This shim delegates to the canonical Render-safe entrypoint:
  app_render.py
"""

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    here = Path(__file__).resolve().parent
    target = here / "app_render.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

