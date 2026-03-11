"""
Save run metadata (command + resolved config) to the output directory.

Creates two files in *output_dir*:
    command.txt  – the exact shell command that was used to launch the script
    config.json  – all argparse arguments as a JSON object

Usage::

    from dfine.utils.run_info import save_run_info
    save_run_info(output_dir, args)          # uses sys.argv automatically
    save_run_info(output_dir, args, argv)    # pass explicit argv list if needed
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def save_run_info(output_dir: str, args, argv: list[str] | None = None) -> None:
    """
    Write ``command.txt`` and ``config.json`` to *output_dir*.

    Args:
        output_dir: Directory where the files will be written (created if needed).
        args:       Parsed ``argparse.Namespace`` (or any object with ``__dict__``).
        argv:       Command-line tokens to record.  Defaults to ``sys.argv``.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    if argv is None:
        argv = sys.argv

    # ---- command.txt ----
    command = " ".join(argv)
    (out / "command.txt").write_text(command + "\n", encoding="utf-8")

    # ---- config.json ----
    cfg = vars(args) if hasattr(args, "__dict__") else dict(args)
    # Make sure every value is JSON-serialisable (torch.device, Path, etc.)
    serialisable = {}
    for k, v in cfg.items():
        try:
            json.dumps(v)
            serialisable[k] = v
        except (TypeError, ValueError):
            serialisable[k] = str(v)

    (out / "config.json").write_text(
        json.dumps(serialisable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
