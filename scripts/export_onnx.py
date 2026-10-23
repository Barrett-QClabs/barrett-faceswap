#!/usr/bin/env python3
"""Print the ONNX export contract. This script does not write a weight file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from barrett_faceswap.i18n import t
from barrett_faceswap.train import export_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the ONNX contract. Does not export weights.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Print the contract and exit 0.")
    args = parser.parse_args(argv)
    print(export_contract())
    if args.check:
        return 0
    if args.checkpoint is None or not args.checkpoint.is_file():
        missing = args.checkpoint if args.checkpoint is not None else Path("-")
        print(t("train.export_missing", path=missing), file=sys.stderr)
        return 2
    print(t("train.export_skipped"), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
