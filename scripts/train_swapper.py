#!/usr/bin/env python3
"""Check a local dataset manifest and print a training plan.

This script does not download data and does not run a training loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from barrett_faceswap.i18n import t
from barrett_faceswap.train import training_plan, validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a commercial-ok manifest. Does not train.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate and exit 0. Do not train.")
    args = parser.parse_args(argv)
    spec = validate_dataset(args.dataset)
    for line in training_plan(spec):
        print(line)
    if args.check:
        return 0
    print(t("train.not_run"), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
