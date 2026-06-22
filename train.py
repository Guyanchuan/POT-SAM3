#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin root-level training entrypoint.

Usage:
  python train.py
  python train.py --config config/config.yaml
"""

from __future__ import annotations

import os
import sys


def _inject_default_config(argv: list[str], default_cfg: str) -> list[str]:
    if "--config" in argv:
        return argv
    return [argv[0], "--config", default_cfg, *argv[1:]]


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    default_cfg = os.path.join(repo_root, "config", "config.yaml")
    sys.argv = _inject_default_config(sys.argv, default_cfg)

    from potsam.main import main as potsam_train_main

    potsam_train_main()


if __name__ == "__main__":
    main()
