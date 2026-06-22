# -*- coding: utf-8 -*-
"""CLI entry for POTSAM training."""

from __future__ import annotations

import argparse
import os
import sys
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from potsam.data import normalize_config_paths
    from potsam.trainer import Trainer
else:
    from .data import normalize_config_paths
    from .trainer import Trainer


def main():
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.yaml",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_cfg)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = normalize_config_paths(cfg, args.config)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
