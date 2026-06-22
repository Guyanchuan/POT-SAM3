# -*- coding: utf-8 -*-
"""Backward-compatible thin entrypoint for training."""

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from potsam.train import main
else:
    from .train import main


if __name__ == "__main__":
    main()
