#!/usr/bin/env python3
"""VERL custom reward entrypoint for Co-Sight RL."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cosight_rl.rewards.reward_evaluator import compute_score  # noqa: E402,F401
