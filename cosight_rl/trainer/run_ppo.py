"""Backward-compatible launcher for the old Co-Sight PPO command name.

The actual default setup is critic-free GRPO-style training. Prefer
``python -m cosight_rl.trainer.run_grpo`` for new runs.
"""

from __future__ import annotations

from .run_grpo import main


if __name__ == "__main__":
    main()
