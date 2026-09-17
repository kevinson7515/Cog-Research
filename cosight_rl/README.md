# Co-Sight RL

This directory adds a JADE/verl-style RL path for Co-Sight without rewriting the
existing SFT, tool, or report-generation stack.

## 1. Background

The SFT model can already follow the Co-Sight workflow, but current failures are
mostly hard failures: wrong visual grounding, fabricated sources/data, format
violations, quantitative mistakes, and repeated report collapse. The RL path is:

Step 1:

```bash
bash cosight_rl/scripts/run_reward_audit.sh
```

Step 2:

```bash
GPU_COUNT=1 bash cosight_rl/scripts/run_cosight_grpo.sh
```

Step 3:

```bash
bash cosight_rl/scripts/merge_rl_lora.sh
bash cosight_rl/scripts/eval_rl_model.sh
```

## 2. Directory Structure

```text
cosight_rl/
  rewards/
    repetition.py
    reward_evaluator.py
    audit_rewards.py
  envs/
    cosight_env.py
    trajectory.py
    role_prompts.py
  trainer/
    config.py
    run_grpo.py
    run_ppo.py
  scripts/
    run_reward_audit.sh
    run_cosight_grpo.sh
    run_cosight_ppo.sh
    merge_rl_lora.sh
    eval_rl_model.sh
```

## 3. Reward Design

The evaluator computes.  `R_evidence_factuality` now blends citation integrity
with optional local-evidence precision/coverage from the prepared RL row, so
unresolved citation ids, orphan reference dumps, and URLs outside the local
evidence pack are penalized during GRPO.

```text
R_raw =
  0.30 * R_visual_grounding
+ 0.25 * R_evidence_factuality
+ 0.15 * R_task_following
+ 0.12 * R_quantitative_reasoning
+ 0.10 * R_report_quality
+ 0.08 * R_workflow
- P_cost
- P_invalid_tool
- P_repeat_soft
```

Then:

```text
R = min(R_raw, reward_cap)
R = clamp(R, 0, 1)
```

The reward is deterministic by default. Optional judge tags can be passed in
`extra_info.failure_tags`, `extra_info.judge`, or `extra_info.judge_reason`.

## 4. Hard Gates

Hard gates cap the final reward:

```text
empty_or_crashed          0.05
wrong_visual_identity    0.15
fabricated_source_data   0.20
repetition_collapse      0.25
format_instruction_fail  0.35
core_quant_error         0.40
```

Wrong visual identity and core quantitative reversals are safest when supplied
by a judge label. Repetition collapse and empty/crashed outputs are detected
directly from the report text.

## 5. Repetition Detection

`rewards/repetition.py` normalizes text before duplicate detection:

- lowercases text
- compresses whitespace
- removes Markdown citation ids such as `[1]`, `[2][3]`, `[12, 14]`
- normalizes URLs and Markdown-link URL parts
- removes punctuation differences while preserving Chinese, English, digits,
  and percent signs

Severe repetition is true when:

```text
duplicate_paragraph_ratio > 0.25
or max_repeated_paragraph_count > 5
or a repeated sentence block is extreme
```

The soft penalty is:

```text
min(0.6,
  0.65 * duplicate_paragraph_ratio
+ 0.15 * duplicate_sentence_ratio
+ 0.10 * min(max_repeated_paragraph_count / 10, 1)
+ 0.06 * min(max_repeated_sentence_count / 25, 1)
+ 0.04 * min(duplicate_ngram_ratio * 2, 1)
)
```

For evidence-conditioned RL data, the launcher defaults
`COSIGHT_MIN_EVIDENCE_COVERAGE=0.75`.  Data preparation fails early when fewer
than 75% of prepared train rows contain local evidence, which catches stale
or empty `data/rl_cosight_rubric` builds.

## 6. Reward Audit

Default audit:

```bash
bash cosight_rl/scripts/run_reward_audit.sh
```

Useful overrides:

```bash
WORKSPACE=./work_space/work_space_20260621_075554 \
OUTPUT=./outputs/reward_audit_20260621.jsonl \
SUMMARY_OUTPUT=./outputs/reward_audit_20260621.summary.json \
bash cosight_rl/scripts/run_reward_audit.sh
```

The JSONL contains per-report rewards, hard gates, sub-scores, repetition
metrics, and a short diagnosis. The summary JSON reports average reward, severe
repetition rate, hard failure rates, average tool calls, estimated report token
cost, worst samples, and top repeated reports. If
`work_space/work_space_20260621_075554` is absent, the shell script falls back to
the latest `work_space/work_space_*` directory unless `WORKSPACE` is set.

## 7. GRPO Training

No-GPU smoke test:

```bash
bash cosight_rl/scripts/run_cosight_grpo.sh --dry-run
```

This writes:

```text
outputs/cosight_rl_dry_run/trajectories.jsonl
outputs/cosight_rl_dry_run/trajectories.unified.jsonl
```

On the A100 Slurm cluster the scripts can be submitted directly. They resolve
`PROJECT_ROOT` from the checked-out repository by default, while still allowing
cluster-specific `PROJECT_ROOT`, `SIF_IMAGE`, `CONDA_ENV`, proxy, compiler, and
GPU settings to be overridden from the environment.

```bash
sbatch cosight_rl/scripts/run_reward_audit.sh
sbatch cosight_rl/scripts/run_cosight_grpo.sh
sbatch cosight_rl/scripts/merge_rl_lora.sh
sbatch cosight_rl/scripts/eval_rl_model.sh
```

`run_cosight_grpo.sh` defaults to 4 A100 GPUs. Override at submission time if needed:

```bash
sbatch --gres=gpu:1 cosight_rl/scripts/run_cosight_grpo.sh
sbatch --gres=gpu:2 cosight_rl/scripts/run_cosight_grpo.sh
sbatch --gres=gpu:4 cosight_rl/scripts/run_cosight_grpo.sh
```

All cluster paths can still be overridden:

```bash
PROJECT_ROOT=/home/export/base/ycsc_chenkh/hitici_07/online1/Co-Sight-0 \
CONDA_ENV=/home/export/base/ycsc_chenkh/hitici_07/online1/anaconda3/envs/co-sight \
sbatch cosight_rl/scripts/run_cosight_grpo.sh
```

One-step JADE trainer smoke test on the server:

```bash
GPU_COUNT=1 bash cosight_rl/scripts/run_cosight_grpo.sh --dry-run-trainer
```

Formal run:

```bash
GPU_COUNT=4 \
COSIGHT_MODEL_PATH=./outputs/qwen3-vl-8b-cosight-stagewise-merged \
COSIGHT_TRAIN_JSONL=./data/quiz_train.jsonl \
COSIGHT_VAL_JSONL=./data/quiz.jsonl \
bash cosight_rl/scripts/run_cosight_grpo.sh
```

Use the `COSIGHT_*` variables for overrides. Generic names such as `MODEL_PATH`
are ignored by the Slurm wrapper so an inherited shell variable cannot silently
switch the training checkpoint.

Defaults match the original lightweight critic-free GRPO path:

- LoRA rank 16, alpha 32
- bf16 rollout
- KL loss to the SFT reference
- vLLM rollout
- GRPO advantage inside the JADE/verl PPO engine
- rollout `n=2`
- rollout `max_num_seqs=8`
- actor/ref/log-prob micro-batch size 1
- torch compile disabled for A100 startup stability
- vLLM LoRA disabled by default
- transfer_queue disabled by default

For a larger rollout group after the first stable run:

```bash
COSIGHT_ROLLOUT_N=4 \
COSIGHT_ROLLOUT_MAX_NUM_SEQS=8 \
COSIGHT_TRAIN_BATCH_SIZE=16 \
sbatch --gres=gpu:4 cosight_rl/scripts/run_cosight_grpo.sh
```

For stricter GAE/PPO:

```bash
COSIGHT_ADV_ESTIMATOR=gae GPU_COUNT=4 bash cosight_rl/scripts/run_cosight_grpo.sh
```

`run_cosight_ppo.sh` is kept only as a backward-compatible wrapper that forwards
to `run_cosight_grpo.sh`.

Multi-turn tool rollouts are exposed but not enabled by default:

```bash
COSIGHT_MULTI_TURN=true \
COSIGHT_TOOL_CONFIG_PATH=/path/to/verl_tool_config.yaml \
COSIGHT_ROLLOUT_BACKEND=sglang \
bash cosight_rl/scripts/run_cosight_grpo.sh
```

## 8. LoRA Merge

```bash
bash cosight_rl/scripts/merge_rl_lora.sh
```

Common overrides:

```bash
BASE_MODEL=./outputs/qwen3-vl-8b-cosight-stage4-report-replay-merged \
RL_CKPT_DIR=./outputs/qwen3-vl-8b-cosight-rl-lora \
OUTPUT_DIR=./outputs/qwen3-vl-8b-cosight-rl-merged \
bash cosight_rl/scripts/merge_rl_lora.sh
```

If the adapter is not under the default checkpoint tree, set `RL_LORA_PATH`.

## 9. Evaluation

```bash
bash cosight_rl/scripts/eval_rl_model.sh
```

This evaluates a workspace with the same reward audit metrics:

- average reward
- hard failure rates
- severe repetition rate
- wrong visual identity rate
- fabricated source/data rate
- format violation rate
- quantitative fatal error rate
- worst samples and repeated reports

To evaluate a newly generated workspace:

```bash
WORKSPACE=./work_space/work_space_new_run bash cosight_rl/scripts/eval_rl_model.sh
```

## 10. Dry-Run Debug

No-GPU trajectory/reward dry-run:

```bash
python -m cosight_rl.trainer.run_grpo \
  --dry-run-env \
  --train-jsonl data/quiz_train.jsonl \
  --max-train-samples 2
```

Print the exact JADE command without running:

```bash
python -m cosight_rl.trainer.run_grpo --print-command --skip-prepare
```

## 11. FAQ

**Does this redo SFT?**
No. Policy and reference initialize from
`outputs/qwen3-vl-8b-cosight-stage4-report-replay-merged`.

**Where is the reward function loaded by JADE?**
`cosight_rl/rewards/reward_evaluator.py::compute_score`.

**Why default to GRPO advantage?**
The entrypoint is still JADE/verl PPO trainer. GRPO avoids a critic model and
is much easier to fit on 1-4 A100 GPUs. Set `COSIGHT_ADV_ESTIMATOR=gae` for
critic-based PPO.

**Can it audit the bad repeated reports now?**
Yes. The unit tests and reward audit target the known repeated reports in
`work_space/work_space_20260621_075554`.

**Does it delete outputs/logs/work_space?**
No. Scripts only write new audit, data, checkpoint, and merged-model outputs.
