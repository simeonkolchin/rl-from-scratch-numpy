# HW3: RL-PLUS on logic_circuit

This repository implements a full compact protocol:
- baseline
- GRPO-only
- SFT-only
- SFT->GRPO
- RL-PLUS (lightweight single-GPU approximation)

Goal: break zero on `Hard-val`, i.e., move baseline `pass@128=0` to `pass@128>0`.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest
bash training/run_experiment.sh
```

## Main scripts
- `training/make_datasets.py`
- `training/make_hard_subsets.py`
- `training/make_gold.py`
- `training/train_sft.py`
- `training/train_grpo.py`
- `training/train_rlplus.py`
- `training/eval_passk.py`
- `training/plot_report.py`

## Fixed evaluation params
- temperature: `1.0`
- top_p: `0.95`
- max_tokens: `256`

## Notes
- Hard subsets are built with two-phase filtering: `n_screen=16`, then strict `n_final=128` with `c=0`.
- Gold trajectories are algorithmic and verifier-checked.
