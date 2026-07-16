# Contributing

Thanks for your interest in this project. It's a small, educational codebase — the priority is that the RL mechanics stay readable and correct.

## Ways to contribute

- **Bug reports** — if a derivation or a piece of code looks wrong, open an issue with the cell and a short explanation.
- **Reproducibility** — multi-seed runs, seed-averaged plots, or confidence intervals for the policy-gradient comparison are especially welcome (see the seed-sensitivity caveat in the README).
- **New variance-reduction schemes** — e.g. GAE, reward-to-go with discounted baselines, or a proper actor-critic — as long as they stay pure NumPy and keep the "no autograd" spirit.

## Guidelines

- Keep dependencies minimal: **NumPy + Matplotlib** only for the core; Jupyter for running the notebook.
- No PyTorch / TensorFlow / gym in the core implementation — hand-written gradients are the whole point.
- Before opening a PR, run the notebook top to bottom and make sure it executes without errors.
- Strip notebook output before committing if the diff is large:
  ```bash
  pip install nbstripout
  nbstripout HW_Week1_AI.ipynb
  ```

## Development setup

```bash
pip install -r requirements.txt
jupyter notebook HW_Week1_AI.ipynb
```
