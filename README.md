<div align="center">

<h1>🧠 Reinforcement Learning from Scratch — Pure NumPy</h1>

**Policy gradients & behavior cloning with a hand-written CartPole, hand-written Adam, and hand-written backprop — no PyTorch, no gym, no autograd.**

![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

## TL;DR

A single self-contained notebook that implements deep RL **entirely from first principles in NumPy**:

- a custom **CartPole** environment (physics integrated by hand),
- an MLP **policy** and MLP **value** network,
- a **hand-written Adam** optimizer and manual **backprop** (no automatic differentiation),
- **policy gradient** with four variance-reduction schemes — `vanilla`, `mean_baseline`, `value_baseline`, `rloo`,
- a **behavior cloning** study with a dataset-size ablation (`full` / `prefix20` / `tiny-10eps`).

The goal is pedagogical: every gradient in this repo is derived and coded by hand, so the mechanics of policy gradients and imitation learning are fully visible.

## 🗂 Repository structure

This repo collects several homeworks from the *"Selected Topics in AI"* course. **The rest of this README documents Week 1 in depth** (the Pure-NumPy RL notebook); the later weeks live in self-contained subdirectories, each with its own README/report.

| Path | Week | Topic |
|---|---|---|
| `HW_Week1_AI.ipynb`, `results_week1/`, `docs/methods.md` | 1 | RL from scratch in pure NumPy — CartPole, policy gradients, behavior cloning (documented below). |
| [`week2/`](week2/) | 2 | `logic_circuit` — a verifiable, difficulty-tunable RL environment for an LLM agent, a FastAPI serving layer, and GRPO training/eval. See [`week2/README.md`](week2/README.md). |
| [`week3/`](week3/) | 3 | RL-PLUS on `logic_circuit` — a compact protocol comparing baseline / GRPO / SFT / SFT→GRPO / RL-PLUS with pass@k evaluation. See [`week3/README.md`](week3/README.md). |
| [`week7/`](week7/) | 7 | Toy models of superposition & sparse autoencoders (SAE) — a NumPy interpretability study of embedding geometry. See [`week7/report.md`](week7/report.md). |

> Weeks 2, 3 and 7 involve LLM/GPU training and are not "pure NumPy"; the pure-NumPy, no-autograd work is Week 1. Each subdirectory is independent and carries its own dependencies and instructions.

## 🎯 What's inside

| Component | Details |
|---|---|
| **Environment** | `CartPoleEnv` — classic cart-pole dynamics (gravity, pole mass/length, semi-implicit Euler), 500-step episodes, reward `+1` per step |
| **Policy network** | `MLPBinaryPolicy` — `4 → 32 → 1` MLP, `tanh` hidden, Bernoulli action via `sigmoid(logit)` |
| **Value network** | `MLPValue` — `4 → 32 → 1` MLP used as a learned baseline |
| **Optimizer** | `Adam` — bias-corrected first/second moments, written from scratch on plain dicts of NumPy arrays |
| **PG variants** | `vanilla`, `mean_baseline` (EMA of returns), `value_baseline` (learned `V(s)`), `rloo` (leave-one-out over episodes) |
| **Regularization** | entropy bonus with a linear schedule (`0.01 → 0.0`), advantage normalization |
| **Imitation** | `train_bc` — collect expert `(state, action)` pairs, train a fresh policy with BCE |

## 📊 Results

<div align="center">
<img src="results_week1/pg_eval_returns.png" width="46%" alt="Policy gradient evaluation returns per update"/>
<img src="results_week1/bc_scores.png" width="46%" alt="Behavior cloning degradation by dataset size"/>
</div>

**Policy gradient (deterministic eval, 20 episodes, seed 42 run — `results_week1/summary.json`):**

| Variant | Final eval return |
|---|---|
| `vanilla` | **500.0** |
| `mean_baseline` | **500.0** |
| `value_baseline` | 471.6 |
| `rloo` | 487.5 |

All four variants learn to (near-)solve CartPole (max return 500). They differ mainly in stability and variance of the gradient, not in final ceiling.

**Behavior cloning ablation (deterministic eval, 40 episodes):**

| Dataset | Size (state-action pairs) | Eval return |
|---|---|---|
| `full` | 99,745 | **500.0** |
| `prefix20` (truncated coverage) | 4,000 | 313.5 |
| `tiny` (10 episodes) | 5,000 | 360.4 |

BC matches the expert on the full dataset but degrades sharply when state coverage is limited — the classic distribution-shift / covariate-shift failure mode of naive imitation.

> ### ⚠️ Seed sensitivity caveat
> This project ships **two** summary files that **disagree**, and that disagreement is the point:
> - `results_week1/summary.json` — best variant `vanilla` (500.0), all variants ≥ 471.
> - `results_week1_notebook/summary.json` — best variant `value_baseline` (494.5), while `vanilla` collapses to **226.75** and `rloo` to 305.1.
>
> Same config, same seed field (42) — but the runs land in very different basins. Single-seed policy-gradient numbers on CartPole are **not** reliable rankings; treat them as illustrative and average over seeds before drawing conclusions.

## 🔁 Reproduce

```bash
git clone https://github.com/simeonkolchin/rl-from-scratch-numpy.git
cd rl-from-scratch-numpy
pip install -r requirements.txt
jupyter notebook HW_Week1_AI.ipynb
```

Run the notebook top to bottom. The training cells regenerate `results_week1_notebook/summary.json`; the PNGs under `results_week1/` are the reference figures referenced above.

## 📚 Documentation

- [`docs/methods.md`](docs/methods.md) — derivations and implementation notes: CartPole dynamics, the from-scratch Adam, the four PG estimators, and the BC setup.

## 🛠 Stack

`NumPy` · `Matplotlib` · `Jupyter`

## 🧑‍💻 Contributing

Contributions and questions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<sub>Coursework note: written for a "Selected Topics in AI" course. This page documents Week 1 (the pure-NumPy RL notebook); Weeks 2, 3 and 7 live under <a href="week2/">week2/</a>, <a href="week3/">week3/</a> and <a href="week7/">week7/</a>. Original narration/reports were in Russian; this top-level README and <a href="docs/methods.md">docs/</a> are an English write-up of the Week 1 work.</sub>
