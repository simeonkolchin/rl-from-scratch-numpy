# Methods & Implementation Notes

This document describes the algorithms implemented in `HW_Week1_AI.ipynb`. Everything is written in plain NumPy — there is no automatic differentiation, so every gradient below is derived and coded by hand.

## 1. Environment — `CartPoleEnv`

A from-scratch re-implementation of the classic cart-pole task:

- **State**: `[x, x_dot, theta, theta_dot]`, initialized uniformly in `[-0.05, 0.05]`.
- **Dynamics**: standard cart-pole equations (gravity `9.8`, cart mass `1.0`, pole mass `0.1`, half-length `0.5`, force magnitude `10.0`), integrated with a semi-implicit Euler step (`tau = 0.02`).
- **Termination**: pole angle beyond `±12°`, cart position beyond `±2.4`, or `500` steps reached.
- **Reward**: `+1` per surviving step, so the maximum episode return is `500`.

## 2. Networks

Two small MLPs, both `4 → 32 → 1` with a `tanh` hidden layer:

- **`MLPBinaryPolicy`** — outputs a single logit; the action is Bernoulli with `p = sigmoid(logit)`.
- **`MLPValue`** — outputs a scalar state-value estimate `V(s)`, used only by the `value_baseline` variant.

Weights use fan-in/fan-out scaled Gaussian init (`scale = sqrt(2 / (in + out))`); biases start at zero. Each network exposes its parameters as a flat dict (`l1.W`, `l1.b`, `l2.W`, `l2.b`) so the optimizer can treat them uniformly.

## 3. Optimizer — hand-written Adam

`Adam` maintains first/second moment estimates `m`, `v` per parameter tensor and applies the bias-corrected update

```
m = b1*m + (1-b1)*g
v = b2*v + (1-b2)*g^2
m_hat = m / (1 - b1^t)
v_hat = v / (1 - b2^t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

with defaults `betas = (0.9, 0.999)`, `eps = 1e-8`.

## 4. Policy gradient

The core estimator is REINFORCE with an advantage signal. Per update:

1. Collect a batch of ~`2500` environment steps under the current policy.
2. Compute discounted returns-to-go (`gamma = 0.99`).
3. Form an advantage `A` according to the chosen variant.
4. **Normalize** the advantage (`(A - mean) / (std + 1e-8)`) — reduces variance and stabilizes the step.
5. Backprop the policy-gradient loss through the logits and take one Adam step.

An **entropy bonus** is added with a linear schedule from `0.01` down to `0.0` over training, encouraging early exploration.

### Variance-reduction variants

| Variant | Advantage |
|---|---|
| `vanilla` | returns-to-go, unbaselined |
| `mean_baseline` | returns minus an exponential moving average of batch mean return (`alpha = 0.05`) |
| `value_baseline` | returns minus learned `V(s)`; the value net is trained with an MSE regression step each update |
| `rloo` | REINFORCE-leave-one-out: each episode's baseline is the mean return of the *other* episodes in the batch |

Config (`TrainConfig`): `updates=120`, `batch_steps=2500`, `hidden_dim=32`, `lr_policy=lr_value=2e-3`, `gamma=0.99`.

## 5. Behavior cloning

After PG training, the best policy is used as an **expert**:

1. Roll out the expert to collect `(state, action)` pairs.
2. Train a fresh `MLPBinaryPolicy` in supervised mode with binary cross-entropy.

Three dataset regimes probe the effect of coverage:

- **`full`** — the complete expert dataset (~100k pairs).
- **`prefix20`** — only the first 20 steps of each episode (limited state coverage).
- **`tiny`** — only 10 expert episodes.

The result (see README) is the textbook covariate-shift story: BC matches the expert with full coverage and degrades sharply as coverage shrinks.

## 6. Reproducibility caveat

CartPole policy-gradient runs are highly seed-sensitive. The two shipped `summary.json` files (`results_week1/` vs `results_week1_notebook/`) were produced with the same nominal config and disagree substantially on which variant "wins" and on whether `vanilla` even converges. Do not read single-seed rankings as conclusions — average across seeds first.
