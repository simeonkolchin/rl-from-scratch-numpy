from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toy_sae import (
    SAEConfig,
    SAETrainConfig,
    ToyModel,
    ToyModelConfig,
    ToyTrainConfig,
    config_to_dict,
    recovery_metrics,
    save_json,
    train_sae,
    train_toy_model,
    evaluate_sae,
)


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "artifacts" / "figures"
RESULTS_DIR = ROOT / "artifacts" / "results"


TOY_CACHE: dict[tuple[Any, ...], ToyModel] = {}
SAE_CACHE: dict[tuple[Any, ...], Any] = {}


def set_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "font.size": 11,
        }
    )


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    value_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        item = {group_keys[index]: key[index] for index in range(len(group_keys))}
        for value_key in value_keys:
            mean_value, std_value = mean_std([float(row[value_key]) for row in group])
            item[f"{value_key}_mean"] = mean_value
            item[f"{value_key}_std"] = std_value
        out_rows.append(item)
    return out_rows


def toy_key(config: ToyModelConfig, train_config: ToyTrainConfig, seed: int) -> tuple[Any, ...]:
    return (
        config.num_features,
        config.hidden_dim,
        config.alpha,
        config.expected_l0,
        config.max_prob,
        train_config.steps,
        train_config.batch_size,
        train_config.learning_rate,
        train_config.weight_decay,
        train_config.log_every,
        train_config.snapshot_steps,
        seed,
    )


def sae_key(
    toy_model: ToyModel,
    toy_seed: int,
    sae_config: SAEConfig,
    train_config: SAETrainConfig,
    seed: int,
) -> tuple[Any, ...]:
    config = toy_model.config
    return (
        config.num_features,
        config.hidden_dim,
        config.alpha,
        config.expected_l0,
        toy_seed,
        sae_config.latent_dim,
        sae_config.l1_coeff,
        sae_config.decoder_row_norm,
        train_config.steps,
        train_config.batch_size,
        train_config.learning_rate,
        train_config.weight_decay,
        train_config.log_every,
        seed,
    )


def get_toy_model(config: ToyModelConfig, train_config: ToyTrainConfig, seed: int) -> ToyModel:
    key = toy_key(config, train_config, seed)
    if key not in TOY_CACHE:
        TOY_CACHE[key] = train_toy_model(config, train_config, seed=seed)
    return TOY_CACHE[key]


def get_sae(
    toy_model: ToyModel,
    toy_seed: int,
    sae_config: SAEConfig,
    train_config: SAETrainConfig,
    seed: int,
):
    key = sae_key(toy_model, toy_seed, sae_config, train_config, seed)
    if key not in SAE_CACHE:
        SAE_CACHE[key] = train_sae(toy_model, sae_config, train_config, seed=seed)
    return SAE_CACHE[key]


def top_eigen_share(hidden: np.ndarray, top_k: int = 1) -> float:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals.sum() <= 1e-12:
        return 0.0
    return float(np.sort(eigvals)[-top_k:].sum() / eigvals.sum())


def norm_statistics(model: ToyModel) -> dict[str, float]:
    norms = np.linalg.norm(model.weights, axis=1)
    head = norms[: min(5, norms.shape[0])]
    tail = norms[-min(5, norms.shape[0]) :]
    return {
        "alive_count": float(np.sum(norms > 0.1)),
        "head_tail_norm_ratio": float(head.mean() / max(tail.mean(), 1e-8)),
        "norm_prob_corr": float(np.corrcoef(model.probabilities, norms)[0, 1]),
    }


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    save_json(path, summary)


def plot_embedding_points(ax: plt.Axes, model: ToyModel, title: str) -> None:
    coords = model.weights
    probs = model.probabilities
    ax.scatter(coords[:, 0], coords[:, 1], c=np.log10(probs), cmap="viridis", s=48, alpha=0.85)
    for index in range(min(6, coords.shape[0])):
        ax.text(coords[index, 0], coords[index, 1], str(index + 1), fontsize=8)
    ax.axhline(0.0, color="#cccccc", lw=0.7)
    ax.axvline(0.0, color="#cccccc", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal", adjustable="box")


def plot_hidden_scatter(ax: plt.Axes, model: ToyModel, title: str, seed: int) -> None:
    hidden, _, _ = model.sample_hidden(batch_size=5000, seed=seed)
    ax.scatter(hidden[:, 0], hidden[:, 1], s=6, alpha=0.12, color="#2b7bba")
    ax.set_title(title)
    ax.set_xlabel("h[0]")
    ax.set_ylabel("h[1]")
    ax.set_aspect("equal", adjustable="box")


def plot_norm_curve(ax: plt.Axes, model: ToyModel, title: str) -> None:
    norms = np.linalg.norm(model.weights, axis=1)
    ax.plot(np.arange(1, norms.shape[0] + 1), norms, color="#c44e52", lw=2)
    ax.set_title(title)
    ax.set_xlabel("feature rank")
    ax.set_ylabel("row norm")
    ax.set_xscale("log")


def plot_line_with_band(
    ax: plt.Axes,
    x: list[float],
    mean_values: list[float],
    std_values: list[float],
    label: str,
    color: str,
) -> None:
    x_arr = np.asarray(x, dtype=np.float64)
    mean_arr = np.asarray(mean_values, dtype=np.float64)
    std_arr = np.asarray(std_values, dtype=np.float64)
    ax.plot(x_arr, mean_arr, color=color, marker="o", lw=2, label=label)
    ax.fill_between(x_arr, mean_arr - std_arr, mean_arr + std_arr, color=color, alpha=0.18)


def run_part1() -> dict[str, Any]:
    part_dir = FIGURES_DIR / "part1"
    part_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = [0.3, 1.0, 2.0]
    seeds = [0, 1, 2]
    alpha_rows: list[dict[str, Any]] = []
    representative_models: list[tuple[float, ToyModel]] = []
    alpha_config_template = dict(num_features=24, hidden_dim=2, expected_l0=3.0)
    alpha_train = ToyTrainConfig(steps=2000, batch_size=512, learning_rate=0.025, log_every=100)

    for alpha in alpha_values:
        for seed in seeds:
            model = get_toy_model(
                ToyModelConfig(alpha=alpha, **alpha_config_template),
                alpha_train,
                seed=seed,
            )
            hidden, _, _ = model.sample_hidden(batch_size=4096, seed=seed + 50)
            row = {
                "alpha": alpha,
                "seed": seed,
                "mean_abs_cosine": model.geometry_summary(batch_size=4096, seed=seed + 100)["mean_abs_cosine"],
                "top1_eigen_share": top_eigen_share(hidden, top_k=1),
                "top2_eigen_share": top_eigen_share(hidden, top_k=2),
            }
            row.update(norm_statistics(model))
            alpha_rows.append(row)
        representative_models.append(
            (
                alpha,
                get_toy_model(
                    ToyModelConfig(alpha=alpha, **alpha_config_template),
                    alpha_train,
                    seed=0,
                ),
            )
        )

    fig, axes = plt.subplots(3, len(representative_models), figsize=(15, 11))
    for column, (alpha, model) in enumerate(representative_models):
        plot_embedding_points(axes[0, column], model, title=f"Embeddings, alpha={alpha}")
        plot_hidden_scatter(axes[1, column], model, title=f"Hidden states, alpha={alpha}", seed=200 + column)
        plot_norm_curve(axes[2, column], model, title=f"Norm profile, alpha={alpha}")
    fig.tight_layout()
    fig.savefig(part_dir / "alpha_geometry.png", bbox_inches="tight")
    plt.close(fig)

    alpha_agg = aggregate_rows(
        alpha_rows,
        group_keys=["alpha"],
        value_keys=["alive_count", "head_tail_norm_ratio", "top1_eigen_share", "mean_abs_cosine"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_line_with_band(
        axes[0],
        [row["alpha"] for row in alpha_agg],
        [row["alive_count_mean"] for row in alpha_agg],
        [row["alive_count_std"] for row in alpha_agg],
        label="alive embeddings",
        color="#2b7bba",
    )
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("count")
    axes[0].set_title("How many embeddings stay active")

    plot_line_with_band(
        axes[1],
        [row["alpha"] for row in alpha_agg],
        [row["head_tail_norm_ratio_mean"] for row in alpha_agg],
        [row["head_tail_norm_ratio_std"] for row in alpha_agg],
        label="head/tail norm ratio",
        color="#c44e52",
    )
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Head features dominate more than tail")

    plot_line_with_band(
        axes[2],
        [row["alpha"] for row in alpha_agg],
        [row["top1_eigen_share_mean"] for row in alpha_agg],
        [row["top1_eigen_share_std"] for row in alpha_agg],
        label="top eig share",
        color="#55a868",
    )
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("share")
    axes[2].set_title("Anisotropy of hidden states")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(part_dir / "alpha_summary.png", bbox_inches="tight")
    plt.close(fig)

    superposition_rows: list[dict[str, Any]] = []
    superposition_features = [12, 24, 48]
    superposition_train = ToyTrainConfig(steps=1500, batch_size=512, learning_rate=0.025, log_every=100)
    for num_features in superposition_features:
        for seed in seeds:
            model = get_toy_model(
                ToyModelConfig(num_features=num_features, hidden_dim=4, alpha=1.2, expected_l0=2.5),
                superposition_train,
                seed=seed,
            )
            hidden, _, _ = model.sample_hidden(batch_size=4096, seed=400 + seed)
            row = {
                "num_features": num_features,
                "superposition_ratio": num_features / 4.0,
                "seed": seed,
                "mean_abs_cosine": model.geometry_summary(batch_size=4096, seed=seed + 500)["mean_abs_cosine"],
                "top1_eigen_share": top_eigen_share(hidden, top_k=1),
            }
            row.update(norm_statistics(model))
            superposition_rows.append(row)

    super_agg = aggregate_rows(
        superposition_rows,
        group_keys=["superposition_ratio"],
        value_keys=["mean_abs_cosine", "alive_count", "top1_eigen_share"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_line_with_band(
        axes[0],
        [row["superposition_ratio"] for row in super_agg],
        [row["mean_abs_cosine_mean"] for row in super_agg],
        [row["mean_abs_cosine_std"] for row in super_agg],
        label="mean abs cosine",
        color="#2b7bba",
    )
    axes[0].set_xlabel("F / d")
    axes[0].set_ylabel("mean abs cosine")
    axes[0].set_title("Superposition increases overlap")

    plot_line_with_band(
        axes[1],
        [row["superposition_ratio"] for row in super_agg],
        [row["alive_count_mean"] for row in super_agg],
        [row["alive_count_std"] for row in super_agg],
        label="alive embeddings",
        color="#c44e52",
    )
    axes[1].set_xlabel("F / d")
    axes[1].set_ylabel("count")
    axes[1].set_title("More features survive under stronger superposition")

    plot_line_with_band(
        axes[2],
        [row["superposition_ratio"] for row in super_agg],
        [row["top1_eigen_share_mean"] for row in super_agg],
        [row["top1_eigen_share_std"] for row in super_agg],
        label="top eig share",
        color="#55a868",
    )
    axes[2].set_xlabel("F / d")
    axes[2].set_ylabel("share")
    axes[2].set_title("Hidden anisotropy under stronger crowding")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(part_dir / "superposition_summary.png", bbox_inches="tight")
    plt.close(fig)

    dynamics_config = ToyModelConfig(num_features=24, hidden_dim=2, alpha=1.2, expected_l0=2.5)
    dynamics_train = ToyTrainConfig(
        steps=2000,
        batch_size=512,
        learning_rate=0.025,
        log_every=50,
        snapshot_steps=(50, 200, 2000),
    )
    dynamics_model = get_toy_model(dynamics_config, dynamics_train, seed=0)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.95])
    snapshot_steps = [0, 50, 200, 2000]
    for idx, step in enumerate(snapshot_steps):
        ax = fig.add_subplot(gs[0, idx])
        snapshot = dynamics_model.snapshots[step]
        temp_model = ToyModel(
            config=dynamics_model.config,
            probabilities=dynamics_model.probabilities,
            weights=snapshot,
            bias=dynamics_model.bias,
            history=dynamics_model.history,
            snapshots={},
        )
        plot_embedding_points(ax, temp_model, title=f"step {step}")

    loss_ax = fig.add_subplot(gs[1, :2])
    history_steps = [row["step"] for row in dynamics_model.history]
    loss_ax.plot(history_steps, [row["loss"] for row in dynamics_model.history], color="#2b7bba", lw=2)
    loss_ax.set_title("Toy model loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("loss")

    cosine_ax = fig.add_subplot(gs[1, 2:])
    cosine_ax.plot(
        history_steps,
        [row["mean_abs_cosine"] for row in dynamics_model.history],
        color="#c44e52",
        lw=2,
    )
    cosine_ax.set_title("Embedding overlap during training")
    cosine_ax.set_xlabel("step")
    cosine_ax.set_ylabel("mean abs cosine")
    fig.tight_layout()
    fig.savefig(part_dir / "training_dynamics.png", bbox_inches="tight")
    plt.close(fig)

    write_csv(RESULTS_DIR / "part1_alpha_rows.csv", alpha_rows)
    write_csv(RESULTS_DIR / "part1_superposition_rows.csv", superposition_rows)
    save_summary(
        RESULTS_DIR / "part1_summary.json",
        {
            "alpha_config": config_to_dict(ToyModelConfig(alpha=1.0, **alpha_config_template)),
            "alpha_train_config": config_to_dict(alpha_train),
            "superposition_train_config": config_to_dict(superposition_train),
            "alpha_aggregates": alpha_agg,
            "superposition_aggregates": super_agg,
        },
    )
    return {
        "alpha_rows": alpha_rows,
        "superposition_rows": superposition_rows,
        "alpha_aggregates": alpha_agg,
        "superposition_aggregates": super_agg,
    }


def run_part2() -> dict[str, Any]:
    part_dir = FIGURES_DIR / "part2"
    part_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    baseline_config = ToyModelConfig(num_features=24, hidden_dim=4, alpha=1.2, expected_l0=2.5)
    baseline_train = ToyTrainConfig(steps=1500, batch_size=512, learning_rate=0.025, log_every=100)
    sae_train = SAETrainConfig(steps=1200, batch_size=512, learning_rate=0.01, log_every=50)
    latent_dims = [8, 12, 24, 48]
    l1_values = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    heatmap_rows: list[dict[str, Any]] = []

    for toy_seed in seeds:
        model = get_toy_model(baseline_config, baseline_train, seed=toy_seed)
        for latent_dim in latent_dims:
            for l1_coeff in l1_values:
                sae = get_sae(
                    model,
                    toy_seed=toy_seed,
                    sae_config=SAEConfig(latent_dim=latent_dim, l1_coeff=l1_coeff),
                    train_config=sae_train,
                    seed=toy_seed,
                )
                metrics = evaluate_sae(model, sae, batch_size=4096, seed=200 + toy_seed)
                heatmap_rows.append(
                    {
                        "toy_seed": toy_seed,
                        "latent_dim": latent_dim,
                        "l1_coeff": l1_coeff,
                        "ev": metrics["ev"],
                        "latent_density": metrics["latent_density"],
                    }
                )

    heatmap_agg = aggregate_rows(
        heatmap_rows,
        group_keys=["latent_dim", "l1_coeff"],
        value_keys=["ev", "latent_density"],
    )
    value_lookup = {
        (row["latent_dim"], row["l1_coeff"]): row["ev_mean"]
        for row in heatmap_agg
    }
    heatmap = np.array(
        [[value_lookup[(latent_dim, l1_coeff)] for latent_dim in latent_dims] for l1_coeff in l1_values],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(heatmap, cmap="magma", aspect="auto", vmin=float(heatmap.min()), vmax=float(heatmap.max()))
    ax.set_xticks(range(len(latent_dims)), labels=[str(item) for item in latent_dims])
    ax.set_yticks(range(len(l1_values)), labels=[str(item) for item in l1_values])
    ax.set_xlabel("SAE latent dim")
    ax.set_ylabel("l1 coefficient")
    ax.set_title("Explained variance on the baseline toy model")
    for row_idx in range(len(l1_values)):
        for col_idx in range(len(latent_dims)):
            ax.text(col_idx, row_idx, f"{heatmap[row_idx, col_idx]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(part_dir / "ev_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    curve_l1_values = [0.01, 0.03, 0.1, 0.3]
    curve_model = get_toy_model(baseline_config, baseline_train, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#2b7bba", "#55a868", "#c44e52", "#8172b2"]
    for l1_coeff, color in zip(curve_l1_values, colors, strict=True):
        sae = get_sae(
            curve_model,
            toy_seed=0,
            sae_config=SAEConfig(latent_dim=24, l1_coeff=l1_coeff),
            train_config=sae_train,
            seed=0,
        )
        steps = [row["step"] for row in sae.history]
        axes[0].plot(steps, [row["ev"] for row in sae.history], lw=2, color=color, label=f"l1={l1_coeff}")
        axes[1].plot(steps, [row["active_fraction"] for row in sae.history], lw=2, color=color, label=f"l1={l1_coeff}")
    axes[0].set_title("EV during SAE training")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("EV")
    axes[1].set_title("Latent density during SAE training")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("fraction of active latents")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(part_dir / "training_curves.png", bbox_inches="tight")
    plt.close(fig)

    model_rows: list[dict[str, Any]] = []
    expected_l0_values = [1.5, 2.5, 4.0]
    hidden_dims = [2, 4, 6]
    for expected_l0 in expected_l0_values:
        for hidden_dim in hidden_dims:
            config = ToyModelConfig(num_features=24, hidden_dim=hidden_dim, alpha=1.2, expected_l0=expected_l0)
            train_config = ToyTrainConfig(steps=1200, batch_size=512, learning_rate=0.025, log_every=100)
            for toy_seed in seeds:
                model = get_toy_model(config, train_config, seed=toy_seed)
                sae = get_sae(
                    model,
                    toy_seed=toy_seed,
                    sae_config=SAEConfig(latent_dim=24, l1_coeff=0.1),
                    train_config=sae_train,
                    seed=toy_seed,
                )
                metrics = evaluate_sae(model, sae, batch_size=4096, seed=400 + toy_seed)
                hidden, _, _ = model.sample_hidden(batch_size=4096, seed=600 + toy_seed)
                model_rows.append(
                    {
                        "expected_l0": expected_l0,
                        "hidden_dim": hidden_dim,
                        "toy_seed": toy_seed,
                        "superposition_ratio": 24 / hidden_dim,
                        "ev": metrics["ev"],
                        "mean_abs_cosine": metrics["mean_abs_cosine"],
                        "hidden_rank_pr": metrics["hidden_rank_pr"],
                        "top1_eigen_share": top_eigen_share(hidden, top_k=1),
                    }
                )

    model_agg = aggregate_rows(
        model_rows,
        group_keys=["expected_l0", "hidden_dim"],
        value_keys=["ev", "mean_abs_cosine", "hidden_rank_pr", "top1_eigen_share"],
    )
    ev_grid_lookup = {(row["expected_l0"], row["hidden_dim"]): row["ev_mean"] for row in model_agg}
    ev_grid = np.array(
        [[ev_grid_lookup[(expected_l0, hidden_dim)] for hidden_dim in hidden_dims] for expected_l0 in expected_l0_values],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    image = ax.imshow(ev_grid, cmap="viridis", aspect="auto", vmin=float(ev_grid.min()), vmax=float(ev_grid.max()))
    ax.set_xticks(range(len(hidden_dims)), labels=[str(item) for item in hidden_dims])
    ax.set_yticks(range(len(expected_l0_values)), labels=[str(item) for item in expected_l0_values])
    ax.set_xlabel("hidden dim d")
    ax.set_ylabel("expected active features")
    ax.set_title("EV as a function of the toy model parameters")
    for row_idx in range(len(expected_l0_values)):
        for col_idx in range(len(hidden_dims)):
            ax.text(col_idx, row_idx, f"{ev_grid[row_idx, col_idx]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(part_dir / "ev_model_grid.png", bbox_inches="tight")
    plt.close(fig)

    mean_abs_cosine = np.array([row["mean_abs_cosine"] for row in model_rows], dtype=np.float64)
    hidden_rank_pr = np.array([row["hidden_rank_pr"] for row in model_rows], dtype=np.float64)
    ev_values = np.array([row["ev"] for row in model_rows], dtype=np.float64)
    corr_cos = float(np.corrcoef(mean_abs_cosine, ev_values)[0, 1])
    corr_rank = float(np.corrcoef(hidden_rank_pr, ev_values)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(mean_abs_cosine, ev_values, alpha=0.8, color="#2b7bba")
    axes[0].set_xlabel("mean abs cosine of W")
    axes[0].set_ylabel("EV")
    axes[0].set_title(f"EV vs embedding overlap, corr={corr_cos:.2f}")
    axes[1].scatter(hidden_rank_pr, ev_values, alpha=0.8, color="#c44e52")
    axes[1].set_xlabel("hidden participation rank")
    axes[1].set_ylabel("EV")
    axes[1].set_title(f"EV vs hidden rank, corr={corr_rank:.2f}")
    fig.tight_layout()
    fig.savefig(part_dir / "ev_vs_geometry.png", bbox_inches="tight")
    plt.close(fig)

    write_csv(RESULTS_DIR / "part2_heatmap_rows.csv", heatmap_rows)
    write_csv(RESULTS_DIR / "part2_model_rows.csv", model_rows)
    save_summary(
        RESULTS_DIR / "part2_summary.json",
        {
            "baseline_config": config_to_dict(baseline_config),
            "baseline_train": config_to_dict(baseline_train),
            "sae_train": config_to_dict(sae_train),
            "heatmap_aggregates": heatmap_agg,
            "model_aggregates": model_agg,
            "correlations": {
                "ev_vs_mean_abs_cosine": corr_cos,
                "ev_vs_hidden_rank_pr": corr_rank,
            },
        },
    )
    return {
        "heatmap_rows": heatmap_rows,
        "model_rows": model_rows,
        "heatmap_aggregates": heatmap_agg,
        "model_aggregates": model_agg,
        "correlations": {
            "ev_vs_mean_abs_cosine": corr_cos,
            "ev_vs_hidden_rank_pr": corr_rank,
        },
    }


def run_part3() -> dict[str, Any]:
    part_dir = FIGURES_DIR / "part3"
    part_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    baseline_config = ToyModelConfig(num_features=24, hidden_dim=4, alpha=1.2, expected_l0=2.5)
    baseline_train = ToyTrainConfig(steps=1500, batch_size=512, learning_rate=0.025, log_every=100)
    baseline_model = get_toy_model(baseline_config, baseline_train, seed=0)
    sae_train = SAETrainConfig(steps=1200, batch_size=512, learning_rate=0.01, log_every=100)
    l1_values = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    recovery_rows: list[dict[str, Any]] = []
    scatter_payloads: dict[float, dict[str, Any]] = {}

    for toy_seed in seeds:
        model = get_toy_model(baseline_config, baseline_train, seed=toy_seed)
        for l1_coeff in l1_values:
            sae = get_sae(
                model,
                toy_seed=toy_seed,
                sae_config=SAEConfig(latent_dim=48, l1_coeff=l1_coeff),
                train_config=sae_train,
                seed=toy_seed,
            )
            recovered = recovery_metrics(model, sae, batch_size=5000, seed=700 + toy_seed)
            recovery_rows.append(
                {
                    "toy_seed": toy_seed,
                    "l1_coeff": l1_coeff,
                    "ev": recovered["ev"],
                    "embedding_mean_cosine": recovered["embedding_mean_cosine"],
                    "embedding_share_above_0_9": recovered["embedding_share_above_0_9"],
                    "frequency_mae": recovered["frequency_mae"],
                    "frequency_pearson": recovered["frequency_pearson"],
                    "mean_f1": recovered["mean_f1"],
                }
            )
            if toy_seed == 0 and l1_coeff in {0.0, 0.03, 0.3}:
                scatter_payloads[l1_coeff] = recovered

    recovery_agg = aggregate_rows(
        recovery_rows,
        group_keys=["l1_coeff"],
        value_keys=["ev", "embedding_mean_cosine", "frequency_mae", "mean_f1"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_line_with_band(
        axes[0],
        [row["l1_coeff"] for row in recovery_agg],
        [row["ev_mean"] for row in recovery_agg],
        [row["ev_std"] for row in recovery_agg],
        label="EV",
        color="#2b7bba",
    )
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("l1 coefficient")
    axes[0].set_ylabel("EV")
    axes[0].set_title("Reconstruction quality")

    plot_line_with_band(
        axes[1],
        [row["l1_coeff"] for row in recovery_agg],
        [row["embedding_mean_cosine_mean"] for row in recovery_agg],
        [row["embedding_mean_cosine_std"] for row in recovery_agg],
        label="decoder vs true embeddings",
        color="#55a868",
    )
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("l1 coefficient")
    axes[1].set_ylabel("matched cosine")
    axes[1].set_title("Embedding recovery")

    plot_line_with_band(
        axes[2],
        [row["l1_coeff"] for row in recovery_agg],
        [row["frequency_mae_mean"] for row in recovery_agg],
        [row["frequency_mae_std"] for row in recovery_agg],
        label="frequency MAE",
        color="#c44e52",
    )
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set_xlabel("l1 coefficient")
    axes[2].set_ylabel("MAE")
    axes[2].set_title("Frequency recovery")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(part_dir / "recovery_vs_l1.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, l1_coeff in zip(axes, [0.0, 0.03, 0.3], strict=True):
        payload = scatter_payloads[l1_coeff]
        true_frequency = np.asarray(payload["true_frequency"], dtype=np.float64)
        recovered_frequency = np.asarray(payload["recovered_frequency"], dtype=np.float64)
        ax.scatter(true_frequency, recovered_frequency, color="#2b7bba", alpha=0.8)
        diagonal = np.linspace(true_frequency.min(), true_frequency.max(), 100)
        ax.plot(diagonal, diagonal, color="#c44e52", lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("true frequency")
        ax.set_ylabel("recovered frequency")
        ax.set_title(f"l1={l1_coeff}")
    fig.tight_layout()
    fig.savefig(part_dir / "frequency_scatter.png", bbox_inches="tight")
    plt.close(fig)

    model_rows: list[dict[str, Any]] = []
    expected_l0_values = [1.5, 2.5, 4.0]
    hidden_dims = [2, 4, 6]
    model_train = ToyTrainConfig(steps=1200, batch_size=512, learning_rate=0.025, log_every=100)
    for expected_l0 in expected_l0_values:
        for hidden_dim in hidden_dims:
            config = ToyModelConfig(num_features=24, hidden_dim=hidden_dim, alpha=1.2, expected_l0=expected_l0)
            for toy_seed in seeds:
                model = get_toy_model(config, model_train, seed=toy_seed)
                sae = get_sae(
                    model,
                    toy_seed=toy_seed,
                    sae_config=SAEConfig(latent_dim=48, l1_coeff=0.03),
                    train_config=sae_train,
                    seed=toy_seed,
                )
                recovered = recovery_metrics(model, sae, batch_size=5000, seed=900 + toy_seed)
                model_rows.append(
                    {
                        "expected_l0": expected_l0,
                        "hidden_dim": hidden_dim,
                        "toy_seed": toy_seed,
                        "superposition_ratio": 24 / hidden_dim,
                        "ev": recovered["ev"],
                        "embedding_mean_cosine": recovered["embedding_mean_cosine"],
                        "frequency_mae": recovered["frequency_mae"],
                        "frequency_pearson": recovered["frequency_pearson"],
                        "mean_f1": recovered["mean_f1"],
                    }
                )

    model_agg = aggregate_rows(
        model_rows,
        group_keys=["expected_l0", "hidden_dim"],
        value_keys=["embedding_mean_cosine", "frequency_mae", "mean_f1", "ev"],
    )

    def make_grid(metric: str) -> np.ndarray:
        lookup = {(row["expected_l0"], row["hidden_dim"]): row[f"{metric}_mean"] for row in model_agg}
        return np.array(
            [[lookup[(expected_l0, hidden_dim)] for hidden_dim in hidden_dims] for expected_l0 in expected_l0_values],
            dtype=np.float64,
        )

    mae_grid = make_grid("frequency_mae")
    f1_grid = make_grid("mean_f1")
    cosine_grid = make_grid("embedding_mean_cosine")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, grid, title, cmap in [
        (axes[0], cosine_grid, "Embedding cosine", "viridis"),
        (axes[1], mae_grid, "Frequency MAE", "magma_r"),
        (axes[2], f1_grid, "Binary alignment F1", "viridis"),
    ]:
        image = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(hidden_dims)), labels=[str(item) for item in hidden_dims])
        ax.set_yticks(range(len(expected_l0_values)), labels=[str(item) for item in expected_l0_values])
        ax.set_xlabel("hidden dim d")
        ax.set_ylabel("expected active features")
        ax.set_title(title)
        for row_idx in range(len(expected_l0_values)):
            for col_idx in range(len(hidden_dims)):
                ax.text(col_idx, row_idx, f"{grid[row_idx, col_idx]:.3f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(part_dir / "recovery_model_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    write_csv(RESULTS_DIR / "part3_recovery_rows.csv", recovery_rows)
    write_csv(RESULTS_DIR / "part3_model_rows.csv", model_rows)
    save_summary(
        RESULTS_DIR / "part3_summary.json",
        {
            "baseline_config": config_to_dict(baseline_config),
            "baseline_train": config_to_dict(baseline_train),
            "sae_train": config_to_dict(sae_train),
            "recovery_aggregates": recovery_agg,
            "model_aggregates": model_agg,
        },
    )
    return {
        "recovery_rows": recovery_rows,
        "model_rows": model_rows,
        "recovery_aggregates": recovery_agg,
        "model_aggregates": model_agg,
    }


def main() -> None:
    ensure_dirs()
    set_style()
    part1 = run_part1()
    part2 = run_part2()
    part3 = run_part3()
    save_summary(
        RESULTS_DIR / "summary.json",
        {
            "part1": {
                "alpha_aggregates": part1["alpha_aggregates"],
                "superposition_aggregates": part1["superposition_aggregates"],
            },
            "part2": {
                "heatmap_aggregates": part2["heatmap_aggregates"],
                "model_aggregates": part2["model_aggregates"],
                "correlations": part2["correlations"],
            },
            "part3": {
                "recovery_aggregates": part3["recovery_aggregates"],
                "model_aggregates": part3["model_aggregates"],
            },
        },
    )


if __name__ == "__main__":
    main()
