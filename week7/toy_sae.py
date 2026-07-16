from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


Array = np.ndarray


@dataclass(frozen=True)
class ToyModelConfig:
    num_features: int
    hidden_dim: int
    alpha: float
    expected_l0: float
    max_prob: float = 0.95


@dataclass(frozen=True)
class ToyTrainConfig:
    steps: int = 3500
    batch_size: int = 1024
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    log_every: int = 50
    snapshot_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class SAEConfig:
    latent_dim: int
    l1_coeff: float
    decoder_row_norm: bool = True


@dataclass(frozen=True)
class SAETrainConfig:
    steps: int = 3000
    batch_size: int = 1024
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    log_every: int = 50


def relu(x: Array) -> Array:
    return np.maximum(x, 0.0)


def normalize_rows(matrix: Array, eps: float = 1e-8) -> Array:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def participation_ratio(values: Array, eps: float = 1e-12) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[vals > eps]
    if vals.size == 0:
        return 0.0
    return float(vals.sum() ** 2 / np.square(vals).sum())


def power_law_probabilities(
    num_features: int,
    alpha: float,
    expected_l0: float,
    max_prob: float = 0.95,
) -> Array:
    ranks = np.arange(1, num_features + 1, dtype=np.float64)
    base = ranks ** (-alpha)
    scale = expected_l0 / base.sum()
    probs = np.clip(scale * base, 1e-6, max_prob)
    return probs


def sample_features(probabilities: Array, batch_size: int, rng: np.random.Generator) -> tuple[Array, Array]:
    mask = (rng.random((batch_size, probabilities.shape[0])) < probabilities).astype(np.float64)
    magnitudes = rng.random((batch_size, probabilities.shape[0]))
    features = mask * magnitudes
    return features, mask


class Adam:
    def __init__(
        self,
        params: dict[str, Array],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_num = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: dict[str, Array], grads: dict[str, Array]) -> None:
        self.step_num += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        for name, value in params.items():
            grad = grads[name]
            self.m[name] = b1 * self.m[name] + (1.0 - b1) * grad
            self.v[name] = b2 * self.v[name] + (1.0 - b2) * np.square(grad)
            m_hat = self.m[name] / (1.0 - b1 ** self.step_num)
            v_hat = self.v[name] / (1.0 - b2 ** self.step_num)
            value -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def embedding_metrics(weight_matrix: Array) -> dict[str, float]:
    normalized = normalize_rows(weight_matrix)
    cosine = normalized @ normalized.T
    off_diagonal = ~np.eye(cosine.shape[0], dtype=bool)
    off_values = cosine[off_diagonal]
    abs_off = np.abs(off_values)
    singular_values = np.linalg.svd(weight_matrix, compute_uv=False)
    return {
        "mean_abs_cosine": float(abs_off.mean()),
        "mean_max_cosine": float(np.mean(np.max(np.abs(cosine - np.eye(cosine.shape[0])), axis=1))),
        "embedding_rank_pr": participation_ratio(np.square(singular_values)),
        "mean_row_norm": float(np.linalg.norm(weight_matrix, axis=1).mean()),
    }


def hidden_metrics(hidden: Array) -> dict[str, float]:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    positive = eigvals[eigvals > 1e-10]
    condition = float(positive.max() / positive.min()) if positive.size > 1 else math.inf
    return {
        "hidden_rank_pr": participation_ratio(eigvals),
        "hidden_variance": float(eigvals.sum()),
        "hidden_condition": condition,
    }


def explained_variance(hidden: Array, reconstructed: Array) -> float:
    residual_var = float(np.var(hidden - reconstructed))
    hidden_var = float(np.var(hidden))
    if hidden_var <= 1e-12:
        return 0.0
    ratio = max(residual_var / hidden_var, 0.0)
    return float(1.0 - math.sqrt(ratio))


@dataclass
class ToyModel:
    config: ToyModelConfig
    probabilities: Array
    weights: Array
    bias: Array
    history: list[dict[str, float]]
    snapshots: dict[int, Array]

    def sample_features(self, batch_size: int, seed: int | None = None) -> tuple[Array, Array]:
        rng = np.random.default_rng(seed)
        return sample_features(self.probabilities, batch_size, rng)

    def sample_hidden(self, batch_size: int, seed: int | None = None) -> tuple[Array, Array, Array]:
        features, mask = self.sample_features(batch_size, seed=seed)
        hidden = features @ self.weights
        return hidden, features, mask

    def reconstruct_features(self, features: Array) -> Array:
        hidden = features @ self.weights
        return relu(hidden @ self.weights.T + self.bias)

    def geometry_summary(self, batch_size: int = 8192, seed: int = 0) -> dict[str, float]:
        hidden, _, _ = self.sample_hidden(batch_size=batch_size, seed=seed)
        summary = {
            "superposition_ratio": float(self.config.num_features / self.config.hidden_dim),
            "effective_expected_l0": float(self.probabilities.sum()),
        }
        summary.update(embedding_metrics(self.weights))
        summary.update(hidden_metrics(hidden))
        return summary


@dataclass
class SparseAutoencoder:
    config: SAEConfig
    encoder: Array
    encoder_bias: Array
    decoder: Array
    decoder_bias: Array
    history: list[dict[str, float]]

    def encode(self, hidden: Array) -> Array:
        return relu(hidden @ self.encoder + self.encoder_bias)

    def reconstruct(self, hidden: Array) -> tuple[Array, Array]:
        latents = self.encode(hidden)
        reconstructed = latents @ self.decoder + self.decoder_bias
        return latents, reconstructed


def train_toy_model(
    model_config: ToyModelConfig,
    train_config: ToyTrainConfig,
    seed: int,
) -> ToyModel:
    rng = np.random.default_rng(seed)
    probabilities = power_law_probabilities(
        num_features=model_config.num_features,
        alpha=model_config.alpha,
        expected_l0=model_config.expected_l0,
        max_prob=model_config.max_prob,
    )
    weights = rng.normal(
        loc=0.0,
        scale=0.25 / math.sqrt(model_config.hidden_dim),
        size=(model_config.num_features, model_config.hidden_dim),
    )
    bias = np.zeros(model_config.num_features, dtype=np.float64)
    params = {"weights": weights, "bias": bias}
    optimizer = Adam(params=params, learning_rate=train_config.learning_rate)
    history: list[dict[str, float]] = []
    snapshots: dict[int, Array] = {0: weights.copy()}

    for step in range(1, train_config.steps + 1):
        features, _ = sample_features(probabilities, train_config.batch_size, rng)
        hidden = features @ weights
        pre_activation = hidden @ weights.T + bias
        reconstructed = relu(pre_activation)
        residual = reconstructed - features
        loss = float(np.mean(np.sum(np.square(residual), axis=1)))

        grad_out = (2.0 / train_config.batch_size) * residual
        grad_pre = grad_out * (pre_activation > 0.0)
        grad_bias = grad_pre.sum(axis=0)
        grad_weights_decoder = grad_pre.T @ hidden
        grad_hidden = grad_pre @ weights
        grad_weights_encoder = features.T @ grad_hidden
        grad_weights = grad_weights_decoder + grad_weights_encoder + train_config.weight_decay * weights

        optimizer.step(
            params=params,
            grads={
                "weights": grad_weights,
                "bias": grad_bias,
            },
        )

        if step in train_config.snapshot_steps:
            snapshots[step] = weights.copy()

        if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
            geometry = embedding_metrics(weights)
            h_metrics = hidden_metrics(hidden)
            history.append(
                {
                    "step": float(step),
                    "loss": loss,
                    "mean_abs_cosine": geometry["mean_abs_cosine"],
                    "mean_max_cosine": geometry["mean_max_cosine"],
                    "hidden_rank_pr": h_metrics["hidden_rank_pr"],
                    "hidden_condition": h_metrics["hidden_condition"],
                }
            )

    return ToyModel(
        config=model_config,
        probabilities=probabilities,
        weights=weights.copy(),
        bias=bias.copy(),
        history=history,
        snapshots=snapshots,
    )


def train_sae(
    toy_model: ToyModel,
    sae_config: SAEConfig,
    train_config: SAETrainConfig,
    seed: int,
) -> SparseAutoencoder:
    rng = np.random.default_rng(seed)
    hidden_dim = toy_model.config.hidden_dim
    encoder = rng.normal(0.0, 0.2 / math.sqrt(hidden_dim), size=(hidden_dim, sae_config.latent_dim))
    encoder_bias = np.zeros(sae_config.latent_dim, dtype=np.float64)
    decoder = normalize_rows(
        rng.normal(0.0, 0.2 / math.sqrt(hidden_dim), size=(sae_config.latent_dim, hidden_dim))
    )
    decoder_bias = np.zeros(hidden_dim, dtype=np.float64)
    params = {
        "encoder": encoder,
        "encoder_bias": encoder_bias,
        "decoder": decoder,
        "decoder_bias": decoder_bias,
    }
    optimizer = Adam(params=params, learning_rate=train_config.learning_rate)
    history: list[dict[str, float]] = []

    for step in range(1, train_config.steps + 1):
        hidden, _, _ = toy_model.sample_hidden(batch_size=train_config.batch_size, seed=int(rng.integers(1 << 30)))
        pre_latent = hidden @ encoder + encoder_bias
        latents = relu(pre_latent)
        reconstructed = latents @ decoder + decoder_bias
        residual = reconstructed - hidden

        recon_loss = float(np.mean(np.sum(np.square(residual), axis=1)))
        sparsity_loss = float(sae_config.l1_coeff * np.mean(np.sum(latents, axis=1)))
        total_loss = recon_loss + sparsity_loss

        grad_recon = (2.0 / train_config.batch_size) * residual
        grad_decoder = latents.T @ grad_recon + train_config.weight_decay * decoder
        grad_decoder_bias = grad_recon.sum(axis=0)
        grad_latents = grad_recon @ decoder.T + (sae_config.l1_coeff / train_config.batch_size)
        grad_pre_latent = grad_latents * (pre_latent > 0.0)
        grad_encoder = hidden.T @ grad_pre_latent + train_config.weight_decay * encoder
        grad_encoder_bias = grad_pre_latent.sum(axis=0)

        optimizer.step(
            params=params,
            grads={
                "encoder": grad_encoder,
                "encoder_bias": grad_encoder_bias,
                "decoder": grad_decoder,
                "decoder_bias": grad_decoder_bias,
            },
        )

        if sae_config.decoder_row_norm:
            decoder[:] = normalize_rows(decoder)

        if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
            ev = explained_variance(hidden, reconstructed)
            history.append(
                {
                    "step": float(step),
                    "loss": total_loss,
                    "recon_loss": recon_loss,
                    "sparsity_loss": sparsity_loss,
                    "ev": ev,
                    "active_fraction": float(np.mean(latents > 1e-4)),
                }
            )

    return SparseAutoencoder(
        config=sae_config,
        encoder=encoder.copy(),
        encoder_bias=encoder_bias.copy(),
        decoder=decoder.copy(),
        decoder_bias=decoder_bias.copy(),
        history=history,
    )


def evaluate_sae(
    toy_model: ToyModel,
    sae: SparseAutoencoder,
    batch_size: int = 8192,
    seed: int = 0,
) -> dict[str, float]:
    hidden, _, _ = toy_model.sample_hidden(batch_size=batch_size, seed=seed)
    latents, reconstructed = sae.reconstruct(hidden)
    metrics = {
        "ev": explained_variance(hidden, reconstructed),
        "mse": float(np.mean(np.square(hidden - reconstructed))),
        "latent_density": float(np.mean(latents > 1e-4)),
        "latent_l1": float(np.mean(np.sum(latents, axis=1))),
    }
    metrics.update(hidden_metrics(hidden))
    metrics.update(embedding_metrics(toy_model.weights))
    return metrics


def match_embeddings(true_embeddings: Array, decoder_atoms: Array) -> dict[str, Any]:
    true_norm = normalize_rows(true_embeddings)
    atom_norm = normalize_rows(decoder_atoms)
    similarity = np.clip(true_norm @ atom_norm.T, -1.0, 1.0)
    row_ind, col_ind = linear_sum_assignment(1.0 - similarity)
    matched = similarity[row_ind, col_ind]
    return {
        "row_ind": row_ind,
        "col_ind": col_ind,
        "similarity": similarity,
        "matched_cosine": matched,
        "mean_cosine": float(matched.mean()) if matched.size else 0.0,
        "median_cosine": float(np.median(matched)) if matched.size else 0.0,
        "share_above_0_9": float(np.mean(matched > 0.9)) if matched.size else 0.0,
    }


def recovery_metrics(
    toy_model: ToyModel,
    sae: SparseAutoencoder,
    batch_size: int = 12000,
    seed: int = 0,
    activation_threshold: float = 1e-3,
) -> dict[str, Any]:
    hidden, _, true_mask = toy_model.sample_hidden(batch_size=batch_size, seed=seed)
    latents, reconstructed = sae.reconstruct(hidden)
    matching = match_embeddings(toy_model.weights, sae.decoder)

    row_ind = matching["row_ind"]
    col_ind = matching["col_ind"]
    true_freq = toy_model.probabilities[row_ind]
    recovered_freq = []
    alignment_matrix = []
    chosen_thresholds = []

    for true_index, latent_index in zip(row_ind, col_ind, strict=False):
        latent_values = latents[:, latent_index]
        target = true_mask[:, true_index] > 0.5
        positive_values = latent_values[latent_values > activation_threshold]
        if positive_values.size == 0:
            thresholds = np.array([activation_threshold], dtype=np.float64)
        else:
            quantiles = np.linspace(0.1, 0.9, 9)
            thresholds = np.unique(
                np.concatenate(
                    [
                        np.array([activation_threshold], dtype=np.float64),
                        np.quantile(positive_values, quantiles),
                    ]
                )
            )

        best_f1 = 0.0
        best_mask = latent_values > activation_threshold
        best_threshold = float(activation_threshold)
        for threshold in thresholds:
            mask = latent_values > threshold
            true_positive = np.logical_and(mask, target).sum()
            precision = true_positive / max(mask.sum(), 1)
            recall = true_positive / max(target.sum(), 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_mask = mask
                best_threshold = float(threshold)

        alignment_matrix.append(best_f1)
        chosen_thresholds.append(best_threshold)
        recovered_freq.append(float(best_mask.mean()))

    recovered_freq = np.asarray(recovered_freq, dtype=np.float64)
    freq_mae = float(np.mean(np.abs(recovered_freq - true_freq)))
    freq_rmse = float(np.sqrt(np.mean(np.square(recovered_freq - true_freq))))
    spearman = float(spearmanr(true_freq, recovered_freq).statistic)
    pearson = float(np.corrcoef(true_freq, recovered_freq)[0, 1]) if true_freq.size > 1 else 0.0

    return {
        "ev": explained_variance(hidden, reconstructed),
        "embedding_mean_cosine": matching["mean_cosine"],
        "embedding_median_cosine": matching["median_cosine"],
        "embedding_share_above_0_9": matching["share_above_0_9"],
        "frequency_mae": freq_mae,
        "frequency_rmse": freq_rmse,
        "frequency_spearman": spearman,
        "frequency_pearson": pearson,
        "mean_f1": float(np.mean(alignment_matrix)) if alignment_matrix else 0.0,
        "true_frequency": true_freq,
        "recovered_frequency": recovered_freq,
        "chosen_thresholds": np.asarray(chosen_thresholds, dtype=np.float64),
        "matched_cosine": matching["matched_cosine"],
        "matched_true_index": row_ind,
        "matched_latent_index": col_ind,
    }


def history_to_array(history: list[dict[str, float]], key: str) -> Array:
    return np.array([item[key] for item in history], dtype=np.float64)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    target.write_text(json.dumps(convert(payload), indent=2, ensure_ascii=False))


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)
