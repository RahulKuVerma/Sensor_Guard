
"""
SensorGuard: Monte Carlo Dropout uncertainty estimation.

Estimates predictive uncertainty using repeated stochastic forward passes.
The model weights are not updated.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    n_passes: int = 50,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    epsilon: float = 1e-7,
) -> dict[str, np.ndarray]:
    """
    Run Monte Carlo Dropout inference for binary classification.

    Parameters
    ----------
    model:
        Trained PyTorch model returning one logit per sample.
    X:
        Input windows with shape (samples, timesteps, sensors).
    n_passes:
        Number of stochastic forward passes per batch.
    batch_size:
        Number of samples processed in each batch.
    device:
        Device used for inference, such as "cpu".
    epsilon:
        Numerical safeguard for entropy calculations.

    Returns
    -------
    Dictionary containing:
        mean_probability
        predictive_variance
        predictive_entropy
        expected_entropy
        mutual_information
        pass_probabilities
    """
    if n_passes < 2:
        raise ValueError("n_passes must be at least 2.")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5.")

    if isinstance(X, torch.Tensor):
        X_tensor = X.detach().to(dtype=torch.float32)
    else:
        X_array = np.asarray(X, dtype=np.float32)

        if X_array.ndim != 3:
            raise ValueError(
                "X must have shape (samples, timesteps, sensors)."
            )

        X_tensor = torch.from_numpy(X_array)

    if X_tensor.ndim != 3:
        raise ValueError(
            "X must have shape (samples, timesteps, sensors)."
        )

    if X_tensor.shape[0] == 0:
        raise ValueError("X contains no samples.")

    if not torch.isfinite(X_tensor).all():
        raise ValueError("X contains NaN or infinite values.")

    device = torch.device(device)
    X_tensor = X_tensor.to(device)

    model.to(device)

    # Preserve every module's current training flag.
    original_states = {
        module: module.training for module in model.modules()
    }

    # Disable ordinary training behavior, then selectively enable
    # dropout in supported PyTorch modules.
    model.eval()

    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )

    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train(True)
        elif isinstance(module, nn.LSTM):
            # PyTorch LSTM's internal inter-layer dropout depends
            # on the LSTM module's own training flag.
            module.train(True)

    pass_results = []

    try:
        with torch.inference_mode():
            for start in range(0, len(X_tensor), batch_size):
                batch = X_tensor[start : start + batch_size]
                batch_passes = []

                for _ in range(n_passes):
                    logits = model(batch)

                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

                    # The SensorGuard baseline returns one logit
                    # per sample.
                    if logits.numel() != batch.shape[0]:
                        raise ValueError(
                            "Expected one binary-classification logit "
                            "per input sample."
                        )

                    probabilities = torch.sigmoid(
                        logits.reshape(-1)
                    )

                    batch_passes.append(
                        probabilities.cpu().numpy()
                    )

                # Shape: (passes, samples_in_batch)
                pass_results.append(np.stack(batch_passes, axis=0))

    finally:
        # Restore the exact previous training/evaluation flags.
        for module, was_training in original_states.items():
            module.training = was_training

    # Shape: (passes, all_samples)
    pass_probabilities = np.concatenate(pass_results, axis=1)

    mean_probability = pass_probabilities.mean(axis=0)
    predictive_variance = pass_probabilities.var(
        axis=0, ddof=1
    )

    clipped = np.clip(
        pass_probabilities, epsilon, 1.0 - epsilon
    )

    # Entropy of each stochastic probability prediction.
    individual_entropies = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )

    # Entropy of the mean predictive probability.
    mean_clipped = np.clip(
        mean_probability, epsilon, 1.0 - epsilon
    )

    predictive_entropy = -(
        mean_clipped * np.log(mean_clipped)
        + (1.0 - mean_clipped)
        * np.log(1.0 - mean_clipped)
    )

    # Average entropy across stochastic passes.
    expected_entropy = individual_entropies.mean(axis=0)

    # Approximate epistemic uncertainty:
    # predictive entropy minus expected entropy.
    mutual_information = np.maximum(
        predictive_entropy - expected_entropy, 0.0
    )

    return {
        "mean_probability": mean_probability,
        "predictive_variance": predictive_variance,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "pass_probabilities": pass_probabilities,
    }
