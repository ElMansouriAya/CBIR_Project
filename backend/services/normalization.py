"""Normalisation z-score des vecteurs de descripteurs.

    z_i = (x_i - mu_i) / sigma_i

Les paramètres (mu, sigma) sont calculés UNE FOIS sur toute la base, sauvegardés,
puis réutilisés à l'identique pour chaque image requête.

Une dimension constante (sigma ~ 0) ne doit pas provoquer de division par zéro :
on lui attribue sigma = 1, ce qui donne z = 0 pour toutes les images (elle ne
contribue donc pas à la distance).
"""
from pathlib import Path

import numpy as np

from backend import cbir_config as config


def _check_matrix(X, name="X"):
    """Convertit en float64 et vérifie la forme (N, d) et l'absence de NaN/inf."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} doit être une matrice (N, d), reçu {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contient des NaN ou des inf")
    return X


def fit_normalizer(X):
    """Calcule moyenne et écart-type par dimension sur la base `X` (N, d).

    Retourne un dictionnaire :
        mean          : (d,) moyenne par dimension
        std           : (d,) écart-type utilisé (1.0 si la dimension est constante)
        constant_mask : (d,) True pour les dimensions constantes
    """
    X = _check_matrix(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    constant_mask = std < config.STD_MIN
    std_safe = np.where(constant_mask, 1.0, std)
    return {"mean": mean, "std": std_safe, "constant_mask": constant_mask}


def transform(X, normalizer):
    """Applique la normalisation. Accepte un vecteur (d,) ou une matrice (N, d).

    Une image requête est donc transformée avec exactement les mêmes
    paramètres que la base.
    """
    X = np.asarray(X, dtype=np.float64)
    single = X.ndim == 1
    X2 = X.reshape(1, -1) if single else X
    X2 = _check_matrix(X2)

    d = normalizer["mean"].shape[0]
    if X2.shape[1] != d:
        raise ValueError(f"Dimension attendue {d}, reçue {X2.shape[1]}")

    Z = (X2 - normalizer["mean"]) / normalizer["std"]
    return Z[0] if single else Z


def save_normalizer(normalizer, path):
    """Sauvegarde les paramètres dans un fichier .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=normalizer["mean"],
        std=normalizer["std"],
        constant_mask=normalizer["constant_mask"],
    )


def load_normalizer(path):
    """Recharge les paramètres sauvegardés par `save_normalizer`."""
    with np.load(Path(path)) as data:
        return {
            "mean": data["mean"],
            "std": data["std"],
            "constant_mask": data["constant_mask"],
        }
