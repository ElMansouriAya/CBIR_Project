"""Données de test partagées par les tests de l'API et du chargeur."""
import numpy as np


def make_raw_world(seed=3, n_cat=5, per_cat=60, n_info=6, n_noise=24):
    """Matrice BRUTE : dimensions informatives groupées par catégorie + bruit.

    Retourne (X, image_ids, categories) ; image_id = index + 100.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_cat, n_info))
    informative = np.vstack(
        [c + rng.normal(0, 0.15, size=(per_cat, n_info)) for c in centers]
    )
    noise = rng.normal(0, 1.0, size=(n_cat * per_cat, n_noise))
    X = np.hstack([informative, noise])
    categories = [f"cat{c}" for c in np.repeat(np.arange(n_cat), per_cat)]
    ids = list(range(100, 100 + len(X)))
    return X, ids, categories
