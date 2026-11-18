"""Tests de backend/services/normalization.py (Tâche 2)."""
import json

import numpy as np
import pytest

from backend import cbir_config as config
from backend.services.normalization import (
    fit_normalizer,
    load_normalizer,
    save_normalizer,
    transform,
)


@pytest.fixture
def X():
    """Matrice (200, d) avec des échelles très différentes par bloc."""
    rng = np.random.default_rng(0)
    X = rng.random((200, config.N_DIMS))
    X[:, 146:162] *= 1000.0   # GLCM : grandes valeurs
    X[:, 0:128] *= 0.01       # histogramme : petites valeurs
    return X


def test_mean_zero_std_one(X):
    norm = fit_normalizer(X)
    Z = transform(X, norm)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_constant_column_has_no_division_by_zero(X):
    X = X.copy()
    X[:, 10] = 3.14  # dimension constante
    norm = fit_normalizer(X)
    Z = transform(X, norm)
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, 10], 0.0)
    assert norm["constant_mask"][10]
    assert norm["constant_mask"].sum() == 1


def test_query_uses_same_parameters_as_database(X):
    """Un vecteur 1D de la base donne la même sortie que sa ligne dans la matrice."""
    norm = fit_normalizer(X)
    Z = transform(X, norm)
    z_query = transform(X[7], norm)
    assert z_query.shape == (config.N_DIMS,)
    assert np.allclose(z_query, Z[7])


def test_save_load_roundtrip(X, tmp_path):
    norm = fit_normalizer(X)
    path = tmp_path / "normalizer.npz"
    save_normalizer(norm, path)
    loaded = load_normalizer(path)
    assert np.array_equal(loaded["mean"], norm["mean"])
    assert np.array_equal(loaded["std"], norm["std"])
    assert np.array_equal(loaded["constant_mask"], norm["constant_mask"])
    assert np.allclose(transform(X, loaded), transform(X, norm))


def test_dimension_mismatch_raises(X):
    norm = fit_normalizer(X)
    with pytest.raises(ValueError):
        transform(np.zeros((3, config.N_DIMS + 1)), norm)


def test_nan_raises(X):
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_normalizer(X)
    norm = fit_normalizer(np.nan_to_num(X))
    bad = np.zeros(config.N_DIMS)
    bad[5] = np.inf
    with pytest.raises(ValueError):
        transform(bad, norm)


def test_normalization_balances_block_scales(X):
    """Après z-score, aucun bloc ne domine par son échelle numérique."""
    norm = fit_normalizer(X)
    Z = transform(X, norm)
    var_per_block = [
        Z[:, s:e].var(axis=0).mean() for s, e in config.BLOCKS.values()
    ]
    assert np.allclose(var_per_block, 1.0, atol=1e-9)


def test_synthetic_files_follow_contract():
    """Contrôle du contrat sur les données synthétiques si elles ont été générées."""
    folder = config.SYNTHETIC_DIR
    npy = folder / "descriptors_raw.npy"
    if not npy.exists():
        pytest.skip("Lancer d'abord : python scripts/make_synthetic_data.py")

    X = np.load(npy)
    with open(folder / "image_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)

    assert X.ndim == 2 and X.shape[1] == config.N_DIMS
    assert np.isfinite(X).all()
    assert len(mapping) == X.shape[0]
    assert all(m["index"] == i for i, m in enumerate(mapping))
