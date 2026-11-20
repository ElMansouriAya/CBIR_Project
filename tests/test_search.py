"""Tests de backend/services/search.py (Tâche 3)."""
import numpy as np
import pytest

from backend import cbir_config as config
from backend.services.normalization import fit_normalizer, transform
from backend.services.search import euclidean_distances, search, weighted_distances


def test_ranking_on_hand_built_example():
    """Exemple du roadmap : distances 0.12, 0.35, 0.18, 0.80 -> A, C, B, D."""
    q = np.zeros(2)
    X = np.array([[0.12, 0], [0.35, 0], [0.18, 0], [0.80, 0]])  # A, B, C, D
    idx, dist = search(q, X, scope=4)
    assert idx.tolist() == [0, 2, 1, 3]
    assert np.allclose(dist, [0.12, 0.18, 0.35, 0.80])


def test_euclidean_distance_value():
    d = euclidean_distances(np.array([0.0, 0.0]), np.array([[3.0, 4.0]]))
    assert d[0] == pytest.approx(5.0)


def test_query_in_database_is_found_at_distance_zero():
    rng = np.random.default_rng(1)
    X = rng.random((50, config.N_DIMS))
    idx, dist = search(X[12], X, scope=5)
    assert idx[0] == 12
    assert dist[0] == pytest.approx(0.0)


def test_exclude_index_removes_the_query():
    rng = np.random.default_rng(1)
    X = rng.random((50, config.N_DIMS))
    idx, dist = search(X[12], X, scope=5, exclude_index=12)
    assert 12 not in idx
    assert dist[0] > 0


def test_distances_are_sorted_ascending():
    rng = np.random.default_rng(2)
    X = rng.random((100, 10))
    _, dist = search(rng.random(10), X, scope=25)
    assert (np.diff(dist) >= 0).all()


def test_scope_larger_than_database():
    X = np.random.default_rng(3).random((10, 4))
    idx, _ = search(X[0], X, scope=25)
    assert len(idx) == 10
    idx, _ = search(X[0], X, scope=25, exclude_index=0)
    assert len(idx) == 9


def test_ties_are_broken_by_row_order():
    X = np.array([[1.0, 0], [-1.0, 0], [1.0, 0]])  # trois points à distance 1
    idx, _ = search(np.zeros(2), X, scope=3)
    assert idx.tolist() == [0, 1, 2]


def test_weighted_distance_matches_paper_formula():
    """d_w = sqrt( sum w (q-x)^2 / sum w ), vérifié à la main."""
    q = np.zeros(2)
    X = np.array([[0.0, 2.0]])
    w = np.array([1.0, 3.0])
    # 3 * 2^2 / (1 + 3) = 3  ->  sqrt(3)
    assert weighted_distances(q, X, w)[0] == pytest.approx(np.sqrt(3.0))


def test_zero_weight_ignores_a_dimension():
    q = np.zeros(2)
    X = np.array([[100.0, 1.0], [0.0, 2.0]])
    idx, _ = search(q, X, scope=2, weights=np.array([0.0, 1.0]))
    assert idx.tolist() == [0, 1]  # la 1re dimension est ignorée


def test_uniform_weights_keep_the_same_ranking():
    rng = np.random.default_rng(4)
    X = rng.random((200, 20))
    q = rng.random(20)
    plain, _ = search(q, X, scope=25)
    weighted, _ = search(q, X, scope=25, weights=np.ones(20))
    assert plain.tolist() == weighted.tolist()


def test_invalid_inputs_raise():
    X = np.zeros((5, 3))
    with pytest.raises(ValueError):
        search(np.zeros(4), X)                       # mauvaise dimension
    with pytest.raises(ValueError):
        search(np.zeros(3), X, scope=0)              # scope invalide
    with pytest.raises(ValueError):
        search(np.zeros(3), X, exclude_index=5)      # index hors limites
    with pytest.raises(ValueError):
        search(np.array([np.nan, 0, 0]), X)          # NaN
    with pytest.raises(ValueError):
        search(np.zeros(3), X, weights=np.array([1.0, -1.0, 1.0]))  # poids négatif
    with pytest.raises(ValueError):
        search(np.zeros(3), X, weights=np.zeros(3))  # somme nulle


def test_search_finds_same_category_on_clustered_data():
    """Sur des groupes bien séparés, les voisins sont de la même catégorie."""
    rng = np.random.default_rng(5)
    n_cat, per_cat, d = 5, 40, config.N_DIMS
    centers = rng.random((n_cat, d))
    X = np.vstack([c + rng.normal(0, 0.15, (per_cat, d)) for c in centers])
    labels = np.repeat(np.arange(n_cat), per_cat)

    Z = transform(X, fit_normalizer(X))
    hits = []
    for i in range(0, len(X), 7):
        idx, _ = search(Z[i], Z, scope=25, exclude_index=i)
        hits.append((labels[idx] == labels[i]).mean())
    assert np.mean(hits) >= 0.9
