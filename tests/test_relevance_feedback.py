"""Tests de backend/services/relevance_feedback.py (Tâche 4)."""
import numpy as np
import pytest

from backend import cbir_config as config
from backend.services.normalization import fit_normalizer, transform
from backend.services.relevance_feedback import (
    compute_weights,
    feedback_step,
    initial_weights,
    raw_weights,
    stabilize_weights,
    update_query,
)
from backend.services.search import search

EPS = config.EPSILON


# --- Cas construit à la main -----------------------------------------------
# 4 images retournées en 2D. Les 2 pertinentes ont la dimension 0 constante.
#   dim 0 : retournées [0, 0, 4, 4] -> sigma_Nr = 2 ; pertinentes [0, 0] -> sigma_rel = 0
#   dim 1 : retournées [0, 2, 0, 2] -> sigma_Nr = 1 ; pertinentes [0, 2] -> sigma_rel = 1
RETRIEVED = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0]])
RELEVANT = RETRIEVED[:2]


def test_raw_weights_match_paper_formula_by_hand():
    w = raw_weights(RETRIEVED, RELEVANT)
    assert w[0] == pytest.approx((EPS + 2.0) / (EPS + 0.0))   # = 20001
    assert w[1] == pytest.approx(1.0)                          # (eps+1)/(eps+1)


def test_stable_dimension_gets_higher_weight():
    w = compute_weights(RETRIEVED, RELEVANT)
    assert w[0] > w[1]


def test_stabilized_weights_by_hand():
    """20001 écrêté à 10, 1 reste 1, moyenne 5.5 -> [10/5.5, 1/5.5]."""
    w = compute_weights(RETRIEVED, RELEVANT)
    assert w == pytest.approx([10 / 5.5, 1 / 5.5])


def test_weights_have_mean_one_and_bounded_ratio():
    rng = np.random.default_rng(0)
    retrieved = rng.normal(size=(25, 40))
    relevant = retrieved[:6] * rng.uniform(0.001, 1.0, size=40)  # dispersions variées
    w = compute_weights(retrieved, relevant)
    assert w.mean() == pytest.approx(1.0)
    # écrêtage [0.1, 10] puis division par une moyenne commune : rapport <= 100
    assert w.max() / w.min() <= config.WEIGHT_MAX / config.WEIGHT_MIN + 1e-9


def test_all_retrieved_relevant_gives_uniform_weights():
    """Si tout est pertinent, sigma_Nr == sigma_rel : tous les poids bruts valent 1."""
    X = np.random.default_rng(1).normal(size=(10, 5))
    assert np.allclose(raw_weights(X, X), 1.0)
    assert np.allclose(compute_weights(X, X), 1.0)


def test_constant_dimensions_get_neutral_weight():
    """Dimension constante partout : (eps+0)/(eps+0) = 1."""
    X = np.zeros((6, 3))
    X[:, 1] = np.arange(6)
    w = raw_weights(X, X[:3])
    assert w[0] == pytest.approx(1.0) and w[2] == pytest.approx(1.0)


def test_stabilize_weights_clips_then_normalizes():
    w = stabilize_weights(np.array([0.0001, 1.0, 1000.0]))
    clipped = np.array([0.1, 1.0, 10.0])
    assert w == pytest.approx(clipped / clipped.mean())


def test_update_query_is_the_mean_of_relevant_vectors():
    q = update_query(np.array([[1.0, 2.0], [3.0, 6.0]]))
    assert q.tolist() == [2.0, 4.0]


def test_initial_weights_are_ones():
    assert initial_weights(7).tolist() == [1.0] * 7


# --- Cas limites de feedback_step ------------------------------------------
@pytest.fixture
def small_world():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6))
    return X, np.arange(10), X[0].copy(), np.full(6, 1.5)


def test_no_relevant_keeps_query_and_weights(small_world):
    X, retrieved, q, w = small_world
    nq, nw, info = feedback_step(X, retrieved, [], q, w)
    assert info == {"mode": "no_feedback", "n_relevant": 0}
    assert np.array_equal(nq, q) and np.array_equal(nw, w)


def test_one_relevant_updates_query_only(small_world):
    X, retrieved, q, w = small_world
    nq, nw, info = feedback_step(X, retrieved, [3], q, w)
    assert info == {"mode": "query_only", "n_relevant": 1}
    assert np.array_equal(nq, X[3])
    assert np.array_equal(nw, w)           # poids précédents conservés


def test_several_relevant_updates_query_and_weights(small_world):
    X, retrieved, q, w = small_world
    nq, nw, info = feedback_step(X, retrieved, [1, 4, 7], q, w)
    assert info == {"mode": "full", "n_relevant": 3}
    assert np.allclose(nq, X[[1, 4, 7]].mean(axis=0))
    assert nw.mean() == pytest.approx(1.0)
    assert not np.allclose(nw, w)


def test_duplicate_relevant_ids_are_ignored(small_world):
    X, retrieved, q, w = small_world
    _, _, info = feedback_step(X, retrieved, [3, 3, 3], q, w)
    assert info["mode"] == "query_only"


def test_relevant_must_be_a_subset_of_retrieved(small_world):
    X, retrieved, q, w = small_world
    with pytest.raises(ValueError):
        feedback_step(X, retrieved, [1, 25], q, w)   # 25 n'a pas été affiché


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        raw_weights(np.zeros((5, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        raw_weights(np.zeros((5, 3)), np.zeros((1, 3)))   # un seul échantillon
    with pytest.raises(ValueError):
        update_query(np.zeros((0, 3)))


# --- Intégration avec la recherche -----------------------------------------
def _precision(labels, idx, label):
    return float((labels[idx] == label).mean())


def test_feedback_boosts_informative_dimensions_and_precision():
    """6 dimensions informatives (groupées par catégorie) noyées dans 24
    dimensions de bruit. La recherche simple est gênée par le bruit ; le
    feedback doit donner plus de poids aux dimensions informatives."""
    rng = np.random.default_rng(3)
    n_cat, per_cat, n_info, n_noise = 5, 60, 6, 24
    centers = rng.normal(0, 1.0, size=(n_cat, n_info))
    informative = np.vstack(
        [c + rng.normal(0, 0.15, size=(per_cat, n_info)) for c in centers]
    )
    noise = rng.normal(0, 1.0, size=(n_cat * per_cat, n_noise))
    X = np.hstack([informative, noise])
    labels = np.repeat(np.arange(n_cat), per_cat)
    Z = transform(X, fit_normalizer(X))

    before, after, info_w, noise_w = [], [], [], []
    for i in range(0, len(Z), 5):
        w0 = initial_weights(Z.shape[1])
        idx0, _ = search(Z[i], Z, scope=config.SCOPE, exclude_index=i, weights=w0)
        relevant = idx0[labels[idx0] == labels[i]]          # feedback oracle
        if len(relevant) < 2:
            continue
        q1, w1, _ = feedback_step(Z, idx0, relevant, Z[i], w0)
        idx1, _ = search(q1, Z, scope=config.SCOPE, exclude_index=i, weights=w1)
        before.append(_precision(labels, idx0, labels[i]))
        after.append(_precision(labels, idx1, labels[i]))
        info_w.append(w1[:n_info].mean())
        noise_w.append(w1[n_info:].mean())

    assert len(before) > 20
    assert np.mean(info_w) > np.mean(noise_w)               # poids : informatives > bruit
    assert np.mean(after) > np.mean(before)                 # le feedback aide en moyenne
