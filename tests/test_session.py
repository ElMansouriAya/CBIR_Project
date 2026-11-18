"""Tests de backend/services/session.py (Tâche 5)."""
import numpy as np
import pytest

from backend import cbir_config as config
from backend.services.normalization import fit_normalizer, transform
from backend.services.session import SearchEngine


def make_world(seed=3, n_cat=5, per_cat=60, n_info=6, n_noise=24):
    """Dimensions informatives groupées par catégorie, noyées dans du bruit."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_cat, n_info))
    informative = np.vstack(
        [c + rng.normal(0, 0.15, size=(per_cat, n_info)) for c in centers]
    )
    noise = rng.normal(0, 1.0, size=(n_cat * per_cat, n_noise))
    X = np.hstack([informative, noise])
    categories = [f"cat{c}" for c in np.repeat(np.arange(n_cat), per_cat)]
    ids = list(range(100, 100 + len(X)))            # image_id différent de l'index
    Z = transform(X, fit_normalizer(X))
    return Z, ids, categories


@pytest.fixture
def world():
    return make_world()


@pytest.fixture
def engine(world):
    Z, ids, cats = world
    return SearchEngine(Z, ids, cats)


def relevant_of(response, category):
    return [r["image_id"] for r in response["results"] if r["category"] == category]


# --- Flux de base ------------------------------------------------------------
def test_start_returns_contract_fields(engine, world):
    _, ids, _ = world
    resp = engine.start(query_image_id=ids[0])
    assert isinstance(resp["session_id"], str)
    assert resp["iteration"] == 1
    assert len(resp["results"]) == config.SCOPE
    assert set(resp["results"][0]) == {"image_id", "distance", "category"}
    d = [r["distance"] for r in resp["results"]]
    assert d == sorted(d)
    assert "feedback" not in resp


def test_query_in_database_is_first_unless_excluded(engine, world):
    _, ids, _ = world
    assert engine.start(query_image_id=ids[7])["results"][0]["image_id"] == ids[7]
    excluded = engine.start(query_image_id=ids[7], exclude_query=True)
    assert ids[7] not in [r["image_id"] for r in excluded["results"]]


def test_external_query_vector(engine, world):
    Z, _, _ = world
    resp = engine.start(query_vector=Z[0] + 0.01)
    assert resp["iteration"] == 1 and len(resp["results"]) == config.SCOPE


def test_feedback_increments_iteration_and_changes_results(engine, world):
    _, ids, cats = world
    s = engine.start(query_image_id=ids[0], exclude_query=True)
    rel = relevant_of(s, cats[0])
    assert len(rel) >= 2
    s2 = engine.feedback(s["session_id"], rel)
    assert s2["iteration"] == 2
    assert s2["session_id"] == s["session_id"]
    assert s2["feedback"]["mode"] == "full"
    assert [r["image_id"] for r in s2["results"]] != [r["image_id"] for r in s["results"]]


def test_empty_feedback_changes_nothing_but_the_iteration(engine, world):
    _, ids, _ = world
    s = engine.start(query_image_id=ids[0], exclude_query=True)
    s2 = engine.feedback(s["session_id"], [])
    assert s2["iteration"] == 2
    assert s2["feedback"]["mode"] == "no_feedback"
    assert [r["image_id"] for r in s2["results"]] == [r["image_id"] for r in s["results"]]


def test_block_weights_follow_config_blocks(world):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, config.N_DIMS))
    ids = list(range(60))
    cats = ["a"] * 30 + ["b"] * 30
    eng = SearchEngine(X, ids, cats)
    s = eng.start(query_image_id=0)
    assert set(s["block_weights"]) == set(config.BLOCKS)
    assert all(v == pytest.approx(1.0) for v in s["block_weights"].values())


def test_informative_dims_get_higher_weight_after_feedback(engine, world):
    _, ids, cats = world
    s = engine.start(query_image_id=ids[0], exclude_query=True)
    s2 = engine.feedback(s["session_id"], relevant_of(s, cats[0]))
    w = engine.get_session(s["session_id"]).weights
    assert w[:6].mean() > w[6:].mean()                    # 6 dims informatives
    assert w.mean() == pytest.approx(1.0)
    assert s2["block_weights"] is None                    # d != 172 : pas de blocs


# --- Erreurs -----------------------------------------------------------------
def test_errors(engine, world):
    _, ids, cats = world
    with pytest.raises(ValueError):
        engine.start()                                              # ni id ni vecteur
    with pytest.raises(ValueError):
        engine.start(query_image_id=ids[0], query_vector=np.zeros(30))
    with pytest.raises(ValueError):
        engine.start(query_image_id=999999)                         # id inconnu
    with pytest.raises(ValueError):
        engine.start(query_image_id=ids[0], method="autre")
    with pytest.raises(ValueError):
        engine.start(query_vector=np.zeros(3))                      # mauvaise dimension
    with pytest.raises(KeyError):
        engine.feedback("inexistante", [])

    s = engine.start(query_image_id=ids[0], exclude_query=True)
    displayed = {r["image_id"] for r in s["results"]}
    not_displayed = next(i for i in ids if i not in displayed)
    with pytest.raises(ValueError):
        engine.feedback(s["session_id"], [not_displayed])           # jamais affichée


def test_constructor_validations():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        SearchEngine(X, [1, 2], ["a", "b", "c"])                    # longueurs
    with pytest.raises(ValueError):
        SearchEngine(X, [1, 1, 2], ["a", "b", "c"])                 # ids dupliqués


def test_max_iterations(world):
    Z, ids, cats = world
    eng = SearchEngine(Z, ids, cats, max_iterations=2)
    s = eng.start(query_image_id=ids[0], exclude_query=True)
    s = eng.feedback(s["session_id"], relevant_of(s, cats[0]))
    assert s["iteration"] == 2
    with pytest.raises(ValueError):
        eng.feedback(s["session_id"], [])


def test_oldest_session_is_evicted(world):
    Z, ids, cats = world
    eng = SearchEngine(Z, ids, cats, max_sessions=2)
    first = eng.start(query_image_id=ids[0])["session_id"]
    eng.start(query_image_id=ids[1])
    eng.start(query_image_id=ids[2])
    with pytest.raises(KeyError):
        eng.get_session(first)


# --- Comparaison des deux méthodes -----------------------------------------
def _mean_precision_by_iteration(world, method, n_iter=5, step=3):
    Z, ids, cats = world
    cats_arr = np.array(cats)
    eng = SearchEngine(Z, ids, cats, max_iterations=n_iter)
    precisions = {k: [] for k in range(1, n_iter + 1)}
    for i in range(0, len(Z), step):
        s = eng.start(query_image_id=ids[i], method=method, exclude_query=True)
        for k in range(1, n_iter + 1):
            hits = [r["category"] == cats_arr[i] for r in s["results"]]
            precisions[k].append(np.mean(hits))
            if k < n_iter:
                s = eng.feedback(s["session_id"], relevant_of(s, cats_arr[i]))
    return [float(np.mean(precisions[k])) for k in range(1, n_iter + 1)]


def test_both_methods_agree_on_the_first_feedback(world):
    """À l'itération 2, les deux méthodes utilisent exactement les mêmes images."""
    a = _mean_precision_by_iteration(world, "per_iteration", n_iter=2)
    b = _mean_precision_by_iteration(world, "cumulative", n_iter=2)
    assert a == pytest.approx(b)


def test_feedback_improves_precision_for_both_methods(world):
    for method in config.METHODS:
        p = _mean_precision_by_iteration(world, method)
        assert p[1] > p[0]


def test_cumulative_is_more_stable_than_per_iteration(world):
    """Quand toutes les images affichées sont pertinentes, la méthode par
    itération perd ses poids (sigma_Nr == sigma_rel) : la précision oscille."""
    per_it = _mean_precision_by_iteration(world, "per_iteration")
    cumul = _mean_precision_by_iteration(world, "cumulative")
    assert np.mean(cumul[2:]) > np.mean(per_it[2:])
    assert min(cumul[1:]) >= min(per_it[1:])
