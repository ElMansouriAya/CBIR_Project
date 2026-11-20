"""Tests de backend/services/evaluation.py et de scripts/evaluate_retrieval.py (Tâche 7)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import make_raw_world

from backend import cbir_config as config
from backend.services import evaluation as ev
from backend.services.normalization import fit_normalizer, transform
from backend.services.relevance_feedback import initial_weights, update_query
from backend.services.search import search
from backend.services.session import SearchEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate_retrieval as script  # noqa: E402


@pytest.fixture(scope="module")
def world():
    """Données de difficulté moyenne : la précision n'est pas saturée."""
    X, ids, cats = make_raw_world(seed=5, n_cat=5, per_cat=50, n_info=6, n_noise=24)
    Z = transform(X, fit_normalizer(X))
    return Z, ids, np.array(cats)


# --- Métriques élémentaires --------------------------------------------------
def test_average_precision_by_hand():
    # pertinents aux rangs 1 et 3 : (1/1 + 2/3) / 2
    assert ev.average_precision([1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)
    assert ev.average_precision([1, 1, 1]) == pytest.approx(1.0)
    assert ev.average_precision([0, 0, 0]) == 0.0
    assert ev.average_precision([0, 0, 1]) == pytest.approx(1 / 3)


def test_mean_ci_by_hand():
    mean, ci = ev.mean_ci(np.array([1.0, 2.0, 3.0]))
    assert mean == pytest.approx(2.0)
    assert ci == pytest.approx(1.96 * 1.0 / np.sqrt(3))
    _, ci_single = ev.mean_ci(np.array([0.7]))
    assert ci_single == 0.0


def test_precision_cap_limited_by_small_categories():
    labels = np.array(["a"] * 10 + ["b"] * 40)
    assert ev.precision_cap(labels, [0], scope=25) == pytest.approx(9 / 25)
    assert ev.precision_cap(labels, [20], scope=25) == pytest.approx(1.0)


# --- Échantillonnage -------------------------------------------------------------
def test_sample_queries_is_stratified_and_reproducible():
    labels = np.repeat(["a", "b", "c", "d"], 50)
    q = ev.sample_queries(labels, 40, seed=1)
    assert len(q) == 40 and len(set(q)) == 40
    assert all((labels[q] == c).sum() == 10 for c in "abcd")
    assert np.array_equal(q, ev.sample_queries(labels, 40, seed=1))
    assert not np.array_equal(q, ev.sample_queries(labels, 40, seed=2))
    assert (np.diff(q) > 0).all()                            # triés


def test_sample_queries_fills_the_remainder_and_clips():
    labels = np.repeat(["a", "b", "c"], 10)
    assert len(ev.sample_queries(labels, 10, seed=0)) == 10   # 3 par catégorie + 1 au hasard
    assert len(ev.sample_queries(labels, 1000, seed=0)) == 30  # plafonné à N


# --- Boucle d'évaluation --------------------------------------------------------
def test_query_is_excluded_from_the_ranking(world):
    Z, _, labels = world
    order, rel = ev._rank(Z, labels, 7, Z[7], initial_weights(Z.shape[1]))
    assert 7 not in order
    assert len(order) == len(Z) - 1 and len(rel) == len(Z) - 1


def test_iteration_one_is_identical_for_all_methods(world):
    Z, _, labels = world
    res = ev.evaluate(Z, labels, [0, 60, 120], n_iter=3)
    base = [res["p_at_scope"][m][:, 0] for m in ev.EVAL_METHODS]
    assert np.array_equal(base[0], base[1]) and np.array_equal(base[0], base[2])
    base_ap = [res["ap"][m][:, 0] for m in ev.EVAL_METHODS]
    assert np.array_equal(base_ap[0], base_ap[1]) and np.array_equal(base_ap[0], base_ap[2])


def test_perfectly_separated_clusters_give_perfect_scores():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 0.01, (30, 5)), rng.normal(10, 0.01, (30, 5))])
    labels = np.array(["a"] * 30 + ["b"] * 30)
    res = ev.evaluate(Z, labels, [0, 40], n_iter=3, scope=10)
    for m in res["methods"]:
        assert np.allclose(res["p_at_scope"][m], 1.0)
        assert np.allclose(res["ap"][m], 1.0)


@pytest.mark.parametrize("method", ["per_iteration", "cumulative"])
def test_evaluation_matches_the_search_engine(world, method):
    """La boucle d'évaluation affiche EXACTEMENT les mêmes images que le moteur
    de sessions (celui que l'application utilise), à chaque itération."""
    Z, ids, labels = world
    eval_name = {"per_iteration": "type1", "cumulative": "type1_cumulative"}[method]
    engine = SearchEngine(Z, ids, list(labels), max_iterations=5)
    id_to_index = {image_id: k for k, image_id in enumerate(ids)}
    n_iter = 5

    for i in (0, 51, 103, 160, 222):
        trace = []
        ev.evaluate_query(Z, labels, i, [eval_name], n_iter, config.SCOPE, [1, 25], trace=trace)
        expected = {k: shown for _, k, shown in trace}

        resp = engine.start(query_image_id=ids[i], method=method, exclude_query=True)
        for k in range(1, n_iter + 1):
            got = [id_to_index[r["image_id"]] for r in resp["results"]]
            assert got == expected[k].tolist(), f"requête {i}, itération {k}"
            if k < n_iter:
                relevant = [r["image_id"] for r in resp["results"] if r["category"] == labels[i]]
                resp = engine.feedback(resp["session_id"], relevant)


def test_mean_only_moves_the_query_but_keeps_unit_weights(world):
    """mean_only = recherche euclidienne simple depuis la moyenne des pertinentes."""
    Z, _, labels = world
    i, trace = 10, []
    ev.evaluate_query(Z, labels, i, ["mean_only"], 4, config.SCOPE, [1, 25], trace=trace)
    shown = {k: s for _, k, s in trace}

    query = Z[i]
    order, _ = search(query, Z, scope=config.SCOPE, exclude_index=i)
    for k in range(1, 4):
        assert shown[k].tolist() == order.tolist()
        relevant = order[labels[order] == labels[i]]
        if len(relevant):
            query = update_query(Z[relevant])
        order, _ = search(query, Z, scope=config.SCOPE, exclude_index=i)    # sans poids
    assert shown[4].tolist() == order.tolist()


def test_type1_differs_from_mean_only_on_noisy_data(world):
    """Sur ces données (bruit dans la majorité des dimensions), les poids changent le résultat."""
    Z, _, labels = world
    q = ev.sample_queries(labels, 50, seed=0)
    res = ev.evaluate(Z, labels, q, n_iter=3)
    assert not np.allclose(res["p_at_scope"]["type1"][:, 1], res["p_at_scope"]["mean_only"][:, 1])
    assert res["p_at_scope"]["type1"][:, 1].mean() > res["p_at_scope"]["mean_only"][:, 1].mean()


def test_rank_points_are_capped_to_the_database_size():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(40, 4))
    labels = np.array(["a"] * 20 + ["b"] * 20)
    res = ev.evaluate(Z, labels, [0, 25], n_iter=2, scope=5)
    assert res["rank_points"].max() <= 39
    assert res["hits"]["type1"].shape == (2, 2, len(res["rank_points"]))


def test_invalid_arguments_raise():
    Z = np.random.default_rng(2).normal(size=(20, 3))
    labels = np.array(["a"] * 10 + ["b"] * 10)
    with pytest.raises(ValueError):
        ev.evaluate(Z, labels, [0], n_iter=1)                       # pas de feedback
    with pytest.raises(ValueError):
        ev.evaluate(Z, labels, [0], scope=20)                       # scope >= N
    with pytest.raises(ValueError):
        ev.evaluate(Z, labels[:-1], [0])                            # longueurs
    with pytest.raises(ValueError):
        ev.evaluate(Z, labels, [])                                  # aucune requête
    with pytest.raises(ValueError):
        ev.evaluate(Z, labels, [0], methods=("inconnue",), n_iter=2)
    single = np.array(["a"] * 19 + ["b"])
    with pytest.raises(ValueError):
        ev.evaluate(Z, single, [19])                                # catégorie d'une image


# --- Statistiques de synthèse ----------------------------------------------------
def _fake_result(p, hits=None, n_relevant=None, ranks=(1, 5)):
    p = np.asarray(p, dtype=float)
    return {"methods": ("m",), "n_iter": p.shape[1], "p_at_scope": {"m": p}, "ap": {"m": p},
            "rank_points": np.asarray(ranks), "n_relevant": np.asarray(n_relevant or [10]),
            "hits": {"m": np.asarray(hits if hits is not None else np.zeros((1, 1, 2)), float)},
            "query_idx": np.arange(p.shape[0])}


def test_paired_differences_by_hand():
    res = _fake_result([[0.5, 0.7], [0.5, 0.4]])
    (row,) = ev.paired_differences(res)
    assert row["iteration"] == 2
    assert row["mean_diff"] == pytest.approx(0.05)
    assert row["pct_better"] == 50.0 and row["pct_worse"] == 50.0
    assert row["ci95"] == pytest.approx(1.96 * np.std([0.2, -0.1], ddof=1) / np.sqrt(2))


def test_pr_curve_by_hand():
    res = _fake_result([[0.5]], hits=[[[1, 3]]], n_relevant=[10], ranks=(1, 5))
    rows = ev.pr_curve(res)
    assert [(r["rank"], r["precision"], r["recall"]) for r in rows] == [
        (1, pytest.approx(1.0), pytest.approx(0.1)),
        (5, pytest.approx(0.6), pytest.approx(0.3)),
    ]


def test_per_category_and_summary_shapes(world):
    Z, _, labels = world
    q = ev.sample_queries(labels, 25, seed=3)
    res = ev.evaluate(Z, labels, q, n_iter=3)
    assert len(ev.summarize(res)) == 3 * 3
    rows = ev.per_category(res, labels)
    assert len(rows) == 5 * 3 * 3 and {r["n_queries"] for r in rows} == {5}


# --- Script de bout en bout ------------------------------------------------------
@pytest.fixture
def features_dir(tmp_path):
    X, ids, cats = make_raw_world(seed=6, n_cat=4, per_cat=40, n_info=5, n_noise=15)
    np.save(tmp_path / "descriptors_raw.npy", X.astype(np.float32))
    mapping = [{"index": i, "image_id": ids[i], "filename": f"{i}.jpg", "category": cats[i],
                "path": ""} for i in range(len(ids))]
    (tmp_path / "image_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    return tmp_path


def test_script_writes_every_output(features_dir, tmp_path):
    out = tmp_path / "res"
    paths = script.run(features_dir, out, n_queries=20, n_iter=4, scope=10, seed=1,
                       expected_dims=20)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0, p.name

    assert paths["fig_precision"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths["fig_pr"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    header = paths["metrics"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "method,iteration,p_at_scope,p_ci95,map,map_ci95"
    methods_in_csv = {line.split(",")[0]
                      for line in paths["metrics"].read_text(encoding="utf-8").splitlines()[1:]}
    assert methods_in_csv == {"simple", "mean_only", "type1", "type1_cumulative"}

    markdown = paths["markdown"].read_text(encoding="utf-8")
    for expected in ("Table 1", "Table 2", "Table 3", "Table 4", "Protocole", "feedback oracle".lower()):
        assert expected.lower() in markdown.lower()

    info = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert info["n_images"] == 160 and info["n_queries"] == 20 and info["seed"] == 1
    assert info["query_excluded_from_results"] is True


def test_script_is_reproducible(features_dir, tmp_path):
    a = script.run(features_dir, tmp_path / "a", n_queries=16, n_iter=3, scope=10, seed=4,
                   expected_dims=20)
    b = script.run(features_dir, tmp_path / "b", n_queries=16, n_iter=3, scope=10, seed=4,
                   expected_dims=20)
    for key in ("metrics", "paired", "per_category", "pr_curve"):
        assert a[key].read_text(encoding="utf-8") == b[key].read_text(encoding="utf-8")


def test_script_rejects_data_violating_the_contract(features_dir, tmp_path):
    with pytest.raises(ValueError):
        script.run(features_dir, tmp_path / "out", expected_dims=172)   # dimension 20 != 172
