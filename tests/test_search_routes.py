"""Tests de backend/routes/search_routes.py (Tâche 6)."""
import functools

import numpy as np
import pytest
from flask import Flask, request
from helpers import make_raw_world

from backend import cbir_config as config
from backend.routes.search_routes import create_search_blueprint
from backend.services.normalization import fit_normalizer, transform
from backend.services.session import SearchEngine


def build(X, ids, cats, **blueprint_kwargs):
    normalizer = fit_normalizer(X)
    engine = SearchEngine(transform(X, normalizer), ids, cats)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(
        create_search_blueprint(engine, normalizer=normalizer, **blueprint_kwargs),
        url_prefix="/api",
    )
    return app, engine, normalizer


@pytest.fixture
def world():
    return make_raw_world()


@pytest.fixture
def client(world):
    app, _, _ = build(*world)
    return app.test_client()


def search(client, **body):
    return client.post("/api/search", json=body)


def relevant(resp_json, category):
    return [r["image_id"] for r in resp_json["results"] if r["category"] == category]


# --- /search -----------------------------------------------------------------
def test_search_follows_the_contract(client):
    r = search(client, query_image_id=100, scope=25)
    assert r.status_code == 200
    body = r.get_json()
    assert body["iteration"] == 1
    assert isinstance(body["session_id"], str)
    assert len(body["results"]) == 25
    assert set(body["results"][0]) == {"image_id", "distance", "category"}
    assert body["results"][0]["image_id"] == 100            # la requête est en rang 1
    assert body["method"] == config.DEFAULT_METHOD


def test_scope_defaults_to_config(client):
    assert len(search(client, query_image_id=100).get_json()["results"]) == config.SCOPE


def test_exclude_query_removes_the_query_image(client):
    body = search(client, query_image_id=100, exclude_query=True).get_json()
    assert 100 not in [r["image_id"] for r in body["results"]]


def test_method_is_selectable(client):
    body = search(client, query_image_id=100, method="cumulative").get_json()
    assert body["method"] == "cumulative"


@pytest.mark.parametrize("body", [
    {},                                              # id manquant
    {"query_image_id": "100"},                       # mauvais type
    {"query_image_id": 100.5},
    {"query_image_id": True},                        # un booléen n'est pas un id
    {"query_image_id": 100, "scope": 0},
    {"query_image_id": 100, "scope": -3},
    {"query_image_id": 100, "scope": config.MAX_SCOPE + 1},
    {"query_image_id": 100, "scope": "25"},
    {"query_image_id": 100, "scope": True},
    {"query_image_id": 100, "method": "autre"},
    {"query_image_id": 100, "exclude_query": "oui"},
])
def test_invalid_search_requests_return_400(client, body):
    r = search(client, **body)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_non_json_body_returns_400(client):
    assert client.post("/api/search", data="pas du json").status_code == 400
    assert client.post("/api/search", json=[1, 2]).status_code == 400


def test_unknown_image_returns_404(client):
    r = search(client, query_image_id=999999)
    assert r.status_code == 404


# --- /feedback ---------------------------------------------------------------
def test_feedback_flow(client, world):
    cats = world[2]
    first = search(client, query_image_id=100, exclude_query=True).get_json()
    rel = relevant(first, cats[0])
    assert len(rel) >= 2

    r = client.post("/api/feedback", json={"session_id": first["session_id"], "relevant_ids": rel})
    assert r.status_code == 200
    second = r.get_json()
    assert second["session_id"] == first["session_id"]
    assert second["iteration"] == 2
    assert second["feedback"]["mode"] == "full"
    assert [x["image_id"] for x in second["results"]] != [x["image_id"] for x in first["results"]]
    assert 100 not in [x["image_id"] for x in second["results"]]     # exclusion conservée


def test_empty_feedback_is_accepted(client):
    first = search(client, query_image_id=100).get_json()
    r = client.post("/api/feedback", json={"session_id": first["session_id"], "relevant_ids": []})
    assert r.status_code == 200
    assert r.get_json()["feedback"]["mode"] == "no_feedback"


@pytest.mark.parametrize("body", [
    {"relevant_ids": [100]},                                  # session_id manquant
    {"session_id": "", "relevant_ids": []},
    {"session_id": 12, "relevant_ids": []},
    {"session_id": "x"},                                      # relevant_ids manquant
    {"session_id": "x", "relevant_ids": 100},                 # pas une liste
    {"session_id": "x", "relevant_ids": ["100"]},
    {"session_id": "x", "relevant_ids": [True]},
    {"session_id": "x", "relevant_ids": list(range(config.MAX_SCOPE + 1))},
])
def test_invalid_feedback_requests_return_400(client, body):
    assert client.post("/api/feedback", json=body).status_code == 400


def test_unknown_session_returns_404(client):
    r = client.post("/api/feedback", json={"session_id": "nope", "relevant_ids": []})
    assert r.status_code == 404


def test_image_not_displayed_returns_400(client, world):
    first = search(client, query_image_id=100, exclude_query=True).get_json()
    shown = {x["image_id"] for x in first["results"]}
    other = next(i for i in world[1] if i not in shown)
    r = client.post("/api/feedback",
                    json={"session_id": first["session_id"], "relevant_ids": [other]})
    assert r.status_code == 400


def test_unknown_image_in_feedback_returns_400(client):
    first = search(client, query_image_id=100).get_json()
    r = client.post("/api/feedback",
                    json={"session_id": first["session_id"], "relevant_ids": [999999]})
    assert r.status_code == 400


def test_iteration_limit_returns_409(world):
    X, ids, cats = world
    app, engine, _ = build(X, ids, cats)
    engine.max_iterations = 2
    client = app.test_client()
    sid = search(client, query_image_id=100).get_json()["session_id"]
    assert client.post("/api/feedback", json={"session_id": sid, "relevant_ids": []}).status_code == 200
    r = client.post("/api/feedback", json={"session_id": sid, "relevant_ids": []})
    assert r.status_code == 409


# --- Authentification fournie par A ----------------------------------------------
def fake_login_required(view):
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if request.headers.get("X-User") != "aya":
            return {"error": "authentification requise"}, 401
        return view(*args, **kwargs)
    return wrapper


def test_auth_decorators_protect_both_endpoints(world):
    app, _, _ = build(*world, auth_decorators=[fake_login_required])
    client = app.test_client()
    assert search(client, query_image_id=100).status_code == 401
    assert client.post("/api/feedback", json={"session_id": "x", "relevant_ids": []}).status_code == 401

    r = client.post("/api/search", json={"query_image_id": 100}, headers={"X-User": "aya"})
    assert r.status_code == 200


def test_endpoints_are_public_without_decorators(client):
    assert search(client, query_image_id=100).status_code == 200


# --- Images uploadées (hors base) -------------------------------------------------
def test_vector_provider_serves_images_outside_the_index(world):
    X, ids, cats = world
    uploaded_id, raw_uploaded = ids[-1], X[-1]
    app, engine, normalizer = build(
        X[:-1], ids[:-1], cats[:-1],
        vector_provider=lambda image_id: raw_uploaded if image_id == uploaded_id else None,
    )
    client = app.test_client()

    r = search(client, query_image_id=uploaded_id, scope=10)
    assert r.status_code == 200
    direct =engine.start(query_vector=transform(raw_uploaded, normalizer), scope=10)
    assert [x["image_id"] for x in r.get_json()["results"]] == \
           [x["image_id"] for x in direct["results"]]

    assert search(client, query_image_id=424242).status_code == 404     # inconnu du provider


def test_vector_provider_requires_a_normalizer(world):
    X, ids, cats = world
    engine = SearchEngine(X, ids, cats)
    with pytest.raises(ValueError):
        create_search_blueprint(engine, vector_provider=lambda i: None)


def test_two_apps_do_not_share_sessions(world):
    app1, _, _ = build(*world)
    app2, _, _ = build(*world)
    sid = search(app1.test_client(), query_image_id=100).get_json()["session_id"]
    r = app2.test_client().post("/api/feedback", json={"session_id": sid, "relevant_ids": []})
    assert r.status_code == 404
