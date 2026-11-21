"""Tests for the protected raw-feature API supplied by Member B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from flask_login import LoginManager, UserMixin, login_required

from backend.routes import features_routes


class TestUser(UserMixin):
    id = "tester"


@pytest.fixture
def feature_api(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    storage_dir = tmp_path / "image_storage"
    dataset_image = storage_dir / "dataset" / "sample" / "ghim.jpg"
    personal_image = storage_dir / "uploads" / "personal.jpg"
    second_personal_image = storage_dir / "uploads" / "second-personal.jpg"
    dataset_image.parent.mkdir(parents=True)
    personal_image.parent.mkdir(parents=True)
    dataset_image.write_bytes(b"ghim fixture")
    personal_image.write_bytes(b"personal fixture")
    second_personal_image.write_bytes(b"second personal fixture")

    vector0 = np.arange(172, dtype=np.float32)
    vector1 = np.arange(172, dtype=np.float32) + np.float32(1000)
    matrix = np.vstack((vector0, vector1)).astype(np.float32)
    mapping = [
        {"index": 0, "image_id": 1, "filename": "ghim.jpg", "category": "sample",
         "path": "dataset/sample/ghim.jpg"},
        {"index": 1, "image_id": 2, "filename": "other.jpg", "category": "sample",
         "path": "dataset/sample/other.jpg"},
    ]
    features_dir.mkdir()
    np.save(features_dir / "descriptors_raw.npy", matrix, allow_pickle=False)
    mapping_path = features_dir / "image_mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    original_matrix = (features_dir / "descriptors_raw.npy").read_bytes()
    original_mapping = mapping_path.read_bytes()

    paths = {1: dataset_image, 2: storage_dir / "dataset" / "sample" / "other.jpg",
             10: personal_image, 11: second_personal_image}
    calls = []

    def get_image_path(image_id):
        calls.append(image_id)
        return paths.get(image_id)

    def build_app(personal_cache_size=1024):
        app = Flask(__name__)
        app.config.update(TESTING=True, SECRET_KEY="test-secret")
        login_manager = LoginManager()
        login_manager.init_app(app)
        login_manager.login_view = "login_page"

        @login_manager.request_loader
        def load_test_user(req):
            return TestUser() if req.headers.get("X-Test-User") == "yes" else None

        @app.get("/login")
        def login_page():
            return "login page"

        @app.get("/private-page")
        @login_required
        def private_page():
            return "private page"

        app.register_blueprint(
            features_routes.create_features_blueprint(
                get_image_path,
                features_dir=features_dir,
                image_storage_dir=storage_dir,
                personal_cache_size=personal_cache_size,
            ),
            url_prefix="/api",
        )
        return app

    app = build_app()
    monkeypatch.setattr(features_routes, "extract_raw_vector", lambda path, preprocessing_mode: vector1.copy())
    return {
        "app": app,
        "client": app.test_client(),
        "paths": paths,
        "calls": calls,
        "features_dir": features_dir,
        "storage_dir": storage_dir,
        "matrix": matrix,
        "matrix_bytes": original_matrix,
        "mapping_bytes": original_mapping,
        "build_app": build_app,
    }


def auth_headers():
    return {"X-Test-User": "yes"}


def assert_feature_shape(body, image_id):
    assert body["image_id"] == image_id
    assert body["dimensions"] == 172
    assert body["dtype"] == "float32"
    assert body["preprocessing_mode"] == "stretch"
    assert len(body["color"]["hsv_histogram"]) == 128
    assert len(body["color"]["color_moments"]) == 9
    assert len(body["color"]["dominant_colors"]) == 9
    assert len(body["texture"]["lbp"]) == 10
    assert set(body["texture"]["glcm"]) == {"contrast", "correlation", "energy", "homogeneity"}
    assert all(len(values) == 4 for values in body["texture"]["glcm"].values())


def test_extract_individual_returns_contract_order_and_stretch(feature_api, monkeypatch):
    observed = {}

    def extract(path, preprocessing_mode):
        observed["path"] = path
        observed["mode"] = preprocessing_mode
        return np.arange(172, dtype=np.float32)

    monkeypatch.setattr(features_routes, "extract_raw_vector", extract)
    response = feature_api["client"].post(
        "/api/features/extract", json={"image_id": 10}, headers=auth_headers()
    )
    body = response.get_json()
    assert response.status_code == 200
    assert_feature_shape(body, 10)
    assert body["color"]["hsv_histogram"] == list(map(float, range(0, 128)))
    assert body["color"]["color_moments"] == list(map(float, range(128, 137)))
    assert body["color"]["dominant_colors"] == list(map(float, range(137, 146)))
    assert body["texture"]["glcm"]["contrast"] == list(map(float, range(146, 150)))
    assert body["texture"]["glcm"]["correlation"] == list(map(float, range(150, 154)))
    assert body["texture"]["glcm"]["energy"] == list(map(float, range(154, 158)))
    assert body["texture"]["glcm"]["homogeneity"] == list(map(float, range(158, 162)))
    assert body["texture"]["lbp"] == list(map(float, range(162, 172)))
    assert observed == {"path": feature_api["paths"][10], "mode": "stretch"}


def test_extract_batch_success(feature_api, monkeypatch):
    seen = []

    def extract(path, preprocessing_mode):
        seen.append((Path(path), preprocessing_mode))
        return np.full(172, len(seen), dtype=np.float32)

    monkeypatch.setattr(features_routes, "extract_raw_vector", extract)
    response = feature_api["client"].post(
        "/api/features/extract-batch", json={"image_ids": [10, 11]}, headers=auth_headers()
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["count"] == 2 and body["failed"] == 0
    assert [item["image_id"] for item in body["results"]] == [10, 11]
    assert all("color" in item and "texture" in item for item in body["results"])
    assert seen == [(feature_api["paths"][10], "stretch"), (feature_api["paths"][11], "stretch")]


def test_extract_batch_keeps_successes_and_per_image_errors(feature_api, monkeypatch):
    def extract(path, preprocessing_mode):
        if Path(path) == feature_api["paths"][11]:
            raise features_routes.ImageReadError("image fixture illisible")
        return np.zeros(172, dtype=np.float32)

    monkeypatch.setattr(features_routes, "extract_raw_vector", extract)
    response = feature_api["client"].post(
        "/api/features/extract-batch", json={"image_ids": [10, 11, 999]}, headers=auth_headers()
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["count"] == 3 and body["failed"] == 2
    assert "color" in body["results"][0]
    assert body["results"][1]["error"]["status"] == 422
    assert body["results"][2]["image_id"] == 999
    assert body["results"][2]["error"]["status"] == 404


@pytest.mark.parametrize("image_ids", [[], None, "123"])
def test_empty_or_malformed_batch_returns_400(feature_api, image_ids):
    response = feature_api["client"].post(
        "/api/features/extract-batch", json={"image_ids": image_ids}, headers=auth_headers()
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_batch_over_100_returns_400_without_resolving_images(feature_api):
    response = feature_api["client"].post(
        "/api/features/extract-batch", json={"image_ids": list(range(1, 102))}, headers=auth_headers()
    )
    assert response.status_code == 400
    assert feature_api["calls"] == []


@pytest.mark.parametrize("image_id", [None, "10", 1.5, True, 0, -1])
def test_invalid_single_identifier_returns_400(feature_api, image_id):
    response = feature_api["client"].post(
        "/api/features/extract", json={"image_id": image_id}, headers=auth_headers()
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_json_request_returns_400(feature_api):
    response = feature_api["client"].post(
        "/api/features/extract", data="not json", headers=auth_headers()
    )
    assert response.status_code == 400


def test_missing_or_inaccessible_image_returns_404(feature_api):
    response = feature_api["client"].post(
        "/api/features/extract", json={"image_id": 999}, headers=auth_headers()
    )
    assert response.status_code == 404
    assert response.get_json() == {"error": "Image introuvable ou inaccessible : 999."}


def test_existing_but_unreadable_image_returns_422(feature_api, monkeypatch):
    def unreadable(*args, **kwargs):
        raise features_routes.ImageReadError("decode failure")

    monkeypatch.setattr(features_routes, "extract_raw_vector", unreadable)
    response = feature_api["client"].post(
        "/api/features/extract", json={"image_id": 10}, headers=auth_headers()
    )
    assert response.status_code == 422
    assert "error" in response.get_json()


def test_nonfinite_extractor_vector_returns_422(feature_api, monkeypatch):
    vector = np.zeros(172, dtype=np.float32)
    vector[17] = np.inf
    monkeypatch.setattr(features_routes, "extract_raw_vector", lambda *a, **k: vector)
    response = feature_api["client"].post(
        "/api/features/extract", json={"image_id": 10}, headers=auth_headers()
    )
    assert response.status_code == 422
    assert "NaN" in response.get_json()["error"] or "infinie" in response.get_json()["error"]


def test_get_ghim_reads_raw_matrix_and_checks_access_first(feature_api):
    response = feature_api["client"].get("/api/features/1", headers=auth_headers())
    body = response.get_json()
    assert response.status_code == 200
    assert_feature_shape(body, 1)
    assert body["color"]["hsv_histogram"] == list(map(float, range(128)))
    assert feature_api["calls"] == [1]


def test_get_personal_features_after_post_uses_cached_raw_vector(feature_api, monkeypatch):
    vector = np.arange(172, dtype=np.float32) + np.float32(50)
    monkeypatch.setattr(features_routes, "extract_raw_vector", lambda *a, **k: vector.copy())
    post = feature_api["client"].post(
        "/api/features/extract", json={"image_id": 10}, headers=auth_headers()
    )
    get = feature_api["client"].get("/api/features/10", headers=auth_headers())
    assert post.status_code == get.status_code == 200
    assert post.get_json() == get.get_json()
    assert feature_api["calls"][-1] == 10


def test_same_image_id_resolved_to_new_path_never_returns_old_vector(feature_api, monkeypatch):
    old_path = feature_api["paths"][10]
    monkeypatch.setattr(
        features_routes, "extract_raw_vector",
        lambda *a, **k: np.full(172, 7, dtype=np.float32),
    )
    client = feature_api["client"]
    assert client.post("/api/features/extract", json={"image_id": 10},
                       headers=auth_headers()).status_code == 200
    feature_api["paths"][10] = feature_api["paths"][11]
    response = client.get("/api/features/10", headers=auth_headers())
    assert response.status_code == 404
    assert "Aucun descripteur" in response.get_json()["error"]
    assert feature_api["paths"][10] != old_path


def test_personal_cache_is_limited_to_1024_entries():
    cache = features_routes._PersonalFeatureCache(1024)
    signature = ("C:/images/personal.jpg", 10, 20)
    vector = np.zeros(172, dtype=np.float32)
    for image_id in range(1025):
        cache.put(image_id, signature, vector)
    assert len(cache._entries) == 1024
    assert cache.get(0, signature) is None
    assert cache.get(1, signature) is not None


def test_personal_cache_uses_lru_eviction():
    cache = features_routes._PersonalFeatureCache(2)
    signature = ("C:/images/personal.jpg", 10, 20)
    vector = np.zeros(172, dtype=np.float32)
    cache.put(1, signature, vector)
    cache.put(2, signature, vector)
    assert cache.get(1, signature) is not None  # make 1 most recently used
    cache.put(3, signature, vector)
    assert cache.get(2, signature) is None
    assert cache.get(1, signature) is not None
    assert cache.get(3, signature) is not None


def test_post_ghim_does_not_evict_personal_feature_cache(feature_api, monkeypatch):
    monkeypatch.setattr(
        features_routes, "extract_raw_vector",
        lambda *a, **k: np.full(172, 77, dtype=np.float32),
    )
    client = feature_api["build_app"](personal_cache_size=1).test_client()
    assert client.post("/api/features/extract", json={"image_id": 10},
                       headers=auth_headers()).status_code == 200
    assert client.post("/api/features/extract", json={"image_id": 1},
                       headers=auth_headers()).status_code == 200
    response = client.get("/api/features/10", headers=auth_headers())
    assert response.status_code == 200
    assert response.get_json()["color"]["hsv_histogram"] == [77.0] * 128


def test_get_personal_image_without_cached_features_returns_404(feature_api):
    response = feature_api["client"].get("/api/features/11", headers=auth_headers())
    assert response.status_code == 404
    assert feature_api["calls"] == [11]


def test_cached_personal_features_are_not_returned_when_access_is_revoked(feature_api, monkeypatch):
    monkeypatch.setattr(features_routes, "extract_raw_vector", lambda *a, **k: np.zeros(172, dtype=np.float32))
    client = feature_api["client"]
    assert client.post("/api/features/extract", json={"image_id": 10},
                       headers=auth_headers()).status_code == 200
    # A's resolver is the access-control boundary; a later denial must beat the cache.
    feature_api["paths"].pop(10)
    response = client.get("/api/features/10", headers=auth_headers())
    assert response.status_code == 404
    assert response.get_json()["error"].startswith("Image introuvable")


def test_all_routes_require_authentication(feature_api):
    client = feature_api["client"]
    responses = [
        client.post("/api/features/extract", json={"image_id": 10}),
        client.post("/api/features/extract-batch", json={"image_ids": [10]}),
        client.get("/api/features/1"),
    ]
    assert [response.status_code for response in responses] == [401, 401, 401]
    assert [response.get_json() for response in responses] == [
        {"error": "Authentification requise"},
    ] * 3
    assert feature_api["calls"] == []


def test_global_login_redirect_is_unchanged(feature_api):
    response = feature_api["client"].get("/private-page")
    assert response.status_code == 302
    assert response.headers["Location"].startswith("/login?next=")


def test_routes_do_not_modify_matrix_or_mapping(feature_api, monkeypatch):
    monkeypatch.setattr(features_routes, "extract_raw_vector", lambda *a, **k: np.zeros(172, dtype=np.float32))
    client = feature_api["client"]
    client.get("/api/features/1", headers=auth_headers())
    client.post("/api/features/extract", json={"image_id": 10}, headers=auth_headers())
    client.post("/api/features/extract-batch", json={"image_ids": [10]}, headers=auth_headers())
    assert (feature_api["features_dir"] / "descriptors_raw.npy").read_bytes() == feature_api["matrix_bytes"]
    assert (feature_api["features_dir"] / "image_mapping.json").read_bytes() == feature_api["mapping_bytes"]

