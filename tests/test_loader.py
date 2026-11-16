"""Tests de backend/services/loader.py."""
import json

import numpy as np
import pytest

from backend import cbir_config as config
from backend.services.loader import load_engine, validate_contract
from backend.services.normalization import load_normalizer


def write_features(folder, X, mapping=None):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "descriptors_raw.npy", X)
    if mapping is None:
        mapping = [
            {"index": i, "image_id": 1000 + i, "filename": f"{i}.jpg",
             "category": "a" if i < len(X) // 2 else "b", "path": f"d/{i}.jpg"}
            for i in range(len(X))
        ]
    with open(folder / "image_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping


@pytest.fixture
def X():
    return np.random.default_rng(0).random((40, config.N_DIMS)).astype(np.float32)


def test_load_engine_builds_a_working_engine(tmp_path, X):
    write_features(tmp_path, X)
    engine, normalizer = load_engine(tmp_path)

    resp = engine.start(query_image_id=1005, scope=5)
    assert resp["results"][0]["image_id"] == 1005             # image_id, pas index
    assert resp["results"][0]["distance"] == pytest.approx(0.0, abs=1e-9)
    assert {r["category"] for r in resp["results"]} <= {"a", "b"}
    assert normalizer["mean"].shape == (config.N_DIMS,)


def test_database_is_normalized(tmp_path, X):
    write_features(tmp_path, X)
    engine, _ = load_engine(tmp_path)
    assert np.allclose(engine.X.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(engine.X.std(axis=0), 1.0, atol=1e-6)


def test_normalizer_can_be_saved(tmp_path, X):
    write_features(tmp_path, X)
    out = tmp_path / "out" / "normalizer.npz"
    _, normalizer = load_engine(tmp_path, normalizer_out=out)
    assert np.array_equal(load_normalizer(out)["mean"], normalizer["mean"])


def test_other_dimension_is_accepted_when_declared(tmp_path):
    write_features(tmp_path, np.random.default_rng(1).random((10, 8)))
    engine, _ = load_engine(tmp_path, expected_dims=8)
    assert engine.start(query_image_id=1000, scope=3)["block_weights"] is None


def test_missing_files_raise_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="descriptors_raw.npy"):
        load_engine(tmp_path)


@pytest.mark.parametrize("bad", ["dims", "nan", "inf", "length", "order", "field"])
def test_contract_violations_are_rejected(tmp_path, X, bad):
    mapping = write_features(tmp_path, X)
    matrix, mp = X.copy(), list(mapping)
    if bad == "dims":
        matrix = matrix[:, :-1]
    elif bad == "nan":
        matrix[3, 4] = np.nan
    elif bad == "inf":
        matrix[3, 4] = np.inf
    elif bad == "length":
        mp = mp[:-1]
    elif bad == "order":
        mp[0], mp[1] = mp[1], mp[0]
    elif bad == "field":
        mp[2] = {k: v for k, v in mp[2].items() if k != "category"}
    with pytest.raises(ValueError):
        validate_contract(matrix, mp)


def test_duplicate_image_ids_are_rejected(tmp_path, X):
    mapping = write_features(tmp_path, X)
    mapping[1]["image_id"] = mapping[0]["image_id"]
    write_features(tmp_path, X, mapping)
    with pytest.raises(ValueError):
        load_engine(tmp_path)
