import json
from pathlib import Path

import numpy as np
import pytest

from feature_extraction import pipeline


def mapping_complet():
    return [
        {"index": i, "image_id": i + 1, "filename": f"{i:05}.jpg",
         "category": f"cat{i // 500:02}", "path": f"dataset/cat{i // 500:02}/{i:05}.jpg"}
        for i in range(10000)
    ]


def ecrire_mapping(path: Path, mapping):
    path.write_text(json.dumps(mapping), encoding="utf-8")


def test_extraction_complete_aligne_et_publie_seulement_apres_verification(tmp_path, monkeypatch):
    mapping = mapping_complet()
    mapping_path = tmp_path / "image_mapping.json"
    ecrire_mapping(mapping_path, mapping)
    mapping_octets = mapping_path.read_bytes()
    validation = tmp_path / "features" / "validation"
    stretched = tmp_path / "features" / "validation_stretched"
    validation.mkdir(parents=True)
    stretched.mkdir()
    (validation / "keep.txt").write_text("validation")
    (stretched / "keep.txt").write_text("stretched")
    sortie = tmp_path / "features" / "descriptors_raw.npy"
    sentinel = np.arange(10000, dtype=np.float32)

    def extraire(recu, storage, mode):
        assert list(recu) == mapping
        assert mode == "stretch"
        # Valeur de ligne distincte pour attester l'alignement à l'entrée.
        return np.repeat(np.arange(len(recu), dtype=np.float32)[:, None], 172, axis=1)

    monkeypatch.setattr(pipeline, "extraire_entrees", extraire)
    matrix, returned_mapping, path = pipeline.generer_complet(mapping_path, tmp_path, sortie, "stretch")

    loaded = np.load(path, allow_pickle=False)
    assert loaded.shape == (10000, 172)
    assert loaded.dtype == np.float32
    assert np.isfinite(loaded).all()
    assert np.array_equal(loaded[:, 0], sentinel)
    assert returned_mapping == mapping
    assert [x["index"] for x in returned_mapping] == list(range(10000))
    assert [x["image_id"] for x in returned_mapping] == list(range(1, 10001))
    assert mapping_path.read_bytes() == mapping_octets
    assert (validation / "keep.txt").read_text() == "validation"
    assert (stretched / "keep.txt").read_text() == "stretched"
    assert list(tmp_path.joinpath("features").glob(".descriptors_raw.npy.staging-*.npy")) == []


def test_echec_extraction_ne_publie_pas_et_nettoie_temporaire(tmp_path, monkeypatch):
    mapping_path = tmp_path / "mapping.json"
    ecrire_mapping(mapping_path, mapping_complet())
    sortie = tmp_path / "features" / "descriptors_raw.npy"

    def echec(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(pipeline, "extraire_entrees", echec)
    with pytest.raises(RuntimeError, match="échec simulé"):
        pipeline.generer_complet(mapping_path, tmp_path, sortie)
    assert not sortie.exists()
    assert list(sortie.parent.glob(".descriptors_raw.npy.staging-*.npy")) == []


def test_sortie_existante_refusee_sans_extraction(tmp_path, monkeypatch):
    mapping_path = tmp_path / "mapping.json"
    ecrire_mapping(mapping_path, mapping_complet())
    sortie = tmp_path / "features" / "descriptors_raw.npy"
    sortie.parent.mkdir()
    sortie.write_bytes(b"conserver")
    monkeypatch.setattr(pipeline, "extraire_entrees", lambda *a, **k: pytest.fail("ne doit pas extraire"))
    with pytest.raises(FileExistsError):
        pipeline.generer_complet(mapping_path, tmp_path, sortie)
    assert sortie.read_bytes() == b"conserver"


@pytest.mark.parametrize("correction", ["count", "index", "id", "order"])
def test_mapping_complet_invalide_est_refuse(tmp_path, correction):
    mapping = mapping_complet()
    if correction == "count":
        mapping.pop()
    elif correction == "index":
        mapping[20]["index"] = 19
    elif correction == "id":
        mapping[20]["image_id"] = 19
    else:
        mapping[0], mapping[1] = mapping[1], mapping[0]
        mapping[0]["index"], mapping[1]["index"] = 0, 1
        mapping[0]["image_id"], mapping[1]["image_id"] = 1, 2
    mapping_path = tmp_path / "mapping.json"
    ecrire_mapping(mapping_path, mapping)
    with pytest.raises(pipeline.MappingError):
        pipeline.generer_complet(mapping_path, tmp_path, tmp_path / "features" / "out.npy")
