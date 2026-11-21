"""Contrôle l'échantillon étiré par rapport à la première validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from feature_extraction.descriptors import extract_raw_vector


RACINE = Path(__file__).resolve().parents[1]
RACINE_IMAGES = RACINE / "image_storage"
DOSSIER_REFERENCE = RACINE / "features" / "validation"
DOSSIER_STRETCHED = RACINE / "features" / "validation_stretched"
FICHIER_REFERENCE_NPY = DOSSIER_REFERENCE / "descriptors_raw.npy"
FICHIER_REFERENCE_JSON = DOSSIER_REFERENCE / "image_mapping.json"
FICHIER_STRETCHED_NPY = DOSSIER_STRETCHED / "descriptors_raw.npy"
FICHIER_STRETCHED_JSON = DOSSIER_STRETCHED / "image_mapping.json"


pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in (
        FICHIER_REFERENCE_NPY, FICHIER_REFERENCE_JSON,
        FICHIER_STRETCHED_NPY, FICHIER_STRETCHED_JSON,
    )),
    reason="Générer les deux jeux de validation avant les tests d'intégration.",
)


def test_validation_stretched_garde_exactement_le_mapping_et_les_500_lignes() -> None:
    mapping_reference = json.loads(FICHIER_REFERENCE_JSON.read_text(encoding="utf-8"))
    mapping_stretched = json.loads(FICHIER_STRETCHED_JSON.read_text(encoding="utf-8"))
    matrice = np.load(FICHIER_STRETCHED_NPY, allow_pickle=False)
    matrice_reference = np.load(FICHIER_REFERENCE_NPY, allow_pickle=False)

    assert len(mapping_stretched) == 500
    assert mapping_stretched == mapping_reference
    assert matrice.shape == (500, 172)
    assert matrice.dtype == np.float32
    assert np.isfinite(matrice).all()
    assert matrice_reference.shape == (500, 172)
    assert [entree["index"] for entree in mapping_stretched] == list(range(500))
    assert Counter(e["category"] for e in mapping_stretched) == Counter(
        {categorie: 25 for categorie in {e["category"] for e in mapping_reference}}
    )

    premier = mapping_stretched[0]
    vecteur_individuel = extract_raw_vector(
        RACINE_IMAGES / premier["path"], preprocessing_mode="stretch"
    )
    np.testing.assert_array_equal(matrice[0], vecteur_individuel)
