"""Vérifie les artefacts de validation produits depuis le dataset réel."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from feature_extraction.descriptors import extract_raw_vector
from feature_extraction.pipeline import lire_mapping


RACINE = Path(__file__).resolve().parents[1]
MAPPING_COMPLET = RACINE / "features" / "image_mapping.json"
RACINE_IMAGES = RACINE / "image_storage"
DOSSIER_VALIDATION = RACINE / "features" / "validation"
FICHIER_NPY = DOSSIER_VALIDATION / "descriptors_raw.npy"
FICHIER_JSON = DOSSIER_VALIDATION / "image_mapping.json"


pytestmark = pytest.mark.skipif(
    not FICHIER_NPY.is_file() or not FICHIER_JSON.is_file(),
    reason="Lancer scripts/extract_features.py avant les tests d'intégration.",
)


def test_artefacts_validation_reels_et_alignement_ligne_mapping() -> None:
    """Contrôle 25 images par catégorie et l'égalité entre les deux voies d'extraction."""
    mapping_complet = lire_mapping(MAPPING_COMPLET)
    mapping_validation = json.loads(FICHIER_JSON.read_text(encoding="utf-8"))
    matrice = np.load(FICHIER_NPY, allow_pickle=False)
    comptes = Counter(entree["category"] for entree in mapping_validation)

    assert len(mapping_validation) == 500
    assert comptes == Counter({categorie: 25 for categorie in {e["category"] for e in mapping_complet}})
    assert [e["index"] for e in mapping_validation] == list(range(500))
    assert matrice.shape == (500, 172)
    assert matrice.dtype == np.float32
    assert np.isfinite(matrice).all()

    # La sélection conserve les premières 25 entrées de chaque catégorie,
    # dans l'ordre global, et ne change que l'index local de la ligne.
    selection_attendue = []
    prises: Counter[str] = Counter()
    for entree in mapping_complet:
        if prises[entree["category"]] < 25:
            selection_attendue.append(entree)
            prises[entree["category"]] += 1
    assert len(selection_attendue) == 500
    for index_local, (attendue, obtenue) in enumerate(zip(selection_attendue, mapping_validation)):
        assert obtenue["index"] == index_local
        assert obtenue["image_id"] == attendue["image_id"]
        assert obtenue["category"] == attendue["category"]
        assert obtenue["filename"] == attendue["filename"]
        assert obtenue["path"] == attendue["path"]

    chemin_premiere_image = RACINE_IMAGES / mapping_validation[0]["path"]
    vecteur_individuel = extract_raw_vector(chemin_premiere_image)
    np.testing.assert_array_equal(matrice[0], vecteur_individuel)
