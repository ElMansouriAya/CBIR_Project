"""Validation d'intégration du mapping contre les images GHIM-10K réelles.

Ces tests ne fabriquent aucune image. Ils sont ignorés dans les environnements
où le dataset n'est pas installé, mais vérifient strictement les 10 000 images
et les 500 images par catégorie lorsqu'il est présent.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from scripts.generate_image_mapping import collecter_images


RACINE_PROJET = Path(__file__).resolve().parents[1]
DOSSIER_STORAGE = RACINE_PROJET / "image_storage"
DOSSIER_DATASET = DOSSIER_STORAGE / "dataset"
FICHIER_MAPPING = RACINE_PROJET / "features" / "image_mapping.json"
CATEGORIES_ATTENDUES = {
    "aircraft", "buildings", "butterfly", "cars", "dragon fly", "fireworks",
    "flowers", "hen", "horses", "insects", "motorcycles", "mountains",
    "sea shores", "ships", "sunset", "temples", "trees", "valleys", "walls", "yacht",
}


pytestmark = pytest.mark.skipif(
    not DOSSIER_DATASET.is_dir(), reason="Le dataset GHIM-10K n'est pas installé."
)


def test_dataset_reel_et_mapping_complet_sont_coherents() -> None:
    """Contrôle l'inventaire, le JSON, les identifiants et chaque image réelle."""
    images_valides = collecter_images(DOSSIER_DATASET)
    comptage_categories = Counter(categorie for categorie, _, _ in images_valides)
    mapping_attendu = [
        {
            "index": index,
            "image_id": index + 1,
            "filename": nom,
            "category": categorie,
            "path": chemin.relative_to(DOSSIER_STORAGE).as_posix(),
        }
        for index, (categorie, nom, chemin) in enumerate(images_valides)
    ]

    assert len(images_valides) == 10000, (
        f"{len(images_valides)} images valides détectées au lieu de 10 000; "
        f"comptage par catégorie : {dict(sorted(comptage_categories.items()))}"
    )
    assert set(comptage_categories) == CATEGORIES_ATTENDUES
    assert comptage_categories == Counter({categorie: 500 for categorie in CATEGORIES_ATTENDUES})
    assert FICHIER_MAPPING.is_file(), "Générer le mapping avant le test d'intégration."

    mapping_json = json.loads(FICHIER_MAPPING.read_text(encoding="utf-8"))
    assert mapping_json == mapping_attendu
    assert len(mapping_json) == 10000
    assert [entree["index"] for entree in mapping_json] == list(range(10000))
    assert [entree["image_id"] for entree in mapping_json] == list(range(1, 10001))
    assert all(entree["image_id"] == entree["index"] + 1 for entree in mapping_json)
    assert len({entree["image_id"] for entree in mapping_json}) == len(mapping_json)
    assert len({(entree["category"], entree["filename"]) for entree in mapping_json}) == len(mapping_json)

    ordre = [
        (entree["category"].casefold(), entree["category"],
         entree["filename"].casefold(), entree["filename"], entree["path"].casefold(), entree["path"])
        for entree in mapping_json
    ]
    assert ordre == sorted(ordre)
    assert all(set(entree) == {"index", "image_id", "filename", "category", "path"}
               for entree in mapping_json)

    for entree in mapping_json:
        chemin_relatif = entree["path"]
        assert "\\" not in chemin_relatif
        assert not Path(chemin_relatif).is_absolute()
        chemin_reel = DOSSIER_STORAGE / chemin_relatif
        assert chemin_reel.is_file(), f"Fichier manquant : {chemin_relatif}"
        with Image.open(chemin_reel) as image:
            image.verify()
        with Image.open(chemin_reel) as image:
            image.load()

    categories_dossiers = {dossier.name for dossier in DOSSIER_DATASET.iterdir() if dossier.is_dir()}
    assert set(comptage_categories) == categories_dossiers
