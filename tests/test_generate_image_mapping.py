"""Tests des règles de génération du mapping d'images."""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.generate_image_mapping import (
    construire_mapping,
    ecrire_mapping,
    est_image_valide,
)


def creer_image(chemin: Path, couleur: tuple[int, int, int] = (20, 40, 60)) -> None:
    """Crée une petite image décodable pour les tests, sans dépendre du dataset."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 2), couleur).save(chemin)


def test_mapping_trie_et_attribue_les_identifiants(tmp_path: Path) -> None:
    """Les catégories et noms sont triés; index et image_id suivent les règles."""
    storage = tmp_path / "image_storage"
    dataset = storage / "dataset"
    creer_image(dataset / "Zebras" / "2.jpg")
    creer_image(dataset / "Animals" / "b.png")
    creer_image(dataset / "Animals" / "A.jpg")
    creer_image(dataset / "Animals" / "nested" / "0.webp")
    creer_image(dataset / "Animals" / "1.gif")
    (dataset / "Animals" / "not-image.txt").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "Animals" / "not-image.txt").write_text("texte", encoding="utf-8")
    creer_image(dataset / "root.jpg")

    mapping = construire_mapping(dataset, 4, storage)

    assert [item["category"] for item in mapping] == ["Animals"] * 4
    assert [item["filename"] for item in mapping] == ["0.webp", "1.gif", "A.jpg", "b.png"]
    assert [item["index"] for item in mapping] == [0, 1, 2, 3]
    assert [item["image_id"] for item in mapping] == [1, 2, 3, 4]
    assert all(item["image_id"] == item["index"] + 1 for item in mapping)
    assert mapping[0]["path"] == "dataset/Animals/nested/0.webp"
    assert all("\\" not in item["path"] and not Path(item["path"]).is_absolute() for item in mapping)
    assert all((storage / item["path"]).is_file() for item in mapping)
    assert len({item["path"] for item in mapping}) == len(mapping)
    assert all(set(item) == {"index", "image_id", "filename", "category", "path"} for item in mapping)


def test_limite_exactement_au_nombre_demande_et_ordre_deterministe(tmp_path: Path) -> None:
    dataset = tmp_path / "storage" / "dataset"
    for categorie, nom in [("Zoo", "b.jpg"), ("Arts", "z.jpg"), ("Arts", "a.jpg")]:
        creer_image(dataset / categorie / nom)

    premier = construire_mapping(dataset, 2)
    second = construire_mapping(dataset, 2)

    assert premier == second
    assert len(premier) == 2
    assert [(x["category"], x["filename"]) for x in premier] == [
        ("Arts", "a.jpg"), ("Arts", "z.jpg")
    ]


def test_genere_500_entrees_avec_indices_et_identifiants_conformes(tmp_path: Path) -> None:
    """Le format reste conforme au contrat sur la taille demandée par défaut."""
    dataset = tmp_path / "storage" / "dataset"
    for index in range(500):
        creer_image(dataset / "Animals" / f"{index:03}.jpg")

    mapping = construire_mapping(dataset, 500)

    assert len(mapping) == 500
    assert [item["index"] for item in mapping] == list(range(500))
    assert [item["image_id"] for item in mapping] == list(range(1, 501))
    assert all(item["image_id"] == item["index"] + 1 for item in mapping)
    assert all((dataset.parent / item["path"]).is_file() for item in mapping)


def test_insuffisance_ne_produit_pas_de_mapping_factice(tmp_path: Path) -> None:
    dataset = tmp_path / "storage" / "dataset"
    creer_image(dataset / "Animals" / "only.jpg")

    with pytest.raises(ValueError, match="Seulement 1"):
        construire_mapping(dataset, 2)


def test_refuse_les_fichiers_invalides_meme_si_extension_image(tmp_path: Path) -> None:
    fichier = tmp_path / "broken.jpg"
    fichier.write_text("ceci n'est pas une image", encoding="utf-8")

    assert not est_image_valide(fichier)


def test_json_ecrit_et_relu_sans_perte(tmp_path: Path) -> None:
    mapping = [{
        "index": 0, "image_id": 1, "filename": "été.jpg",
        "category": "Animaux", "path": "dataset/Animaux/été.jpg",
    }]
    sortie = tmp_path / "features" / "image_mapping.json"

    ecrire_mapping(mapping, sortie)

    assert json.loads(sortie.read_text(encoding="utf-8")) == mapping
    assert "été" in sortie.read_text(encoding="utf-8")


def test_rejette_nombre_non_positif(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        construire_mapping(tmp_path, 0)
