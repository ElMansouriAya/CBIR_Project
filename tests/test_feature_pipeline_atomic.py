"""Tests de publication cohérente des artefacts du pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from feature_extraction import pipeline


def test_echec_de_basculement_restaure_l_ancienne_paire(tmp_path, monkeypatch) -> None:
    """Une erreur au commit restaure le dossier précédent sans mélanger les fichiers."""
    dossier = tmp_path / "validation"
    dossier.mkdir()
    ancienne_matrice = np.full((1, 172), 3.0, dtype=np.float32)
    with (dossier / "descriptors_raw.npy").open("wb") as fichier:
        np.save(fichier, ancienne_matrice, allow_pickle=False)
    ancien_mapping = [{"index": 0, "image_id": 17, "filename": "old.jpg", "category": "A", "path": "old.jpg"}]
    (dossier / "image_mapping.json").write_text(json.dumps(ancien_mapping), encoding="utf-8")

    vrai_replace = os.replace

    def remplacer_sauf_publication(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.startswith(".validation.staging-") and destination_path == dossier:
            raise OSError("échec simulé pendant la publication")
        return vrai_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", remplacer_sauf_publication)
    nouvelle_matrice = np.full((1, 172), 8.0, dtype=np.float32)
    nouveau_mapping = [{"index": 0, "image_id": 18, "filename": "new.jpg", "category": "B", "path": "new.jpg"}]

    with pytest.raises(OSError, match="échec simulé"):
        pipeline._ecrire_sorties_atomiquement(nouvelle_matrice, nouveau_mapping, dossier)

    matrice_restaurée = np.load(dossier / "descriptors_raw.npy", allow_pickle=False)
    mapping_restauré = json.loads((dossier / "image_mapping.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(matrice_restaurée, ancienne_matrice)
    assert mapping_restauré == ancien_mapping
    assert not (tmp_path / ".validation.previous").exists()
    assert not list(tmp_path.glob(".validation.staging-*"))
