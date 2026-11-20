"""Pipeline par lot piloté exclusivement par l'ordre du mapping JSON."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extraction.config import N_DIMS, PREPROCESSING_LETTERBOX
from feature_extraction.descriptors import extract_raw_vector


class MappingError(ValueError):
    """Mapping absent, invalide ou incompatible avec l'ordre attendu."""


class ImageExtractionError(RuntimeError):
    """Erreur d'extraction contextualisée sans sauter une ligne du mapping."""


def lire_mapping(chemin_mapping: str | Path) -> list[dict[str, Any]]:
    """Charge et contrôle l'indexation du mapping, sans réordonner ses entrées."""
    chemin = Path(chemin_mapping)
    try:
        mapping = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erreur:
        raise MappingError(f"Impossible de lire le mapping {chemin}: {erreur}") from erreur
    if not isinstance(mapping, list) or not mapping:
        raise MappingError("Le mapping doit être une liste JSON non vide.")

    identifiants: set[int] = set()
    champs_requis = {"index", "image_id", "filename", "category", "path"}
    for position, entree in enumerate(mapping):
        if not isinstance(entree, dict) or not champs_requis.issubset(entree):
            raise MappingError(f"Entrée {position}: champs requis absents ou entrée invalide.")
        if entree["index"] != position:
            raise MappingError(
                f"Entrée de position {position} porte index={entree['index']!r}; "
                "les index doivent correspondre aux lignes du mapping."
            )
        image_id = entree["image_id"]
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in identifiants:
            raise MappingError(f"Entrée {position}: image_id invalide ou dupliqué ({image_id!r}).")
        identifiants.add(image_id)
        chemin_relatif = entree["path"]
        if not isinstance(chemin_relatif, str) or not chemin_relatif or "\\" in chemin_relatif:
            raise MappingError(f"Entrée {position}: path doit être relatif avec des séparateurs '/'.")
        if Path(chemin_relatif).is_absolute() or Path(chemin_relatif).drive:
            raise MappingError(f"Entrée {position}: path absolu interdit ({chemin_relatif!r}).")
    return mapping


def selectionner_echantillon_equilibre(
    mapping: Sequence[dict[str, Any]], images_par_categorie: int = 25
) -> list[dict[str, Any]]:
    """Prend les premières entrées de chaque catégorie en gardant l'ordre source.

    Les chemins et image_id restent ceux du mapping complet; seul ``index`` est
    renuméroté localement pour la matrice d'évaluation.
    """
    if images_par_categorie <= 0:
        raise ValueError("images_par_categorie doit être strictement positif.")
    compteurs: Counter[str] = Counter()
    selection: list[dict[str, Any]] = []
    for entree in mapping:
        categorie = entree["category"]
        if compteurs[categorie] >= images_par_categorie:
            continue
        copie = dict(entree)
        copie["index"] = len(selection)
        selection.append(copie)
        compteurs[categorie] += 1

    categories_incompletes = {
        categorie: nombre
        for categorie, nombre in compteurs.items()
        if nombre != images_par_categorie
    }
    if categories_incompletes:
        raise MappingError(
            f"Certaines catégories ont moins de {images_par_categorie} images : "
            f"{categories_incompletes}"
        )
    return selection


def extraire_entrees(
    mapping: Sequence[dict[str, Any]],
    dossier_image_storage: str | Path,
    preprocessing_mode: str = PREPROCESSING_LETTERBOX,
) -> np.ndarray:
    """Calcule une ligne par entrée, dans l'ordre fourni, sans en ignorer."""
    racine = Path(dossier_image_storage).resolve()
    matrice = np.empty((len(mapping), N_DIMS), dtype=np.float32)
    for position, entree in enumerate(mapping):
        chemin = (racine / entree["path"]).resolve()
        if not chemin.is_relative_to(racine):
            raise MappingError(f"Entrée {position}: le chemin sort de image_storage: {entree['path']}")
        try:
            vecteur = extract_raw_vector(chemin, preprocessing_mode=preprocessing_mode)
        except Exception as erreur:
            raise ImageExtractionError(
                "Échec d'extraction : "
                f"index={entree['index']}, image_id={entree['image_id']}, path={entree['path']}"
            ) from erreur
        if vecteur.shape != (N_DIMS,) or vecteur.dtype != np.float32:
            raise ImageExtractionError(
                f"Vecteur invalide pour index={entree['index']}, image_id={entree['image_id']}, "
                f"path={entree['path']}: shape={vecteur.shape}, dtype={vecteur.dtype}."
            )
        matrice[position] = vecteur
    if matrix_error := _erreur_matrice(matrice, len(mapping)):
        raise ValueError(matrix_error)
    return matrice


def _erreur_matrice(matrice: np.ndarray, nombre_lignes: int) -> str | None:
    """Retourne la première anomalie de forme, type ou finitude détectée."""
    if matrice.shape != (nombre_lignes, N_DIMS):
        return f"Matrice de forme {matrice.shape}; ({nombre_lignes}, {N_DIMS}) attendue."
    if matrice.dtype != np.float32:
        return f"Matrice de type {matrice.dtype}; float32 attendu."
    if not np.isfinite(matrice).all():
        return "La matrice contient au moins une valeur NaN ou infinie."
    return None


def _ecrire_sorties_atomiquement(
    matrice: np.ndarray, mapping: Sequence[dict[str, Any]], dossier_sortie: Path
) -> tuple[Path, Path]:
    """Publie une paire cohérente en basculant le dossier de sortie complet.

    Les deux artefacts sont préparés dans un dossier voisin avant publication.
    Le renommage du dossier évite qu'un lecteur voie un nouveau ``.npy`` avec
    l'ancien JSON. Si le processus est interrompu après le déplacement de
    l'ancien dossier, le dossier de sauvegarde est restauré au prochain appel.
    """
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.parent.mkdir(parents=True, exist_ok=True)
    dossier_precedent = dossier_sortie.with_name(f".{dossier_sortie.name}.previous")

    # Récupère un ancien dossier cohérent laissé par une interruption.
    if dossier_precedent.exists():
        if not dossier_sortie.exists():
            os.replace(dossier_precedent, dossier_sortie)
        else:
            shutil.rmtree(dossier_precedent, ignore_errors=True)

    dossier_temporaire = Path(
        tempfile.mkdtemp(prefix=f".{dossier_sortie.name}.staging-", dir=dossier_sortie.parent)
    )
    chemin_npy = dossier_sortie / "descriptors_raw.npy"
    chemin_json = dossier_sortie / "image_mapping.json"
    chemin_npy_temp = dossier_temporaire / chemin_npy.name
    chemin_json_temp = dossier_temporaire / chemin_json.name
    ancien_deplace = False
    try:
        # Préserver d'éventuels fichiers annexes déjà présents dans ce dossier.
        if dossier_sortie.exists():
            if not dossier_sortie.is_dir():
                raise NotADirectoryError(f"Le chemin de sortie n'est pas un dossier : {dossier_sortie}")
            shutil.copytree(dossier_sortie, dossier_temporaire, dirs_exist_ok=True)

        with chemin_npy_temp.open("wb") as fichier:
            np.save(fichier, matrice, allow_pickle=False)
        with chemin_json_temp.open("w", encoding="utf-8", newline="\n") as fichier:
            json.dump(list(mapping), fichier, ensure_ascii=False, indent=2)
            fichier.write("\n")

        # Les nouveaux fichiers sont tous deux prêts avant le basculement.
        if dossier_sortie.exists():
            os.replace(dossier_sortie, dossier_precedent)
            ancien_deplace = True
        try:
            os.replace(dossier_temporaire, dossier_sortie)
        except Exception:
            if ancien_deplace and dossier_precedent.exists():
                os.replace(dossier_precedent, dossier_sortie)
                ancien_deplace = False
            raise
        if ancien_deplace:
            shutil.rmtree(dossier_precedent, ignore_errors=True)
    finally:
        if dossier_temporaire.exists():
            shutil.rmtree(dossier_temporaire, ignore_errors=True)
    return chemin_npy, chemin_json


def generer_validation(
    chemin_mapping: str | Path,
    dossier_image_storage: str | Path,
    dossier_sortie: str | Path,
    images_par_categorie: int = 25,
    preprocessing_mode: str = PREPROCESSING_LETTERBOX,
) -> tuple[np.ndarray, list[dict[str, Any]], Path, Path]:
    """Extrait l'échantillon équilibré et publie sa matrice et son mapping."""
    mapping_complet = lire_mapping(chemin_mapping)
    selection = selectionner_echantillon_equilibre(mapping_complet, images_par_categorie)
    matrice = extraire_entrees(selection, dossier_image_storage, preprocessing_mode)
    if matrice.shape[0] != 500:
        raise ValueError(
            f"L'échantillon doit contenir 500 lignes (20 catégories × 25); "
            f"{matrice.shape[0]} obtenue(s)."
        )
    chemin_npy, chemin_json = _ecrire_sorties_atomiquement(matrice, selection, Path(dossier_sortie))
    return matrice, selection, chemin_npy, chemin_json
