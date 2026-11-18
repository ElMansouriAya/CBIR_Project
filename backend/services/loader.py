"""Chargement des fichiers de B et construction du moteur de recherche.

Fichiers attendus dans `features/` (contrat B <-> C, voir CONTRAT.md) :
    descriptors_raw.npy   matrice (N, d), valeurs BRUTES, sans NaN ni inf
    image_mapping.json    une entrée par ligne, dans le même ordre :
                          {"index", "image_id", "filename", "category", "path"}

Le normaliseur est recalculé à chaque chargement à partir de la matrice : il ne
peut donc jamais être périmé si B régénère ses descripteurs. Il est retourné
pour pouvoir normaliser les images externes (uploads) de la même façon.
"""
import json
from pathlib import Path

import numpy as np

from backend import cbir_config as config
from backend.services.normalization import fit_normalizer, save_normalizer, transform
from backend.services.session import SearchEngine


def validate_contract(X_raw, mapping, expected_dims=config.N_DIMS):
    """Vérifie le contrat de données ; lève ValueError avec un message précis."""
    if X_raw.ndim != 2:
        raise ValueError(f"La matrice doit être (N, d), reçu {X_raw.shape}")
    if expected_dims is not None and X_raw.shape[1] != expected_dims:
        raise ValueError(f"Dimension attendue {expected_dims}, reçue {X_raw.shape[1]}")
    if not np.isfinite(X_raw).all():
        raise ValueError("La matrice contient des NaN ou des inf")
    if len(mapping) != X_raw.shape[0]:
        raise ValueError(
            f"mapping ({len(mapping)} entrées) et matrice ({X_raw.shape[0]} lignes) "
            "de tailles différentes"
        )
    for i, entry in enumerate(mapping):
        missing = {"index", "image_id", "category"} - set(entry)
        if missing:
            raise ValueError(f"Entrée {i} du mapping : champs manquants {sorted(missing)}")
        if entry["index"] != i:
            raise ValueError(f"Ordre du mapping incorrect à la ligne {i} (index={entry['index']})")


def load_raw(features_dir=config.FEATURES_DIR,
             matrix_name="descriptors_raw.npy", mapping_name="image_mapping.json"):
    """Charge la matrice brute et le mapping (sans validation)."""
    features_dir = Path(features_dir)
    matrix_path, mapping_path = features_dir / matrix_name, features_dir / mapping_name
    for p in (matrix_path, mapping_path):
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable : {p} (livrable de B)")
    X_raw = np.load(matrix_path)
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return X_raw, mapping


def load_normalized(features_dir=config.FEATURES_DIR, expected_dims=config.N_DIMS,
                    normalizer_out=None):
    """Charge, valide et normalise les données de B.

    Retourne `(Z, mapping, normalizer)` : `Z` est la matrice normalisée (N, d).
    `expected_dims=None` désactive le contrôle de la dimension.
    `normalizer_out` : chemin .npz où sauvegarder le normaliseur (optionnel).
    """
    X_raw, mapping = load_raw(features_dir)
    validate_contract(X_raw, mapping, expected_dims)

    normalizer = fit_normalizer(X_raw)
    if normalizer_out is not None:
        save_normalizer(normalizer, normalizer_out)
    return transform(X_raw, normalizer), mapping, normalizer


def load_engine(features_dir=config.FEATURES_DIR, expected_dims=config.N_DIMS,
                normalizer_out=None, **engine_kwargs):
    """Charge les données de B et retourne `(engine, normalizer)`.

    `engine_kwargs` : transmis à SearchEngine (scope, max_iterations, ...).
    """
    Z, mapping, normalizer = load_normalized(features_dir, expected_dims, normalizer_out)
    engine = SearchEngine(
        Z,
        image_ids=[m["image_id"] for m in mapping],
        categories=[m["category"] for m in mapping],
        **engine_kwargs,
    )
    return engine, normalizer
