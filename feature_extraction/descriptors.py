"""Assemblage du vecteur brut de 172 dimensions pour une seule image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extraction.color import couleurs_dominantes, histogramme_hsv, moments_couleur
from feature_extraction.config import BLOCK_SLICES, N_DIMS, PREPROCESSING_LETTERBOX
from feature_extraction.preprocessing import preprocess_image
from feature_extraction.texture import descripteurs_glcm, histogramme_lbp_uniforme


def assembler_descripteurs(
    bloc_histogramme_hsv: np.ndarray,
    bloc_moments: np.ndarray,
    bloc_couleurs_dominantes: np.ndarray,
    bloc_glcm: np.ndarray,
    bloc_lbp: np.ndarray,
) -> np.ndarray:
    """Concatène les cinq blocs dans l'ordre et les plages du contrat."""
    blocs = (
        ("hsv_hist", bloc_histogramme_hsv),
        ("color_moments", bloc_moments),
        ("dominant_colors", bloc_couleurs_dominantes),
        ("glcm", bloc_glcm),
        ("lbp", bloc_lbp),
    )
    valeurs: list[np.ndarray] = []
    for nom, bloc in blocs:
        tableau = np.asarray(bloc).reshape(-1)
        taille_attendue = BLOCK_SLICES[nom].stop - BLOCK_SLICES[nom].start
        if tableau.size != taille_attendue:
            raise ValueError(
                f"Bloc {nom!r} de taille {tableau.size}; {taille_attendue} attendue(s)."
            )
        if not np.isfinite(tableau).all():
            raise FloatingPointError(
                f"Le bloc {nom!r} contient une valeur NaN ou infinie inattendue."
            )
        valeurs.append(tableau)

    vecteur = np.concatenate(valeurs).astype(np.float32, copy=False)
    if vecteur.size != N_DIMS:
        raise ValueError(f"Vecteur de taille {vecteur.size}; {N_DIMS} attendue(s).")
    if not np.isfinite(vecteur).all():
        raise FloatingPointError("Le vecteur final contient une valeur non finie après conversion float32.")
    return vecteur


def extract_raw_vector(
    image_path: str | Path,
    preprocessing_mode: str = PREPROCESSING_LETTERBOX,
) -> np.ndarray:
    """Extrait les 172 valeurs brutes d'une image sans normalisation Z-score.

    Cette fonction commune peut être appelée par le pipeline de lot ou plus
    tard par une route d'API pour une image uploadée.
    """
    image = preprocess_image(image_path, preprocessing_mode=preprocessing_mode)
    return assembler_descripteurs(
        histogramme_hsv(image.hsv),
        moments_couleur(image.hsv),
        couleurs_dominantes(image.rgb),
        descripteurs_glcm(image.glcm_gray),
        histogramme_lbp_uniforme(image.lbp_gray),
    )
