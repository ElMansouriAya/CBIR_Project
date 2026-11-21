"""Descripteurs de texture GLCM et LBP uniforme."""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from feature_extraction.config import (
    GLCM_ANGLES,
    GLCM_DISTANCE,
    GLCM_LEVELS,
    GLCM_PROPERTIES,
    LBP_BINS,
    LBP_POINTS,
    LBP_RADIUS,
)


def descripteurs_glcm(niveaux_gris_quantifies: np.ndarray) -> np.ndarray:
    """Retourne 16 valeurs : chaque propriété suivie de ses quatre angles.

    La matrice est symétrique et normalisée. Une propriété non finie (par
    exemple une corrélation indéfinie) est remplacée par zéro.
    """
    gris = np.asarray(niveaux_gris_quantifies, dtype=np.uint8)
    if gris.ndim != 2 or gris.size == 0:
        raise ValueError("La GLCM attend une image 2D non vide.")

    matrice = graycomatrix(
        gris,
        distances=[GLCM_DISTANCE],
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    valeurs: list[float] = []
    for propriete in GLCM_PROPERTIES:
        mesures = np.asarray(graycoprops(matrice, propriete), dtype=np.float64).reshape(-1)
        if propriete == "correlation":
            # La bibliothèque renvoie 1 par convention si l'écart-type d'une
            # marginale est nul. Mathématiquement la corrélation est alors
            # indéfinie; le contrat exige zéro. On vérifie chaque angle, car
            # une seule direction peut être dégénérée dans une image non uniforme.
            indices_niveaux = np.arange(GLCM_LEVELS, dtype=np.float64)
            for indice_angle in range(len(GLCM_ANGLES)):
                probabilites = matrice[:, :, 0, indice_angle]
                marginale_i = probabilites.sum(axis=1)
                marginale_j = probabilites.sum(axis=0)
                moyenne_i = np.dot(indices_niveaux, marginale_i)
                moyenne_j = np.dot(indices_niveaux, marginale_j)
                variance_i = np.dot((indices_niveaux - moyenne_i) ** 2, marginale_i)
                variance_j = np.dot((indices_niveaux - moyenne_j) ** 2, marginale_j)
                if variance_i == 0.0 or variance_j == 0.0:
                    mesures[indice_angle] = 0.0
        if not np.isfinite(mesures).all():
            raise FloatingPointError(
                f"La propriété GLCM {propriete!r} a produit une valeur non finie."
            )
        valeurs.extend(mesures.tolist())
    resultat = np.asarray(valeurs, dtype=np.float32)
    if not np.isfinite(resultat).all():
        raise FloatingPointError("La conversion des propriétés GLCM en float32 a débordé.")
    return resultat


def histogramme_lbp_uniforme(niveaux_gris_uint8: np.ndarray) -> np.ndarray:
    """Calcule l'histogramme uniforme LBP à 10 cases, normalisé à somme 1."""
    codes = local_binary_pattern(
        np.asarray(niveaux_gris_uint8, dtype=np.uint8),
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )
    histogramme, _ = np.histogram(
        codes,
        bins=LBP_BINS,
        range=(-0.5, LBP_BINS - 0.5),
    )
    total = histogramme.sum()
    if total > 0:
        histogramme = histogramme / total
    return histogramme.astype(np.float32)
