"""Descripteurs de couleur calculés à partir de l'image HSV/RGB commune."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv

from feature_extraction.config import (
    HSV_BINS,
    KMEANS_MAX_ITER,
    KMEANS_MAX_SAMPLES,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    N_DOMINANT_COLORS,
)


def histogramme_hsv(hsv: np.ndarray) -> np.ndarray:
    """Calcule l'histogramme H×S×V de 128 cases, normalisé à somme égale à 1."""
    valeurs = np.asarray(hsv, dtype=np.float64).reshape(-1, 3)
    histogramme, _ = np.histogramdd(
        valeurs,
        bins=HSV_BINS,
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    total = histogramme.sum()
    if total > 0:
        histogramme /= total
    return histogramme.reshape(-1).astype(np.float32)


def moments_couleur(hsv: np.ndarray) -> np.ndarray:
    """Retourne moyenne, écart-type et asymétrie pour H, puis S, puis V.

    L'écart-type est la définition population (ddof=0). Pour un canal
    constant, l'asymétrie est fixée à zéro, car sa division n'est pas définie.
    """
    valeurs = np.asarray(hsv, dtype=np.float64).reshape(-1, 3)
    resultat: list[float] = []
    epsilon = 1e-12
    for indice_canal in range(3):
        canal = valeurs[:, indice_canal]
        moyenne = float(np.mean(canal))
        centre = canal - moyenne
        ecart_type = float(np.sqrt(np.mean(centre * centre)))
        asymetrie = (
            float(np.mean(centre * centre * centre) / (ecart_type ** 3))
            if ecart_type > epsilon
            else 0.0
        )
        resultat.extend((moyenne, ecart_type, asymetrie))
    return np.asarray(resultat, dtype=np.float32)


def couleurs_dominantes(rgb: np.ndarray) -> np.ndarray:
    """Ajuste K-Means dans RGB, convertit les centres en HSV et les ordonne.

    Un échantillon uniforme déterminé par la graine 42 limite le coût si
    l'image contient plus de 10 000 pixels. Les tailles des clusters sont
    ensuite comptées sur tous les pixels prétraités, pas seulement l'échantillon.
    À tailles égales, l'ordre stable des indices K-Means départage les centres.
    """
    pixels = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    nombre_pixels = len(pixels)
    # K-Means ne peut pas séparer trois groupes dans une couleur uniforme.
    # Trois centres égaux préservent les 9 dimensions sans émettre de faux
    # avertissement de convergence; deux centres portent une taille nulle.
    if np.all(pixels == pixels[0]):
        couleur_hsv = rgb2hsv(pixels[0].reshape(1, 1, 3))[0, 0]
        return np.tile(couleur_hsv, N_DOMINANT_COLORS).astype(np.float32)

    nombre_echantillon = min(nombre_pixels, KMEANS_MAX_SAMPLES)
    if nombre_pixels > nombre_echantillon:
        generateur = np.random.default_rng(KMEANS_RANDOM_STATE)
        indices = generateur.choice(nombre_pixels, size=nombre_echantillon, replace=False)
        echantillon = pixels[indices]
    else:
        echantillon = pixels

    modele = KMeans(
        n_clusters=N_DOMINANT_COLORS,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        algorithm="lloyd",
    )
    modele.fit(echantillon)
    etiquettes = modele.predict(pixels)
    tailles = np.bincount(etiquettes, minlength=N_DOMINANT_COLORS)
    ordre = np.argsort(-tailles, kind="stable")

    centres_rgb = np.clip(modele.cluster_centers_[ordre], 0.0, 1.0)
    centres_hsv = rgb2hsv(centres_rgb[:, np.newaxis, :])[:, 0, :]
    centres_hsv = np.clip(centres_hsv, 0.0, 1.0)
    return centres_hsv.reshape(-1).astype(np.float32)
