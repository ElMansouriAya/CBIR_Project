"""Lecture et prétraitement communs aux cinq familles de descripteurs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv

from feature_extraction.config import (
    GLCM_LEVELS,
    IMAGE_SIZE,
    LETTERBOX_RGB,
    PREPROCESSING_LETTERBOX,
    PREPROCESSING_MODES,
    PREPROCESSING_STRETCH,
)


class ImageReadError(RuntimeError):
    """Erreur explicite de lecture ou de décodage d'une image."""


@dataclass(frozen=True)
class ImageData:
    """Représentations partagées d'une image après le prétraitement commun."""

    rgb: np.ndarray
    hsv: np.ndarray
    grayscale: np.ndarray
    glcm_gray: np.ndarray
    lbp_gray: np.ndarray


def _ouvrir_rgb(chemin_image: Path) -> Image.Image:
    """Ouvre l'image, applique son orientation EXIF et compose l'alpha sur noir."""
    try:
        with Image.open(chemin_image) as image_source:
            image_orientee = ImageOps.exif_transpose(image_source)
            rgba = image_orientee.convert("RGBA")
            fond = Image.new("RGBA", rgba.size, (*LETTERBOX_RGB, 255))
            fond.alpha_composite(rgba)
            return fond.convert("RGB")
    except (OSError, ValueError) as erreur:
        raise ImageReadError(f"Impossible de lire ou décoder l'image : {chemin_image}") from erreur


def _redimensionner_avec_marges(image_rgb: Image.Image) -> Image.Image:
    """Ajuste l'image dans la taille cible sans déformer son ratio.

    L'image est réduite ou agrandie pour tenir dans le cadre, avec LANCZOS.
    Le cadre restant est rempli en noir et l'image est centrée. Ce choix
    d'implémentation provisoire évite d'étirer des objets ou des scènes.
    """
    largeur_cible, hauteur_cible = IMAGE_SIZE
    largeur, hauteur = image_rgb.size
    facteur = min(largeur_cible / largeur, hauteur_cible / hauteur)
    nouvelle_taille = (
        max(1, round(largeur * facteur)),
        max(1, round(hauteur * facteur)),
    )
    reduite = image_rgb.resize(nouvelle_taille, Image.Resampling.LANCZOS)
    cadre = Image.new("RGB", IMAGE_SIZE, LETTERBOX_RGB)
    position = (
        (largeur_cible - nouvelle_taille[0]) // 2,
        (hauteur_cible - nouvelle_taille[1]) // 2,
    )
    cadre.paste(reduite, position)
    return cadre


def _redimensionner_etirer(image_rgb: Image.Image) -> Image.Image:
    """Redimensionne directement en 256×256, sans ratio conservé ni marges."""
    return image_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def preprocess_image(
    image_path: str | Path,
    preprocessing_mode: str = PREPROCESSING_LETTERBOX,
) -> ImageData:
    """Construit RGB, HSV et niveaux de gris selon le même prétraitement.

    RGB, HSV et gris sont des tableaux float32 en [0, 1]. La GLCM utilise une
    quantification explicite sur 32 niveaux; LBP reçoit une image gris uint8.
    ``letterbox`` conserve le ratio avec des marges noires; ``stretch`` étire
    directement l'image. Les autres étapes restent identiques dans les deux cas.
    """
    if preprocessing_mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Mode de prétraitement inconnu {preprocessing_mode!r}; "
            f"choisir parmi {PREPROCESSING_MODES}."
        )
    chemin = Path(image_path)
    image_source = _ouvrir_rgb(chemin)
    if preprocessing_mode == PREPROCESSING_LETTERBOX:
        image_rgb = _redimensionner_avec_marges(image_source)
    else:
        image_rgb = _redimensionner_etirer(image_source)
    rgb = np.asarray(image_rgb, dtype=np.float32) / np.float32(255.0)
    hsv = rgb2hsv(rgb).astype(np.float32, copy=False)
    grayscale = rgb2gray(rgb).astype(np.float32, copy=False)

    # Quantification uniforme : 0..31 pour la GLCM configurée à 32 niveaux.
    glcm_gray = np.floor(grayscale * GLCM_LEVELS).astype(np.uint8)
    np.clip(glcm_gray, 0, GLCM_LEVELS - 1, out=glcm_gray)
    lbp_gray = np.rint(grayscale * 255.0).astype(np.uint8)

    return ImageData(
        rgb=rgb,
        hsv=hsv,
        grayscale=grayscale,
        glcm_gray=glcm_gray,
        lbp_gray=lbp_gray,
    )
