"""Descripteurs GHIM issus de la matrice brute de M2."""

from colorsys import hsv_to_rgb
import logging
from math import isclose

from backend.cbir_config import BLOCKS

log = logging.getLogger(__name__)


def load_descriptor_catalog(features_dir):
    """Indexe les lignes du lot par image_id, indépendamment de leur index."""
    try:
        import numpy as np

        from backend.services.loader import load_raw, validate_contract

        matrix, mapping = load_raw(features_dir)
        if not isinstance(matrix, np.ndarray) or matrix.dtype.kind != "f":
            raise ValueError("matrice de nombres flottants attendue pour les descripteurs")
        if not isinstance(mapping, list) or not all(isinstance(entry, dict) for entry in mapping):
            raise ValueError("liste d'objets attendue pour le mapping GHIM")
        validate_contract(matrix, mapping)

        catalog = {}
        for entry, vector in zip(mapping, matrix):
            image_id = entry["image_id"]
            if type(image_id) is not int or image_id <= 0 or image_id in catalog:
                raise ValueError(f"image_id invalide ou dupliqué : {image_id}")
            catalog[image_id] = (entry, vector)
        return catalog
    except FileNotFoundError:
        return {}
    except (ImportError, OSError, TypeError, ValueError) as exc:
        log.warning("Catalogue de descripteurs GHIM indisponible : %s", exc)
        return {}


def image_descriptors(catalog, image):
    """Prépare les cinq blocs d'affichage si la ligne correspond à l'image importée."""
    match = catalog.get(image.id)
    if match is None:
        return None

    entry, vector = match
    if (entry.get("filename"), entry.get("category"), entry.get("path")) != (
        image.filename, image.dataset_category, f"dataset/{image.filepath}"
    ):
        return None

    blocks = {name: vector[start:end] for name, (start, end) in BLOCKS.items()}
    hsv_hist = blocks["hsv_hist"]
    hsv_values = hsv_hist.tolist()
    lbp = blocks["lbp"].tolist()
    palette = blocks["dominant_colors"].reshape(3, 3).tolist()
    for probabilities in (hsv_values, lbp):
        if min(probabilities) < 0 or not isclose(sum(probabilities), 1, abs_tol=1e-3):
            return None
    if not all(0 <= channel <= 1 for hsv in palette for channel in hsv):
        return None

    return {
        "histogram": hsv_hist.reshape(8, 16).tolist(),
        "histogram_peak": max(hsv_values),
        "moments": blocks["color_moments"].reshape(3, 3).tolist(),
        "dominant_colors": [
            {"hsv": hsv, "hex": "#" + "".join(f"{round(channel * 255):02x}"
                                             for channel in hsv_to_rgb(*hsv))}
            for hsv in palette
        ],
        "glcm": blocks["glcm"].reshape(4, 4).tolist(),
        "lbp": lbp,
    }
