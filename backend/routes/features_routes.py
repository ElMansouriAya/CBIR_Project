"""Protected API for extracting and reading raw image descriptors."""

from __future__ import annotations

import functools
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from flask import Blueprint, request
from flask_login import current_user
from flask_restful import Api, Resource

from backend import cbir_config
from backend.services.loader import load_raw, validate_contract
from feature_extraction.config import GLCM_PROPERTIES, PREPROCESSING_STRETCH
from feature_extraction.descriptors import extract_raw_vector
from feature_extraction.preprocessing import ImageReadError

log = logging.getLogger(__name__)

MAX_BATCH_SIZE = 100
DEFAULT_PERSONAL_CACHE_SIZE = 1024


class FeaturesApiError(Exception):
    """An expected API error with an HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class _PersonalFeatureCache:
    """Bounded process-local LRU cache; entries are bound to a file signature."""

    def __init__(self, max_entries: int):
        if max_entries <= 0:
            raise ValueError("max_entries doit être strictement positif.")
        self.max_entries = max_entries
        self._entries: OrderedDict[int, tuple[tuple[str, int, int], np.ndarray]] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, image_id: int, signature: tuple[str, int, int], vector: np.ndarray) -> None:
        with self._lock:
            self._entries[image_id] = (signature, vector.copy())
            self._entries.move_to_end(image_id)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get(self, image_id: int, signature: tuple[str, int, int]) -> np.ndarray | None:
        with self._lock:
            cached = self._entries.get(image_id)
            if cached is None:
                return None
            cached_signature, vector = cached
            if cached_signature != signature:
                self._entries.pop(image_id, None)
                return None
            self._entries.move_to_end(image_id)
            return vector.copy()


def _api_errors(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except FeaturesApiError as error:
            return {"error": error.message}, error.status
        except Exception:
            log.exception("Unexpected error in feature API")
            return {"error": "Erreur interne lors du traitement des descripteurs."}, 500

    return wrapper


def _json_object() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise FeaturesApiError(400, "Corps JSON (objet) attendu.")
    return data


def _valid_image_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _features_login_required(view):
    """Use API-local JSON 401 handling without changing Flask-Login globally."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            return {"error": "Authentification requise"}, 401
        return view(*args, **kwargs)

    return wrapped


def _validated_vector(vector) -> np.ndarray:
    result = np.asarray(vector)
    if result.shape != (cbir_config.N_DIMS,) or result.dtype != np.float32:
        raise FeaturesApiError(422, "Le descripteur doit contenir 172 valeurs float32.")
    if not np.isfinite(result).all():
        raise FeaturesApiError(422, "Le descripteur contient une valeur NaN ou infinie.")
    return result


def _json_vector(image_id: int, vector: np.ndarray) -> dict:
    """Expose les blocs bruts en respectant l'ordre du contrat B-C."""
    hsv_start, hsv_end = cbir_config.BLOCKS["hsv_hist"]
    moments_start, moments_end = cbir_config.BLOCKS["color_moments"]
    dominant_start, dominant_end = cbir_config.BLOCKS["dominant_colors"]
    glcm_start, glcm_end = cbir_config.BLOCKS["glcm"]
    lbp_start, lbp_end = cbir_config.BLOCKS["lbp"]
    glcm = {
        name: [float(value) for value in vector[glcm_start + i * 4:glcm_start + (i + 1) * 4]]
        for i, name in enumerate(GLCM_PROPERTIES)
    }
    if glcm_start + len(GLCM_PROPERTIES) * 4 != glcm_end:
        raise RuntimeError("La configuration GLCM ne correspond pas au contrat API.")
    return {
        "image_id": image_id,
        "color": {
            "hsv_histogram": [float(value) for value in vector[hsv_start:hsv_end]],
            "color_moments": [float(value) for value in vector[moments_start:moments_end]],
            "dominant_colors": [float(value) for value in vector[dominant_start:dominant_end]],
        },
        "texture": {
            "glcm": glcm,
            "lbp": [float(value) for value in vector[lbp_start:lbp_end]],
        },
        "dimensions": cbir_config.N_DIMS,
        "dtype": "float32",
        "preprocessing_mode": PREPROCESSING_STRETCH,
    }


def _fingerprint(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def create_features_blueprint(
    get_image_path: Callable[[int], str | Path | None],
    *,
    features_dir: str | Path = cbir_config.FEATURES_DIR,
    image_storage_dir: str | Path | None = None,
    personal_cache_size: int = DEFAULT_PERSONAL_CACHE_SIZE,
    name: str = "cbir_features",
) -> Blueprint:
    """Create protected feature routes.

    `get_image_path` must enforce A's ownership/access rules and return an
    absolute image path or None. Personal-image features live in this
    blueprint instance's bounded in-memory cache; they are not persisted.
    """
    if not callable(get_image_path):
        raise TypeError("get_image_path doit être une fonction.")
    if personal_cache_size <= 0:
        raise ValueError("personal_cache_size doit être strictement positif.")

    feature_root = Path(features_dir).resolve()
    storage_root = Path(image_storage_dir).resolve() if image_storage_dir else feature_root.parent / "image_storage"
    matrix, mapping = load_raw(feature_root)
    validate_contract(matrix, mapping, cbir_config.N_DIMS)
    if matrix.dtype != np.float32:
        raise ValueError(f"Matrice brute de type {matrix.dtype}; float32 attendu.")

    mapping_by_id: dict[int, dict] = {}
    for entry in mapping:
        image_id = entry["image_id"]
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in mapping_by_id:
            raise ValueError("Le mapping contient un image_id invalide ou dupliqué.")
        mapping_by_id[image_id] = entry

    personal_cache = _PersonalFeatureCache(personal_cache_size)

    def resolve_accessible_path(image_id: int) -> Path:
        try:
            supplied_path = get_image_path(image_id)
        except Exception as error:
            log.exception("get_image_path a échoué pour image_id=%s", image_id)
            raise FeaturesApiError(500, "Erreur interne lors de la résolution de l'image.") from error
        if supplied_path is None:
            raise FeaturesApiError(404, f"Image introuvable ou inaccessible : {image_id}.")
        try:
            path = Path(supplied_path)
            if not path.is_absolute():
                raise FeaturesApiError(404, f"Image introuvable ou inaccessible : {image_id}.")
            path = path.resolve(strict=True)
            if not path.is_file():
                raise FeaturesApiError(404, f"Image introuvable ou inaccessible : {image_id}.")
            # stat here so inaccessible paths are rejected before serving cache data.
            _fingerprint(path)
        except FeaturesApiError:
            raise
        except OSError as error:
            raise FeaturesApiError(404, f"Image introuvable ou inaccessible : {image_id}.") from error
        return path

    def extract_for_path(image_id: int, path: Path) -> np.ndarray:
        try:
            vector = extract_raw_vector(path, preprocessing_mode=PREPROCESSING_STRETCH)
        except (OSError, ValueError, FloatingPointError, ImageReadError) as error:
            raise FeaturesApiError(422, f"L'image {image_id} ne peut pas être traitée.") from error
        return _validated_vector(vector)

    def remember_personal(image_id: int, path: Path, vector: np.ndarray) -> None:
        if ghim_path_matches(image_id, path):
            return
        personal_cache.put(image_id, _fingerprint(path), vector)

    def ghim_path_matches(image_id: int, path: Path) -> bool:
        entry = mapping_by_id.get(image_id)
        if entry is None:
            return False
        try:
            expected_path = (storage_root / entry["path"]).resolve(strict=True)
        except OSError:
            return False
        return expected_path == path

    def ghim_vector_if_path_matches(image_id: int, path: Path) -> np.ndarray | None:
        entry = mapping_by_id.get(image_id)
        if entry is None or not ghim_path_matches(image_id, path):
            return None
        return _validated_vector(matrix[entry["index"]])

    def remembered_personal(image_id: int, path: Path) -> np.ndarray | None:
        return personal_cache.get(image_id, _fingerprint(path))

    blueprint = Blueprint(name, __name__)
    api = Api(blueprint, decorators=[_features_login_required])

    class ExtractResource(Resource):
        @_api_errors
        def post(self):
            data = _json_object()
            image_id = data.get("image_id")
            if not _valid_image_id(image_id):
                raise FeaturesApiError(400, "image_id doit être un entier positif.")
            path = resolve_accessible_path(image_id)
            vector = extract_for_path(image_id, path)
            remember_personal(image_id, path, vector)
            return _json_vector(image_id, vector)

    class ExtractBatchResource(Resource):
        @_api_errors
        def post(self):
            data = _json_object()
            image_ids = data.get("image_ids")
            if not isinstance(image_ids, list) or not image_ids:
                raise FeaturesApiError(400, "image_ids doit être une liste non vide.")
            if len(image_ids) > MAX_BATCH_SIZE:
                raise FeaturesApiError(400, f"Un lot ne peut pas dépasser {MAX_BATCH_SIZE} identifiants.")
            if not all(_valid_image_id(image_id) for image_id in image_ids):
                raise FeaturesApiError(400, "Chaque image_id doit être un entier positif.")

            results = []
            for image_id in image_ids:
                try:
                    path = resolve_accessible_path(image_id)
                    vector = extract_for_path(image_id, path)
                    remember_personal(image_id, path, vector)
                    results.append(_json_vector(image_id, vector))
                except FeaturesApiError as error:
                    results.append({
                        "image_id": image_id,
                        "error": {"status": error.status, "message": error.message},
                    })
                except Exception:
                    log.exception("Unexpected batch extraction error for image_id=%s", image_id)
                    results.append({
                        "image_id": image_id,
                        "error": {"status": 500, "message": "Erreur interne lors de l'extraction."},
                    })
            failed = sum("error" in result for result in results)
            return {"results": results, "count": len(results), "failed": failed}

    class ReadFeatureResource(Resource):
        @_api_errors
        def get(self, image_id: int):
            if not _valid_image_id(image_id):
                raise FeaturesApiError(400, "image_id doit être un entier positif.")
            # Always resolve through A before accessing either stored or cached data.
            path = resolve_accessible_path(image_id)
            vector = ghim_vector_if_path_matches(image_id, path)
            if vector is None:
                vector = remembered_personal(image_id, path)
            if vector is None:
                raise FeaturesApiError(404, f"Aucun descripteur disponible pour l'image {image_id}.")
            return _json_vector(image_id, vector)

    api.add_resource(ExtractResource, "/features/extract")
    api.add_resource(ExtractBatchResource, "/features/extract-batch")
    api.add_resource(ReadFeatureResource, "/features/<int:image_id>")
    return blueprint
