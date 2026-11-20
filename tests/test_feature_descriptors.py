"""Tests unitaires des cinq blocs de caractéristiques et de leur assemblage."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

from feature_extraction.color import couleurs_dominantes, histogramme_hsv, moments_couleur
from feature_extraction.config import GLCM_ANGLES, GLCM_LEVELS, GLCM_PROPERTIES
from feature_extraction.descriptors import assembler_descripteurs, extract_raw_vector
from feature_extraction import texture
from feature_extraction.pipeline import ImageExtractionError, extraire_entrees
from feature_extraction.preprocessing import preprocess_image
from feature_extraction.texture import descripteurs_glcm, histogramme_lbp_uniforme


def creer_image_test(chemin, taille=(40, 28)) -> None:
    """Crée une image test avec plusieurs couleurs et un motif de texture."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((taille[1], taille[0]))
    tableau = np.stack(
        ((xx * 7) % 256, (yy * 11) % 256, ((xx + yy) * 5) % 256), axis=2
    ).astype(np.uint8)
    Image.fromarray(tableau, mode="RGB").save(chemin)


def test_tailles_sommes_et_bornes_des_blocs(tmp_path) -> None:
    chemin = tmp_path / "motif.png"
    creer_image_test(chemin)
    image = preprocess_image(chemin)

    bloc_hsv = histogramme_hsv(image.hsv)
    bloc_moments = moments_couleur(image.hsv)
    bloc_dominantes = couleurs_dominantes(image.rgb)
    bloc_glcm = descripteurs_glcm(image.glcm_gray)
    bloc_lbp = histogramme_lbp_uniforme(image.lbp_gray)

    assert bloc_hsv.shape == (128,)
    assert bloc_moments.shape == (9,)
    assert bloc_dominantes.shape == (9,)
    assert bloc_glcm.shape == (16,)
    assert bloc_lbp.shape == (10,)
    assert np.isclose(bloc_hsv.sum(), 1.0)
    assert np.isclose(bloc_lbp.sum(), 1.0)
    assert np.all((image.hsv >= 0.0) & (image.hsv <= 1.0))
    assert np.all((bloc_dominantes >= 0.0) & (bloc_dominantes <= 1.0))


def test_modes_pretraitement_et_mode_initial_inchange(tmp_path) -> None:
    """Le mode par défaut reste letterbox; stretch remplit le cadre sans marges."""
    chemin = tmp_path / "paysage.png"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 20), (200, 40, 20)).save(chemin)

    mode_par_defaut = preprocess_image(chemin)
    mode_letterbox = preprocess_image(chemin, preprocessing_mode="letterbox")
    mode_stretch = preprocess_image(chemin, preprocessing_mode="stretch")

    np.testing.assert_array_equal(mode_par_defaut.rgb, mode_letterbox.rgb)
    np.testing.assert_array_equal(mode_par_defaut.hsv, mode_letterbox.hsv)
    assert mode_stretch.rgb.shape == (256, 256, 3)
    # Cette image paysage uniforme doit remplir tout le cadre en mode étiré.
    attendu = np.asarray((200, 40, 20), dtype=np.float32) / 255.0
    assert np.allclose(mode_stretch.rgb, attendu[None, None, :], atol=1e-7)
    assert np.any(np.all(mode_letterbox.rgb == 0.0, axis=2))
    with pytest.raises(ValueError, match="Mode de prétraitement inconnu"):
        preprocess_image(chemin, preprocessing_mode="autre")


def test_moments_stables_sur_image_uniforme(tmp_path) -> None:
    chemin = tmp_path / "uniforme.png"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), (90, 90, 90)).save(chemin)

    moments = moments_couleur(preprocess_image(chemin).hsv)

    assert moments.shape == (9,)
    assert np.isfinite(moments).all()
    assert np.all(moments[[2, 5, 8]] == 0.0)


def test_glcm_est_ordonnee_par_propriete_puis_angle_et_stable_sur_uniforme() -> None:
    yy, xx = np.indices((24, 24))
    gris = ((xx + 2 * yy) % GLCM_LEVELS).astype(np.uint8)
    glcm = graycomatrix(
        gris, distances=[1], angles=list(GLCM_ANGLES), levels=GLCM_LEVELS,
        symmetric=True, normed=True,
    )
    attendu = np.concatenate(
        [graycoprops(glcm, propriete).reshape(-1) for propriete in GLCM_PROPERTIES]
    ).astype(np.float32)

    valeurs = descripteurs_glcm(gris)
    uniformes = descripteurs_glcm(np.zeros((24, 24), dtype=np.uint8))

    assert valeurs.shape == (16,)
    np.testing.assert_allclose(valeurs, attendu)
    assert np.isfinite(uniformes).all()
    np.testing.assert_array_equal(uniformes[:4], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(uniformes[4:8], np.zeros(4, dtype=np.float32))
    assert np.isfinite(uniformes[8:]).all()


@pytest.mark.parametrize("valeur_non_finie", [np.nan, np.inf, -np.inf])
def test_glcm_refuse_une_valeur_non_finie_inattendue(monkeypatch, valeur_non_finie) -> None:
    monkeypatch.setattr(
        texture,
        "graycoprops",
        lambda _matrice, _propriete: np.array([[valeur_non_finie, 0.0, 0.0, 0.0]]),
    )
    with pytest.raises(FloatingPointError, match="non finie"):
        texture.descripteurs_glcm(np.zeros((8, 8), dtype=np.uint8))


def test_kmeans_est_reproductible_et_supporte_une_image_uniforme(tmp_path) -> None:
    chemin = tmp_path / "couleurs.png"
    creer_image_test(chemin)
    rgb = preprocess_image(chemin).rgb

    etat_global_avant = np.random.get_state()
    premier = couleurs_dominantes(rgb)
    etat_global_apres = np.random.get_state()
    second = couleurs_dominantes(rgb)
    uniforme = np.full((16, 16, 3), 0.5, dtype=np.float32)
    centres_uniformes = couleurs_dominantes(uniforme)

    np.testing.assert_array_equal(premier, second)
    assert etat_global_avant[0] == etat_global_apres[0]
    np.testing.assert_array_equal(etat_global_avant[1], etat_global_apres[1])
    assert etat_global_avant[2:] == etat_global_apres[2:]
    assert premier.shape == (9,)
    assert np.all((premier >= 0.0) & (premier <= 1.0))
    assert centres_uniformes.shape == (9,)
    assert np.isfinite(centres_uniformes).all()


def test_vecteur_final_est_float32_fini_et_de_taille_172(tmp_path) -> None:
    chemin = tmp_path / "final.png"
    creer_image_test(chemin)

    vecteur = extract_raw_vector(chemin)

    assert vecteur.shape == (172,)
    assert vecteur.dtype == np.float32
    assert np.isfinite(vecteur).all()


@pytest.mark.parametrize("valeur_non_finie", [np.nan, np.inf, -np.inf])
def test_assemblage_signale_nan_ou_inf_inattendu(valeur_non_finie) -> None:
    with pytest.raises(FloatingPointError, match="bloc 'hsv_hist'"):
        assembler_descripteurs(
            np.r_[valeur_non_finie, np.zeros(127, dtype=np.float32)],
            np.zeros(9), np.zeros(9), np.zeros(16), np.zeros(10),
        )


def test_assemblage_refuse_un_bloc_de_mauvaise_taille() -> None:
    with pytest.raises(ValueError, match="taille"):
        assembler_descripteurs(np.zeros(127), np.zeros(9), np.zeros(9), np.zeros(16), np.zeros(10))


def test_pipeline_preserve_ordre_et_signale_l_image_illisible(tmp_path) -> None:
    racine = tmp_path / "image_storage"
    premier = racine / "dataset" / "A" / "a.png"
    second = racine / "dataset" / "B" / "b.png"
    creer_image_test(premier)
    creer_image_test(second)
    mapping = [
        {"index": 0, "image_id": 71, "filename": "a.png", "category": "A", "path": "dataset/A/a.png"},
        {"index": 1, "image_id": 93, "filename": "b.png", "category": "B", "path": "dataset/B/b.png"},
    ]

    matrice = extraire_entrees(mapping, racine)
    np.testing.assert_array_equal(matrice[0], extract_raw_vector(premier))
    np.testing.assert_array_equal(matrice[1], extract_raw_vector(second))

    mapping[1] = {**mapping[1], "path": "dataset/B/missing.png"}
    with pytest.raises(ImageExtractionError, match="index=1, image_id=93"):
        extraire_entrees(mapping, racine)
