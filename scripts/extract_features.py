"""Extrait le jeu de validation équilibré de 500 images GHIM-10K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RACINE_PROJET = Path(__file__).resolve().parents[1]
if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))

from feature_extraction.config import PREPROCESSING_LETTERBOX, PREPROCESSING_MODES, PREPROCESSING_STRETCH
from feature_extraction.pipeline import ImageExtractionError, MappingError, generer_complet, generer_validation


def main() -> int:
    """Point d'entrée de la première extraction, limitée à 25 images/catégorie."""
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--mode", choices=("validation", "full"), default="validation",
                         help="validation extrait 500 images; full extrait les 10 000 images.")
    parseur.add_argument(
        "--mapping", type=Path, default=RACINE_PROJET / "features" / "image_mapping.json",
        help="Mapping complet servant de source d'ordre.",
    )
    parseur.add_argument(
        "--image-storage", type=Path, default=RACINE_PROJET / "image_storage",
        help="Racine à laquelle les chemins relatifs du mapping se rapportent.",
    )
    parseur.add_argument(
        "--preprocessing-mode", choices=PREPROCESSING_MODES, default=PREPROCESSING_LETTERBOX,
        help="letterbox conserve le ratio; stretch remplit le cadre en déformant le ratio.",
    )
    parseur.add_argument(
        "--output-dir", type=Path, default=None,
        help="Dossier de sortie (par défaut, dépend du mode de prétraitement).",
    )
    args = parseur.parse_args()
    dossier_sortie = args.output_dir
    if dossier_sortie is None:
        nom_dossier = "validation_stretched" if args.preprocessing_mode == PREPROCESSING_STRETCH else "validation"
        dossier_sortie = RACINE_PROJET / "features" / nom_dossier
    try:
        if args.mode == "full":
            chemin_final = RACINE_PROJET / "features" / "descriptors_raw.npy"
            matrice, mapping, chemin_npy = generer_complet(
                args.mapping, args.image_storage, chemin_final,
                preprocessing_mode=args.preprocessing_mode,
            )
            chemin_json = args.mapping
        else:
            matrice, mapping, chemin_npy, chemin_json = generer_validation(
                args.mapping,
                args.image_storage,
                dossier_sortie,
                images_par_categorie=25,
                preprocessing_mode=args.preprocessing_mode,
            )
    except (MappingError, ImageExtractionError, OSError, ValueError) as erreur:
        parseur.error(str(erreur))

    print(f"Extraction terminée : {matrice.shape[0]} images, {matrice.shape[1]} dimensions.")
    print(f"Matrice : {chemin_npy}")
    print(f"Mapping : {chemin_json}")
    print(f"Mode de prétraitement : {args.preprocessing_mode}")
    if args.mode == "validation":
        print("Les index du mapping de validation sont locaux; les image_id restent globaux.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
