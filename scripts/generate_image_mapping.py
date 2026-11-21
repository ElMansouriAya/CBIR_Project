"""Génère un mapping JSON déterministe des images du dataset GHIM-10K.

Le script ne calcule aucun descripteur et ne communique pas avec la base de
données. Il recense les images valides sous ``image_storage/dataset`` et
attribue les identifiants demandés par le membre A : ``image_id = index + 1``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def est_image_valide(chemin: Path) -> bool:
    """Vérifie l'extension puis vérifie que Pillow peut décoder le fichier.

    ``verify`` contrôle l'intégrité structurelle sans charger toute l'image en
    mémoire. Une seconde ouverture avec ``load`` force le décodage des pixels.
    """
    if chemin.suffix.lower() not in EXTENSIONS_IMAGE or not chemin.is_file():
        return False
    try:
        with Image.open(chemin) as image:
            image.verify()
        with Image.open(chemin) as image:
            image.load()
        return True
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return False


def collecter_images(dossier_dataset: Path) -> list[tuple[str, str, Path]]:
    """Retourne les images valides sous la forme (catégorie, nom, chemin).

    La catégorie est le premier sous-dossier sous le dossier du dataset. Les
    chemins plus profonds restent supportés et servent de critère de départage
    déterministe si deux fichiers ont même catégorie et même nom.
    Les fichiers directement à la racine sont ignorés car ils n'ont pas de
    catégorie attestée par un sous-dossier.
    """
    racine = dossier_dataset.resolve()
    if not racine.is_dir():
        raise FileNotFoundError(f"Dossier dataset introuvable : {dossier_dataset}")

    images: list[tuple[str, str, Path]] = []
    for chemin in racine.rglob("*"):
        if not chemin.is_file():
            continue
        relatif = chemin.relative_to(racine)
        if len(relatif.parts) < 2:
            continue
        if est_image_valide(chemin):
            images.append((relatif.parts[0], chemin.name, chemin))

    # casefold donne un ordre alphabétique insensible à la casse; les valeurs
    # originales départagent les égalités de façon stable entre exécutions.
    images.sort(
        key=lambda element: (
            element[0].casefold(), element[0], element[1].casefold(), element[1],
            element[2].relative_to(racine).as_posix().casefold(),
            element[2].relative_to(racine).as_posix(),
        )
    )
    return images


def construire_mapping(
    dossier_dataset: Path, nombre: int, dossier_image_storage: Path | None = None
) -> list[dict[str, Any]]:
    """Construit les ``nombre`` premières entrées du mapping.

    Les chemins sont relatifs à ``image_storage`` et utilisent toujours ``/``.
    Une erreur explicite est levée si le dataset ne contient pas assez d'images
    valides : le script ne complète jamais le mapping avec des entrées fictives.
    """
    if nombre <= 0:
        raise ValueError("Le nombre d'images doit être strictement positif.")

    dossier_dataset = dossier_dataset.resolve()
    base_storage = (dossier_image_storage or dossier_dataset.parent).resolve()
    images = collecter_images(dossier_dataset)
    if len(images) < nombre:
        raise ValueError(
            f"Seulement {len(images)} image(s) valide(s) trouvée(s), "
            f"mais {nombre} demandée(s). Aucun mapping n'a été écrit."
        )

    mapping: list[dict[str, Any]] = []
    vus: set[str] = set()
    for index, (categorie, nom, chemin) in enumerate(images[:nombre]):
        if not chemin.is_file():
            raise FileNotFoundError(f"Fichier disparu pendant le traitement : {chemin}")
        relatif_storage = chemin.resolve().relative_to(base_storage).as_posix()
        cle = relatif_storage.casefold()
        if cle in vus:
            raise ValueError(f"Fichier image en doublon dans le parcours : {relatif_storage}")
        vus.add(cle)
        mapping.append(
            {
                "index": index,
                "image_id": index + 1,
                "filename": nom,
                "category": categorie,
                "path": relatif_storage,
            }
        )
    return mapping


def ecrire_mapping(mapping: list[dict[str, Any]], fichier_sortie: Path) -> None:
    """Écrit un JSON UTF-8 lisible en remplaçant atomiquement l'ancien fichier."""
    fichier_sortie.parent.mkdir(parents=True, exist_ok=True)
    contenu = json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    temporaire: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=fichier_sortie.parent,
            prefix=f".{fichier_sortie.name}.", suffix=".tmp", delete=False,
        ) as fichier:
            temporaire = fichier.name
            fichier.write(contenu)
        os.replace(temporaire, fichier_sortie)
    finally:
        if temporaire and os.path.exists(temporaire):
            os.unlink(temporaire)


def main() -> int:
    """Point d'entrée de la commande en ligne."""
    racine_projet = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", type=int, default=10000,
        help="Nombre d'images à inclure (défaut : 10 000).",
    )
    parser.add_argument(
        "--dataset", type=Path, default=racine_projet / "image_storage" / "dataset",
        help="Dossier du dataset (défaut : image_storage/dataset).",
    )
    parser.add_argument(
        "--output", type=Path, default=racine_projet / "features" / "image_mapping.json",
        help="Fichier JSON de sortie.",
    )
    args = parser.parse_args()
    try:
        mapping = construire_mapping(args.dataset, args.count, args.dataset.parent)
        ecrire_mapping(mapping, args.output)
    except (FileNotFoundError, ValueError, OSError) as erreur:
        parser.error(str(erreur))
    print(f"Mapping créé : {args.output} ({len(mapping)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
