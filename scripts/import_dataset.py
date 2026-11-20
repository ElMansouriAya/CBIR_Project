"""Importe les métadonnées GHIM. Lancer avec python -m scripts.import_dataset."""

import argparse
import json
from pathlib import Path

from flask import Flask
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.config import Config
from backend.dataset import resolve_dataset_jpeg
from backend.extensions import db
from backend.models import Image


def load_images(mapping_path, dataset_folder):
    """Valide le mapping et les JPEG avant toute écriture en base."""
    with mapping_path.open(encoding="utf-8-sig") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("le mapping doit être une liste JSON non vide")

    images = []
    seen_ids = set()
    required = {"index", "image_id", "filename", "category", "path"}
    for index, entry in enumerate(mapping):
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(f"entrée {index} : les champs {', '.join(sorted(required))} sont requis")
        if type(entry["index"]) is not int or entry["index"] != index:
            raise ValueError(f"entrée {index} : index attendu {index}")

        image_id = entry["image_id"]
        if type(image_id) is not int or not 0 < image_id < 2**63:
            raise ValueError(f"entrée {index} : image_id doit être un entier positif compatible SQLite")
        if image_id in seen_ids:
            raise ValueError(f"image_id {image_id} en double dans le mapping")
        seen_ids.add(image_id)

        filename, category, storage_path = entry["filename"], entry["category"], entry["path"]
        if not all(isinstance(value, str) and value.strip()
                   for value in (filename, category, storage_path)):
            raise ValueError(f"image_id {image_id} : filename, category et path doivent être renseignés")
        if not storage_path.startswith("dataset/"):
            raise ValueError(f"image_id {image_id} : path doit commencer par dataset/")
        relative_path = storage_path.removeprefix("dataset/")
        if relative_path.split("/")[-1] != filename:
            raise ValueError(f"image_id {image_id} : filename ne correspond pas à path")
        if len(filename) > 255 or len(category) > 50 or len(relative_path) > 500:
            raise ValueError(f"image_id {image_id} : filename, category ou path trop long")

        try:
            path = resolve_dataset_jpeg(dataset_folder, relative_path)
            with PILImage.open(path) as jpeg:
                if jpeg.format != "JPEG":
                    raise ValueError("le fichier n'est pas un JPEG")
                jpeg.load()
        except (OSError, RuntimeError, ValueError, PILImage.DecompressionBombError) as exc:
            raise ValueError(f"image_id {image_id}, {storage_path} : {exc}") from exc

        images.append(Image(id=image_id, filename=filename, filepath=relative_path,
                            dataset_category=category, is_dataset=True, user_id=None))
    return images


def import_images(images):
    """Insère les nouvelles lignes en une transaction et conserve les lignes identiques."""
    ids = [image.id for image in images]
    new_images = []
    with db.session.begin():
        if db.engine.dialect.name == "sqlite":
            # Sérialise le contrôle des IDs et l'insertion avec les autres écritures SQLite.
            db.session.execute(text("BEGIN IMMEDIATE"))

        existing = {}
        for start in range(0, len(ids), 500):
            query = db.select(Image).where(Image.id.in_(ids[start:start + 500]))
            batch = db.session.scalars(query)
            existing.update((image.id, image) for image in batch)

        for image in images:
            current = existing.get(image.id)
            if current is None:
                new_images.append(image)
            elif (
                not current.is_dataset
                or current.user_id is not None
                or current.category_id is not None
                or current.parent_image_id is not None
                or current.filename != image.filename
                or current.filepath != image.filepath
                or current.dataset_category != image.dataset_category
            ):
                raise ValueError(f"image_id {image.id} est déjà attribué à une autre image")
        db.session.add_all(new_images)
    return len(new_images)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Importer le dataset GHIM dans VisualVault.")
    parser.add_argument(
        "--mapping", type=Path, default=Path(Config.FEATURES_FOLDER) / "image_mapping.json",
        help="chemin du mapping M2 (par défaut dans FEATURES_FOLDER)",
    )
    args = parser.parse_args(argv)

    try:
        images = load_images(args.mapping, Config.DATASET_FOLDER)
        # Le script utilise la même configuration SQLite que Flask sans charger le moteur CBIR.
        app = Flask(__name__)
        app.config.from_object(Config)
        db.init_app(app)
        with app.app_context():
            db.create_all()
            inserted = import_images(images)
    except (OSError, ValueError, SQLAlchemyError) as exc:
        parser.exit(1, f"Import GHIM refusé : {exc}\n")
    print(f"Import GHIM terminé. {inserted} image(s) ajoutée(s), "
          f"{len(images) - inserted} déjà présente(s).")


if __name__ == "__main__":
    main()
