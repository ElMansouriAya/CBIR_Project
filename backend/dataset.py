from pathlib import Path


def resolve_dataset_jpeg(dataset_folder, relative_path):
    """Résout un chemin de JPEG stocké relativement au dossier dataset."""
    parts = relative_path.split("/")
    if (
        "\\" in relative_path
        or ":" in relative_path
        or any(ord(char) < 32 for char in relative_path)
        or any(part in ("", ".", "..") for part in parts)
        or Path(parts[-1]).suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise ValueError("chemin JPEG relatif au dataset attendu")

    root = Path(dataset_folder).resolve()
    path = root.joinpath(*parts).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("le chemin doit désigner un fichier dans le dataset")
    return path
