"""Génère un jeu de test synthétique conforme au contrat B <-> C.

Chaque catégorie a un centre propre dans l'espace des descripteurs, et les
images d'une catégorie sont ce centre plus du bruit gaussien : le moteur de
recherche doit donc pouvoir retrouver des voisins de la même catégorie.

Les blocs sont multipliés par des échelles très différentes (histogrammes
petits, GLCM grands) pour reproduire le problème que la normalisation doit
corriger.

Usage :
    python scripts/make_synthetic_data.py
    python scripts/make_synthetic_data.py --per-category 30 --noise 0.2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import cbir_config as config  # noqa: E402

CATEGORIES = [
    "Animals", "Architecture", "Beach", "Buildings", "Cars",
    "Flowers", "Food", "Mountains", "Sunsets", "Trees",
]

# Échelle brute par bloc, pour simuler des ordres de grandeur différents
BLOCK_SCALES = {
    "hsv_hist": 0.01,
    "color_moments": 1.0,
    "dominant_colors": 1.0,
    "glcm": 20.0,
    "lbp": 0.1,
}


def generate(per_category: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    n_cat = len(CATEGORIES)
    centers = rng.random((n_cat, config.N_DIMS))

    rows, labels = [], []
    for c, name in enumerate(CATEGORIES):
        block = centers[c] + rng.normal(0.0, noise, size=(per_category, config.N_DIMS))
        rows.append(np.clip(block, 0.0, None))
        labels += [name] * per_category
    X = np.vstack(rows)

    for block_name, (start, end) in config.BLOCKS.items():
        X[:, start:end] *= BLOCK_SCALES[block_name]
    return X.astype(np.float32), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=config.SYNTHETIC_DIR)
    args = parser.parse_args()

    X, labels = generate(args.per_category, args.noise, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mapping = [
        {
            "index": i,
            "image_id": i + 1,
            "filename": f"synthetic_{i}.jpg",
            "category": labels[i],
            "path": f"dataset/{labels[i]}/synthetic_{i}.jpg",
        }
        for i in range(len(labels))
    ]

    np.save(args.out / "descriptors_raw.npy", X)
    with open(args.out / "image_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Données synthétiques écrites dans {args.out}")
    print(f"  matrice : {X.shape}, {len(set(labels))} catégories")


if __name__ == "__main__":
    main()
