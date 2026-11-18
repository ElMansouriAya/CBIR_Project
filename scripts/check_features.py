"""Contrôle rapide des descripteurs livrés par B (avant toute évaluation).

Vérifie le contrat de données (CONTRAT.md) puis signale les anomalies fréquentes :
histogrammes qui ne somment pas à 1, valeurs hors bornes, dimensions constantes,
images en double, catégories déséquilibrées.

Usage :
    python scripts/check_features.py                      # dossier features/
    python scripts/check_features.py --features features/synthetic

Code de sortie : 0 si le contrat est respecté (les avertissements n'échouent pas),
1 sinon.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import cbir_config as config  # noqa: E402
from backend.services.loader import load_raw, validate_contract  # noqa: E402

TOL = 1e-3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.FEATURES_DIR)
    parser.add_argument("--dims", type=int, default=config.N_DIMS,
                        help="dimension attendue (0 = ne pas vérifier)")
    args = parser.parse_args()

    try:
        X, mapping = load_raw(args.features)
        validate_contract(X, mapping, args.dims or None)
    except (FileNotFoundError, ValueError) as e:
        print(f"CONTRAT NON RESPECTÉ : {e}")
        return 1

    labels = np.array([m["category"] for m in mapping])
    ids = [m["image_id"] for m in mapping]
    categories, counts = np.unique(labels, return_counts=True)
    warnings = []

    print(f"Contrat respecté : {X.shape[0]} images x {X.shape[1]} dimensions "
          f"({X.dtype}), {len(categories)} catégories")
    print(f"Images par catégorie : min {counts.min()}, max {counts.max()}")
    if counts.min() != counts.max():
        warnings.append("catégories de tailles différentes (GHIM-10K : 500 images chacune)")
    if len(set(ids)) != len(ids):
        warnings.append("image_id dupliqués")

    # Convention confirmée avec A : image_id = index + 1, images triées par catégorie
    if any(m["image_id"] != m["index"] + 1 for m in mapping):
        warnings.append("image_id différent de index + 1 (convention confirmée avec A)")
    block_categories = [labels[0]] + [labels[i] for i in range(1, len(labels))
                                      if labels[i] != labels[i - 1]]
    if len(block_categories) != len(categories):
        warnings.append("les images d'une même catégorie ne sont pas contiguës "
                        "(attendu : triées par catégorie puis par nom de fichier)")
    elif block_categories not in (sorted(block_categories),
                                  sorted(block_categories, key=str.casefold)):
        warnings.append("les catégories ne sont pas dans l'ordre alphabétique")
    paths = [str(m.get("path", "")) for m in mapping]
    if any("\\" in p or p.startswith("/") or ":" in p for p in paths):
        warnings.append("path : chemins avec des \\ ou non relatifs "
                        "(attendu : chemins relatifs à image_storage/, avec des /)")

    print("\nBloc                 colonnes   min        max        écart-type moyen  dims constantes")
    for name, (start, end) in config.BLOCKS.items():
        if end > X.shape[1]:
            continue
        block = X[:, start:end]
        n_const = int((block.std(axis=0) < config.STD_MIN).sum())
        print(f"{name:<20} {start:>3}-{end - 1:<3}   {block.min():<10.4g} {block.max():<10.4g} "
              f"{block.std(axis=0).mean():<17.4g} {n_const}/{end - start}")
        if n_const == end - start:
            warnings.append(f"{name} : toutes les dimensions sont constantes (extraction à vérifier)")

    # Contrôles issus du contrat (sections 2 et 3 de CONTRAT.md)
    if X.shape[1] == config.N_DIMS:
        for name in ("hsv_hist", "lbp"):
            s, e = config.BLOCKS[name]
            ok = np.abs(X[:, s:e].sum(axis=1) - 1.0) < TOL
            if not ok.all():
                warnings.append(f"{name} : la somme vaut 1 pour {ok.mean():.1%} des images seulement "
                                "(le contrat demande une somme de 1)")
        s, e = config.BLOCKS["dominant_colors"]
        if X[:, s:e].min() < -TOL or X[:, s:e].max() > 1 + TOL:
            warnings.append("dominant_colors : valeurs hors de [0, 1]")

    n_unique = len(np.unique(X, axis=0))
    if n_unique < len(X):
        warnings.append(f"{len(X) - n_unique} vecteurs identiques à un autre (images en double ?)")
    n_const_total = int((X.std(axis=0) < config.STD_MIN).sum())
    if n_const_total:
        print(f"\nDimensions constantes au total : {n_const_total} (leur poids sera neutre)")

    print()
    if warnings:
        print(f"{len(warnings)} avertissement(s) :")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Aucun avertissement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
