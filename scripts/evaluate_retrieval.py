"""Évalue la recherche simple et le retour de pertinence (Tâche 7).

Produit dans le dossier de sortie (par défaut `results/`) :
    metrics_by_iteration.csv   Precision@25 et mAP par méthode et itération (± IC95)
    paired_differences.csv     écart à la recherche simple, requête par requête
    per_category.csv           Precision@25 par catégorie
    pr_curve.csv               précision / rappel moyens à différents rangs
    precision_par_iteration.png, precision_rappel.png   figures du rapport
    tableaux_rapport.md        tableaux et protocole, prêts à coller dans le rapport
    evaluation_summary.json    paramètres et dimensions, pour la reproductibilité

Usage :
    python scripts/evaluate_retrieval.py --features features
    python scripts/evaluate_retrieval.py --features features --queries 200 --seed 42

Données de test (NE PAS citer ces chiffres dans le rapport) :
    python scripts/make_synthetic_data.py --per-category 100 --noise 0.6
    python scripts/evaluate_retrieval.py --features features/synthetic --out results_synthetic
"""
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import cbir_config as config  # noqa: E402
from backend.services import evaluation as ev  # noqa: E402
from backend.services.loader import load_normalized  # noqa: E402

METHOD_LABELS = {
    "simple": "Recherche simple",
    "mean_only": "Requête moyennée (sans poids)",
    "type1": "Type-1 (papier)",
    "type1_cumulative": "Type-1 cumulatif (extension)",
}
TABLE_LABELS = {
    "mean_only": "Requête moyennée",
    "type1": "Type-1 (papier)",
    "type1_cumulative": "Type-1 cumulatif",
}
# Palette validée (scripts/validate_palette.js, mode clair) : slots 1 à 3
SERIES_COLORS = {"type1": "#2a78d6", "type1_cumulative": "#eb6834", "mean_only": "#1baf7a"}
PLOT_ORDER = ("type1", "type1_cumulative", "mean_only")
SURFACE, INK, INK2, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e89"


# --------------------------------------------------------------------------
# Exports CSV
# --------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _zorder(method):
    """Type-1 (papier) est tracé au-dessus : il reste visible quand les courbes coïncident."""
    return 3 + 0.1 * (len(PLOT_ORDER) - PLOT_ORDER.index(method))


def _row_major(handles, ncol=2):
    """matplotlib remplit la légende par colonnes ; on réordonne pour lire par lignes."""
    return [h for start in range(ncol) for h in handles[start::ncol]]


def _spread(values, min_gap):
    """Écarte des étiquettes trop proches (même centre de gravité)."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        placed[b] = max(placed[b], placed[a] + min_gap)
    return placed - (placed.mean() - np.mean(values))


def plot_precision(result, summary, path):
    scope, n_iter = result["scope"], result["n_iter"]
    methods = [m for m in PLOT_ORDER if m in result["methods"]]
    iters = np.arange(1, n_iter + 1)

    stats = {m: {"mean": np.array([r["p_at_scope"] for r in summary
                                   if r["method"] == m]),
                 "ci": np.array([r["p_ci95"] for r in summary if r["method"] == m])}
             for m in methods}
    baseline = stats[methods[0]]["mean"][0]

    lo = min((s["mean"] - s["ci"]).min() for s in stats.values())
    hi = max((s["mean"] + s["ci"]).max() for s in stats.values())
    y_lo = max(0.0, math.floor((lo - 0.03) * 20) / 20)
    y_hi = min(1.0, math.ceil((hi + 0.03) * 20) / 20)

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200, facecolor=SURFACE)
    _style_axes(ax)
    base_line = ax.axhline(baseline, color=BASELINE, linewidth=1.2, zorder=1,
                           label=f"{METHOD_LABELS['simple']} (= itération 1)")

    handles = [base_line]
    for m in methods:
        mean, ci, color = stats[m]["mean"], stats[m]["ci"], SERIES_COLORS[m]
        ax.fill_between(iters, mean - ci, mean + ci, color=color, alpha=0.10, linewidth=0)
        (line,) = ax.plot(iters, mean, color=color, linewidth=1.8, solid_capstyle="round",
                          marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                          label=METHOD_LABELS[m], zorder=_zorder(m))
        handles.append(line)

    # Valeur finale : une seule étiquette quand plusieurs méthodes ont la même valeur
    final_values = sorted({round(float(stats[m]["mean"][-1]), 3) for m in methods})
    label_y = _spread(final_values, 0.045 * (y_hi - y_lo))
    for value, y in zip(final_values, label_y):
        ax.text(n_iter + 0.15, y, f"{value:.3f}", color=INK2, fontsize=9, va="center", ha="left")

    ax.set_xlim(0.8, n_iter + 0.9)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(iters)
    ax.set_xlabel("Itération (1 = recherche simple)", color=INK2, fontsize=10)
    ax.set_ylabel(f"Precision@{scope} moyenne", color=INK2, fontsize=10)
    ax.set_title(f"Precision@{scope} selon l'itération de retour de pertinence",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.03, f"{len(result['query_idx'])} requêtes, feedback oracle, "
            "bandes = intervalle de confiance à 95 %",
            transform=ax.transAxes, fontsize=9, color=INK2)

    fig.legend(handles=_row_major(handles), loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_pr(result, pr_rows, path, iterations):
    scope = result["scope"]
    methods = [m for m in PLOT_ORDER if m in result["methods"]]
    fig, axes = plt.subplots(1, len(iterations), figsize=(5.2 * len(iterations), 5.0),
                             dpi=200, facecolor=SURFACE, sharey=True, squeeze=False)

    def curve(method, iteration):
        rows = [r for r in pr_rows if r["method"] == method and r["iteration"] == iteration]
        return (np.array([r["recall"] for r in rows]), np.array([r["precision"] for r in rows]),
                np.array([r["rank"] for r in rows]))

    max_recall = max(curve(m, result["n_iter"])[0].max() for m in methods)
    handles = []
    for ax, iteration in zip(axes[0], iterations):
        _style_axes(ax)
        recall, precision, ranks = curve(methods[0], 1)
        (base,) = ax.plot(recall, precision, color=BASELINE, linewidth=1.8,
                          solid_capstyle="round", label=f"{METHOD_LABELS['simple']} (itération 1)")
        for m in methods:
            recall, precision, ranks = curve(m, iteration)
            (line,) = ax.plot(recall, precision, color=SERIES_COLORS[m], linewidth=1.8,
                              solid_capstyle="round", label=METHOD_LABELS[m],
                              zorder=_zorder(m))
            at_scope = np.flatnonzero(ranks == scope)
            if len(at_scope):                                   # repère : le rang = scope
                ax.plot(recall[at_scope], precision[at_scope], "o", color=SERIES_COLORS[m],
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        zorder=_zorder(m) + 0.05)
            if iteration == iterations[0]:
                handles.append(line)
        if iteration == iterations[0]:
            handles.insert(0, base)
        ax.set_title(f"Itération {iteration}", loc="left", fontsize=11, fontweight="bold",
                     color=INK)
        ax.set_xlabel("Rappel", color=INK2, fontsize=10)
        ax.set_xlim(0, min(1.0, max_recall * 1.05))
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("Précision", color=INK2, fontsize=10)
    fig.suptitle("Courbes précision-rappel (classement complet, requête exclue)",
                 x=0.01, y=0.975, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, f"{len(result['query_idx'])} requêtes, feedback oracle ; "
             f"le point repère le rang {scope}", fontsize=9, color=INK2)

    fig.legend(handles=_row_major(handles), loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.25, wspace=0.08)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Tableaux Markdown pour le rapport
# --------------------------------------------------------------------------
def _fmt(mean, ci):
    return f"{mean:.3f} ± {ci:.3f}"


def _iteration_table(summary, methods, n_iter, key, ci_key):
    lines = ["| Itération | " + " | ".join(TABLE_LABELS[m] for m in methods) + " |",
             "|---|" + "---|" * len(methods)]
    for k in range(1, n_iter + 1):
        cells = []
        for m in methods:
            row = next(r for r in summary if r["method"] == m and r["iteration"] == k)
            cells.append(_fmt(row[key], row[ci_key]))
        label = "1 (recherche simple)" if k == 1 else str(k)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown(result, summary, paired, categories, info):
    methods = list(result["methods"])
    n_iter, scope = result["n_iter"], result["scope"]
    shown = sorted({2, min(3, n_iter), n_iter})

    sizes = (f"{info['min_category']} images par catégorie"
             if info["min_category"] == info["max_category"]
             else f"{info['min_category']} à {info['max_category']} images par catégorie")
    out = [f"# Résultats de l'évaluation (généré le {info['date']})", "",
           "## Protocole", "",
           f"- Base : {info['n_images']} images, {info['n_categories']} catégories ({sizes}).",
           f"- Requêtes : {info['n_queries']} images tirées de façon stratifiée par catégorie "
           f"(graine {info['seed']}).",
           "- Vérité terrain : une image est pertinente si elle a la même catégorie que la requête.",
           "- La requête est exclue des résultats.",
           f"- Feedback oracle : parmi les {scope} images affichées, celles de la bonne catégorie "
           "sont déclarées pertinentes. Un vrai utilisateur est moins fiable.",
           f"- {n_iter} itérations ; l'itération 1 est la recherche simple (poids égaux à 1).",
           f"- Descripteurs normalisés (z-score par dimension), ε = {config.EPSILON}, poids Type-1 "
           f"écrêtés dans [{config.WEIGHT_MIN}, {config.WEIGHT_MAX}] puis ramenés à une moyenne de 1 "
           "(choix d'implémentation, hors papier).",
           "- Intervalles : IC à 95 % (loi normale) sur les requêtes.",
           f"- Plafond de Precision@{scope} imposé par la taille des catégories : "
           f"{info['precision_cap']:.3f}.", ""]

    out += [f"## Table 1 : Precision@{scope} moyenne ± IC95 par itération", "",
            _iteration_table(summary, methods, n_iter, "p_at_scope", "p_ci95"), "",
            "## Table 2 : mAP (classement complet) ± IC95 par itération", "",
            _iteration_table(summary, methods, n_iter, "map", "map_ci95"), ""]

    out += [f"## Table 3 : écart de Precision@{scope} par rapport à la recherche simple "
            "(comparaison appariée)", "",
            "| Méthode | Itération | Écart moyen ± IC95 | Requêtes améliorées | Requêtes dégradées |",
            "|---|---|---|---|---|"]
    for row in paired:
        if row["iteration"] in shown:
            out.append(f"| {TABLE_LABELS[row['method']]} | {row['iteration']} | "
                       f"{row['mean_diff']:+.3f} ± {row['ci95']:.3f} | "
                       f"{row['pct_better']:.1f} % | {row['pct_worse']:.1f} % |")
    out.append("")

    out += [f"## Table 4 : Precision@{scope} par catégorie (itération 1 puis itération {n_iter})", "",
            "| Catégorie | Requêtes | Recherche simple | "
            + " | ".join(TABLE_LABELS[m] for m in methods) + " |",
            "|---|---|---|" + "---|" * len(methods)]
    for c in sorted({r["category"] for r in categories}):
        rows = [r for r in categories if r["category"] == c]
        first = next(r for r in rows if r["method"] == methods[0] and r["iteration"] == 1)
        cells = [f"{next(r for r in rows if r['method'] == m and r['iteration'] == n_iter)['p_at_scope']:.3f}"
                 for m in methods]
        out.append(f"| {c} | {first['n_queries']} | {first['p_at_scope']:.3f} | "
                   + " | ".join(cells) + " |")
    out += ["", "Figures : `precision_par_iteration.png`, `precision_rappel.png`.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Exécution
# --------------------------------------------------------------------------
def run(features_dir, out_dir, n_queries=200, n_iter=config.MAX_ITERATIONS, scope=config.SCOPE,
        seed=42, methods=ev.EVAL_METHODS, expected_dims=config.N_DIMS, verbose=False):
    """Lance l'évaluation complète et écrit tous les fichiers. Retourne les chemins."""
    start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Z, mapping, _ = load_normalized(features_dir, expected_dims)
    labels = np.array([m["category"] for m in mapping])
    query_idx = ev.sample_queries(labels, n_queries, seed)

    categories, counts = np.unique(labels, return_counts=True)
    cap = ev.precision_cap(labels, query_idx, scope)
    if cap < 1.0:
        print(f"Attention : Precision@{scope} ne peut pas dépasser {cap:.3f} "
              f"(catégories de {counts.min()} à {counts.max()} images).", file=sys.stderr)

    def progress(done, total):
        if verbose and (done % 10 == 0 or done == total):
            print(f"\r  requêtes évaluées : {done}/{total}", end="", file=sys.stderr, flush=True)
            if done == total:
                print(file=sys.stderr)

    result = ev.evaluate(Z, labels, query_idx, methods, n_iter, scope, progress=progress)
    summary = ev.summarize(result)
    paired = ev.paired_differences(result)
    paired_map = ev.paired_differences(result, metric="ap")
    by_category = ev.per_category(result, labels)
    pr_rows = ev.pr_curve(result)

    simple = [dict(r, method="simple") for r in summary
              if r["method"] == result["methods"][0] and r["iteration"] == 1]
    paths = {
        "metrics": out_dir / "metrics_by_iteration.csv",
        "paired": out_dir / "paired_differences.csv",
        "per_category": out_dir / "per_category.csv",
        "pr_curve": out_dir / "pr_curve.csv",
        "fig_precision": out_dir / "precision_par_iteration.png",
        "fig_pr": out_dir / "precision_rappel.png",
        "markdown": out_dir / "tableaux_rapport.md",
        "summary": out_dir / "evaluation_summary.json",
    }
    write_csv(paths["metrics"], simple + summary)
    write_csv(paths["paired"], paired + paired_map)
    write_csv(paths["per_category"], by_category)
    write_csv(paths["pr_curve"], pr_rows)

    plot_precision(result, summary, paths["fig_precision"])
    plot_pr(result, pr_rows, paths["fig_pr"], sorted({min(3, n_iter), n_iter}))

    info = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "features_dir": str(features_dir),
        "n_images": int(len(labels)),
        "n_dims": int(Z.shape[1]),
        "n_categories": int(len(categories)),
        "min_category": int(counts.min()),
        "max_category": int(counts.max()),
        "n_queries": int(len(query_idx)),
        "seed": seed,
        "scope": scope,
        "n_iterations": n_iter,
        "methods": list(result["methods"]),
        "precision_cap": cap,
        "feedback": "oracle (catégories du jeu de données)",
        "query_excluded_from_results": True,
        "epsilon": config.EPSILON,
        "weight_clip": [config.WEIGHT_MIN, config.WEIGHT_MAX],
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    paths["markdown"].write_text(build_markdown(result, summary, paired, by_category, info),
                                 encoding="utf-8")
    paths["summary"].write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=config.FEATURES_DIR)
    parser.add_argument("--out", type=Path, default=config.ROOT_DIR / "results")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=config.MAX_ITERATIONS)
    parser.add_argument("--scope", type=int, default=config.SCOPE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dims", type=int, default=config.N_DIMS,
                        help="dimension attendue des vecteurs (0 = ne pas vérifier)")
    args = parser.parse_args()

    paths = run(args.features, args.out, args.queries, args.iterations, args.scope, args.seed,
                expected_dims=args.dims or None, verbose=True)
    print(f"Résultats écrits dans {args.out} :")
    for p in paths.values():
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
