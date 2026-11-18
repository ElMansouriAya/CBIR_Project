"""Évaluation expérimentale du retour de pertinence (Tâche 7).

Protocole
---------
- Requêtes : images de la base, tirées de façon stratifiée (autant par catégorie).
- Vérité terrain : une image est pertinente si elle a la MÊME catégorie que la
  requête. La requête est EXCLUE du classement (sinon elle arrive première à
  distance 0 et gonfle la précision).
- Feedback simulé (« oracle ») : parmi les `scope` images affichées, celles de
  la bonne catégorie sont déclarées pertinentes. C'est un retour idéal : un vrai
  utilisateur se trompe et oublie des images.
- Itération 1 = recherche simple (poids égaux à 1, requête d'origine). Elle est
  donc identique pour toutes les méthodes.
- Mesures : Precision@scope (sur les `scope` premières images), AP/mAP et
  précision/rappel à différents rangs, tous calculés sur le classement COMPLET de
  la base (requête exclue) à chaque itération.

Méthodes comparées (à partir de l'itération 2) :
    mean_only         requête = moyenne des pertinentes, poids restent à 1
    type1             Type-1 fidèle au papier (itération courante uniquement)
    type1_cumulative  Type-1 sur toutes les images vues/pertinentes de la session

`mean_only` isole l'effet des poids : si `type1` ne fait pas mieux, le gain ne
vient que du déplacement de la requête.

La boucle reproduit exactement la logique de `session.SearchEngine` (vérifié
par tests/test_evaluation.py).
"""
import numpy as np

from backend import cbir_config as config
from backend.services.relevance_feedback import feedback_step, initial_weights, update_query
from backend.services.search import search

EVAL_METHODS = ("mean_only", "type1", "type1_cumulative")
DEFAULT_RANK_POINTS = (1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500)
Z95 = 1.96


# --------------------------------------------------------------------------
# Requêtes et métriques élémentaires
# --------------------------------------------------------------------------
def sample_queries(labels, n_queries, seed=42):
    """Tire `n_queries` indices de façon stratifiée (autant par catégorie)."""
    labels = np.asarray(labels)
    n_queries = min(int(n_queries), len(labels))
    rng = np.random.default_rng(seed)
    categories = np.unique(labels)
    per_category = n_queries // len(categories)

    chosen = []
    for c in categories:
        idx = np.flatnonzero(labels == c)
        chosen.extend(rng.choice(idx, size=min(per_category, len(idx)), replace=False))

    missing = n_queries - len(chosen)
    if missing > 0:
        pool = np.setdiff1d(np.arange(len(labels)), chosen)
        chosen.extend(rng.choice(pool, size=min(missing, len(pool)), replace=False))
    return np.sort(np.asarray(chosen, dtype=int))


def average_precision(rel):
    """AP d'un classement : `rel` est un tableau booléen dans l'ordre du classement."""
    rel = np.asarray(rel, dtype=bool)
    n_relevant = rel.sum()
    if n_relevant == 0:
        return 0.0
    precision_at_k = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return float(precision_at_k[rel].sum() / n_relevant)


def precision_cap(labels, query_idx, scope):
    """Precision@scope maximale possible (une catégorie trop petite la plafonne)."""
    labels = np.asarray(labels)
    sizes = {c: n for c, n in zip(*np.unique(labels, return_counts=True))}
    caps = [min(1.0, (sizes[labels[i]] - 1) / scope) for i in query_idx]
    return float(np.mean(caps))


def _rank(Z, labels, i, query, weights):
    """Classement complet de la base (requête `i` exclue) et pertinence de chaque rang."""
    order, _ = search(query, Z, scope=len(Z) - 1, exclude_index=i, weights=weights)
    return order, labels[order] == labels[i]


def _measures(rel, scope, rank_points):
    hits = np.cumsum(rel)[np.asarray(rank_points) - 1]
    return float(rel[:scope].mean()), average_precision(rel), hits


# --------------------------------------------------------------------------
# Boucle d'évaluation
# --------------------------------------------------------------------------
def evaluate_query(Z, labels, i, methods, n_iter, scope, rank_points, trace=None):
    """Évalue une requête `i` pour chaque méthode, sur `n_iter` itérations.

    `trace` : liste optionnelle qui reçoit `(méthode, itération, images affichées)`
    (itération numérotée à partir de 1) ; sert aux tests d'équivalence.
    """
    d = Z.shape[1]
    q0, w0 = Z[i].copy(), initial_weights(d)
    order0, rel0 = _rank(Z, labels, i, q0, w0)
    base = _measures(rel0, scope, rank_points)

    out = {}
    for method in methods:
        p = np.empty(n_iter)
        ap = np.empty(n_iter)
        hits = np.empty((n_iter, len(rank_points)))
        p[0], ap[0], hits[0] = base
        if trace is not None:
            trace.append((method, 1, order0[:scope].copy()))

        query, weights = q0.copy(), w0.copy()
        order, rel = order0, rel0
        seen = order[:scope]
        relevant_all = np.array([], dtype=int)

        for k in range(1, n_iter):
            displayed = order[:scope]
            relevant = displayed[rel[:scope]]                     # feedback oracle

            if method == "mean_only":
                if len(relevant) > 0:
                    query = update_query(Z[relevant])
            elif method == "type1":
                query, weights, _ = feedback_step(Z, displayed, relevant, query, weights)
            elif method == "type1_cumulative":
                relevant_all = np.union1d(relevant_all, relevant).astype(int)
                query, weights, _ = feedback_step(Z, seen, relevant_all, query, weights)
            else:
                raise ValueError(f"Méthode inconnue : {method!r}")

            order, rel = _rank(Z, labels, i, query, weights)
            seen = np.union1d(seen, order[:scope]).astype(int)
            p[k], ap[k], hits[k] = _measures(rel, scope, rank_points)
            if trace is not None:
                trace.append((method, k + 1, order[:scope].copy()))

        out[method] = (p, ap, hits)
    return out


def evaluate(Z, labels, query_idx, methods=EVAL_METHODS, n_iter=config.MAX_ITERATIONS,
             scope=config.SCOPE, rank_points=DEFAULT_RANK_POINTS, progress=None):
    """Évalue toutes les requêtes. `Z` doit être normalisée (normalization.py).

    Retourne un dictionnaire de tableaux, indexés [requête, itération(, rang)] :
        p_at_scope[m], ap[m], hits[m], n_relevant, rank_points, ...
    """
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels)
    query_idx = np.asarray(query_idx, dtype=int)
    if len(labels) != len(Z):
        raise ValueError("labels et Z doivent avoir la même longueur")
    if n_iter < 2:
        raise ValueError("n_iter doit être >= 2 (itération 1 = recherche simple)")
    if scope < 1 or scope >= len(Z):
        raise ValueError("scope doit être compris entre 1 et N-1")
    if len(query_idx) == 0:
        raise ValueError("Aucune requête à évaluer")

    sizes = {c: n for c, n in zip(*np.unique(labels, return_counts=True))}
    n_relevant = np.array([sizes[labels[i]] - 1 for i in query_idx])
    if (n_relevant < 1).any():
        raise ValueError("Une requête appartient à une catégorie d'une seule image")

    rank_points = np.array(sorted({r for r in rank_points if r <= len(Z) - 1}), dtype=int)
    n_q = len(query_idx)
    result = {
        "query_idx": query_idx,
        "n_iter": n_iter,
        "scope": scope,
        "methods": tuple(methods),
        "rank_points": rank_points,
        "n_relevant": n_relevant,
        "p_at_scope": {m: np.empty((n_q, n_iter)) for m in methods},
        "ap": {m: np.empty((n_q, n_iter)) for m in methods},
        "hits": {m: np.empty((n_q, n_iter, len(rank_points))) for m in methods},
    }

    for n, i in enumerate(query_idx):
        per_method = evaluate_query(Z, labels, i, methods, n_iter, scope, rank_points)
        for m, (p, ap, hits) in per_method.items():
            result["p_at_scope"][m][n] = p
            result["ap"][m][n] = ap
            result["hits"][m][n] = hits
        if progress is not None:
            progress(n + 1, n_q)
    return result


# --------------------------------------------------------------------------
# Statistiques de synthèse
# --------------------------------------------------------------------------
def mean_ci(values, axis=0):
    """Moyenne et demi-largeur de l'intervalle de confiance à 95 % (loi normale)."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    if n < 2:
        return mean, np.zeros_like(mean)
    return mean, Z95 * values.std(axis=axis, ddof=1) / np.sqrt(n)


def summarize(result):
    """Une ligne par (méthode, itération) : Precision@scope et mAP, moyenne ± IC95."""
    rows = []
    for m in result["methods"]:
        p_mean, p_ci = mean_ci(result["p_at_scope"][m])
        ap_mean, ap_ci = mean_ci(result["ap"][m])
        for k in range(result["n_iter"]):
            rows.append({"method": m, "iteration": k + 1,
                         "p_at_scope": p_mean[k], "p_ci95": p_ci[k],
                         "map": ap_mean[k], "map_ci95": ap_ci[k]})
    return rows


def paired_differences(result, metric="p_at_scope"):
    """Écart par rapport à la recherche simple (itération 1), requête par requête.

    Comparaison appariée : les mêmes requêtes servent aux deux mesures, donc
    l'IC porte sur la différence et non sur chaque moyenne séparément.
    """
    rows = []
    for m in result["methods"]:
        values = result[metric][m]
        for k in range(1, result["n_iter"]):
            diff = values[:, k] - values[:, 0]
            mean, ci = mean_ci(diff)
            rows.append({"method": m, "iteration": k + 1, "metric": metric,
                         "mean_diff": float(mean), "ci95": float(ci),
                         "pct_better": float((diff > 0).mean() * 100),
                         "pct_worse": float((diff < 0).mean() * 100)})
    return rows


def per_category(result, labels):
    """Precision@scope moyenne par catégorie, méthode et itération."""
    labels = np.asarray(labels)
    q_labels = labels[result["query_idx"]]
    rows = []
    for c in np.unique(q_labels):
        mask = q_labels == c
        for m in result["methods"]:
            means = result["p_at_scope"][m][mask].mean(axis=0)
            for k in range(result["n_iter"]):
                rows.append({"category": c, "method": m, "iteration": k + 1,
                             "p_at_scope": float(means[k]), "n_queries": int(mask.sum())})
    return rows


def pr_curve(result):
    """Précision et rappel moyens à chaque rang de `rank_points` (par méthode/itération)."""
    ranks = result["rank_points"]
    rows = []
    for m in result["methods"]:
        hits = result["hits"][m]                                       # (q, iter, rang)
        precision = (hits / ranks).mean(axis=0)
        recall = (hits / result["n_relevant"][:, None, None]).mean(axis=0)
        for k in range(result["n_iter"]):
            for j, r in enumerate(ranks):
                rows.append({"method": m, "iteration": k + 1, "rank": int(r),
                             "precision": float(precision[k, j]), "recall": float(recall[k, j])})
    return rows
