"""Recherche par similarité (Tâche 3).

Recherche simple : distance euclidienne entre le vecteur requête et tous les
vecteurs de la base, puis classement croissant.

    d(Q, X) = sqrt( sum_i (Q_i - X_i)^2 )

Si des poids sont fournis (retour de pertinence Type-1), on utilise la
distance euclidienne pondérée du papier (équation 3) :

    d_w(Q, X) = sqrt( sum_i w_i (Q_i - X_i)^2 / sum_i w_i )

Les vecteurs doivent déjà être normalisés (voir normalization.py).
"""
import numpy as np

from backend import cbir_config as config


def euclidean_distances(query, X):
    """Distances euclidiennes entre `query` (d,) et chaque ligne de `X` (N, d)."""
    query = np.asarray(query, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    _check_shapes(query, X)
    diff = X - query
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def weighted_distances(query, X, weights):
    """Distances euclidiennes pondérées (papier, éq. 3) entre `query` et `X`."""
    query = np.asarray(query, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _check_shapes(query, X)
    if weights.shape != query.shape:
        raise ValueError(f"weights doit avoir la forme {query.shape}, reçu {weights.shape}")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights doit être positif avec une somme strictement positive")
    diff = X - query
    return np.sqrt((diff ** 2) @ weights / weights.sum())


def search(query, X, scope=config.SCOPE, exclude_index=None, weights=None):
    """Retourne les `scope` images les plus proches de `query`.

    Paramètres
    ----------
    query : (d,) vecteur requête déjà normalisé
    X : (N, d) matrice de la base déjà normalisée
    scope : nombre de résultats demandés
    exclude_index : ligne de `X` à écarter des résultats. À utiliser en
        évaluation, quand la requête est une image de la base : sinon elle
        revient en rang 1 avec une distance nulle et gonfle la précision.
    weights : (d,) poids optionnels. Sans poids, distance euclidienne simple.

    Retourne
    --------
    (indices, distances) : deux tableaux triés par distance croissante.
    Les égalités sont départagées par l'ordre des lignes (tri stable).
    """
    if scope < 1:
        raise ValueError("scope doit être >= 1")
    X = np.asarray(X)
    if exclude_index is not None and not 0 <= exclude_index < X.shape[0]:
        raise ValueError(f"exclude_index hors limites : {exclude_index}")

    if weights is None:
        dist = euclidean_distances(query, X)
    else:
        dist = weighted_distances(query, X, weights)

    if exclude_index is not None:
        dist = dist.copy()
        dist[exclude_index] = np.inf

    order = np.argsort(dist, kind="stable")
    n_available = X.shape[0] - (0 if exclude_index is None else 1)
    order = order[: min(scope, n_available)]
    return order, dist[order]


def _check_shapes(query, X):
    if X.ndim != 2:
        raise ValueError(f"X doit être une matrice (N, d), reçu {X.shape}")
    if query.ndim != 1 or query.shape[0] != X.shape[1]:
        raise ValueError(
            f"query doit avoir la forme ({X.shape[1]},), reçu {query.shape}"
        )
    if not (np.isfinite(query).all() and np.isfinite(X).all()):
        raise ValueError("query ou X contient des NaN ou des inf")
