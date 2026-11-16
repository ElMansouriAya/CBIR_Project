"""Retour de pertinence : Re-weighting Type-1 (Tâche 4).

Référence : Renuka Devi SM, « Reweighing Strategies for Relevance Feedback in
CBIR », IJARCCE 5(11), 2016, section III-C (éq. 5) et tableau 1.

Formule du papier, pour chaque dimension i à l'itération k :

    w_i^(k+1) = (eps + sigma_Nr,i^k) / (eps + sigma_rel,i^k)

    sigma_Nr,i  : écart-type de la dimension i sur les images RETOURNÉES
                  (les N_r images du scope, pertinentes ou non)
    sigma_rel,i : écart-type de la dimension i sur les images PERTINENTES
    eps         : 0.0001

Une dimension peu dispersée chez les images pertinentes reçoit un poids élevé.

Requête de l'itération suivante (tableau 1) : moyenne des vecteurs pertinents.

Choix d'implémentation (NON attribués au papier) :
    - les poids bruts sont écrêtés dans [WEIGHT_MIN, WEIGHT_MAX] puis divisés
      par leur moyenne (moyenne finale = 1) ;
    - écarts-types calculés avec ddof=0 (population), comme dans normalization.py ;
    - avec 0 ou 1 image pertinente, les poids précédents sont conservés (un
      écart-type sur un seul échantillon vaut 0 et ferait exploser les poids) ;
    - sigma_Nr et sigma_rel sont calculés sur l'itération courante uniquement
      (le papier parle des images pertinentes « in k-th iteration »).

Les vecteurs doivent déjà être normalisés (voir normalization.py).
"""
import numpy as np

from backend import cbir_config as config


def initial_weights(n_dims=config.N_DIMS):
    """Poids de l'itération 1 : tous égaux à 1 (recherche simple)."""
    return np.ones(n_dims, dtype=np.float64)


def raw_weights(X_retrieved, X_relevant, epsilon=config.EPSILON):
    """Poids bruts du papier : (eps + sigma_Nr) / (eps + sigma_rel), par dimension."""
    X_retrieved = np.asarray(X_retrieved, dtype=np.float64)
    X_relevant = np.asarray(X_relevant, dtype=np.float64)
    if X_retrieved.ndim != 2 or X_relevant.ndim != 2:
        raise ValueError("X_retrieved et X_relevant doivent être des matrices (n, d)")
    if X_retrieved.shape[1] != X_relevant.shape[1]:
        raise ValueError("X_retrieved et X_relevant doivent avoir la même dimension d")
    if X_relevant.shape[0] < 2:
        raise ValueError("Au moins 2 images pertinentes sont nécessaires pour un écart-type")

    sigma_nr = X_retrieved.std(axis=0)
    sigma_rel = X_relevant.std(axis=0)
    return (epsilon + sigma_nr) / (epsilon + sigma_rel)


def stabilize_weights(w, w_min=config.WEIGHT_MIN, w_max=config.WEIGHT_MAX):
    """Écrête les poids dans [w_min, w_max], puis les ramène à une moyenne de 1."""
    w = np.clip(np.asarray(w, dtype=np.float64), w_min, w_max)
    return w / w.mean()


def compute_weights(X_retrieved, X_relevant, epsilon=config.EPSILON):
    """Poids Type-1 prêts à l'emploi : poids bruts puis garde-fous."""
    return stabilize_weights(raw_weights(X_retrieved, X_relevant, epsilon))


def update_query(X_relevant):
    """Nouvelle requête : moyenne des vecteurs des images pertinentes."""
    X_relevant = np.asarray(X_relevant, dtype=np.float64)
    if X_relevant.ndim != 2 or X_relevant.shape[0] < 1:
        raise ValueError("X_relevant doit contenir au moins une image (n, d)")
    return X_relevant.mean(axis=0)


def feedback_step(X, retrieved_idx, relevant_idx, query, weights):
    """Une itération de retour de pertinence.

    Paramètres
    ----------
    X : (N, d) matrice de la base, normalisée
    retrieved_idx : indices (lignes de X) des images affichées à l'itération k
    relevant_idx : indices jugés pertinents par l'utilisateur (sous-ensemble)
    query, weights : requête et poids courants

    Retourne
    --------
    (new_query, new_weights, info) où info = {"mode": ..., "n_relevant": ...}
    avec mode dans {"no_feedback", "query_only", "full"}.
    """
    X = np.asarray(X, dtype=np.float64)
    retrieved_idx = np.asarray(retrieved_idx, dtype=int)
    relevant_idx = np.unique(np.asarray(relevant_idx, dtype=int))  # sans doublons

    if not np.isin(relevant_idx, retrieved_idx).all():
        raise ValueError("relevant_idx doit être un sous-ensemble de retrieved_idx")

    query = np.asarray(query, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_rel = len(relevant_idx)

    if n_rel == 0:
        return query.copy(), weights.copy(), {"mode": "no_feedback", "n_relevant": 0}

    X_relevant = X[relevant_idx]
    new_query = update_query(X_relevant)

    if n_rel == 1:
        return new_query, weights.copy(), {"mode": "query_only", "n_relevant": 1}

    new_weights = compute_weights(X[retrieved_idx], X_relevant)
    return new_query, new_weights, {"mode": "full", "n_relevant": n_rel}
