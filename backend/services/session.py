"""Boucle de recherche itérative avec retour de pertinence (Tâche 5).

`SearchEngine` garde en mémoire l'état de chaque session de recherche :
requête courante, poids courants, numéro d'itération, images affichées et
images jugées pertinentes. Aucune dépendance à Flask : les routes (Tâche 6)
appelleront simplement `start` et `feedback`.

Deux méthodes de feedback (même formule Type-1, ensembles d'images différents) :

    "per_iteration" (fidèle au papier)
        sigma_Nr  : images affichées à l'itération courante
        sigma_rel : images jugées pertinentes à l'itération courante

    "cumulative" (extension, à présenter comme telle dans le rapport)
        sigma_Nr  : toutes les images affichées depuis le début de la session
        sigma_rel : toutes les images jugées pertinentes depuis le début

La méthode cumulative évite que les poids retombent à 1 quand toutes les
images affichées sont pertinentes (sigma_Nr == sigma_rel).

`X` doit être la matrice de la base DÉJÀ normalisée (normalization.py), et
`query_vector` doit avoir été transformé avec le même normaliseur.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from backend import cbir_config as config
from backend.services.relevance_feedback import feedback_step, initial_weights
from backend.services.search import search


@dataclass
class SearchSession:
    session_id: str
    method: str
    query: np.ndarray
    weights: np.ndarray
    exclude_index: int | None
    iteration: int = 1
    scope: int = config.SCOPE
    last_retrieved: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    last_distances: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    seen: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    relevant_all: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


class SearchEngine:
    def __init__(
        self,
        X,
        image_ids,
        categories,
        scope=config.SCOPE,
        max_iterations=config.MAX_ITERATIONS,
        max_sessions=config.MAX_SESSIONS,
        blocks=None,
        max_scope=config.MAX_SCOPE,
    ):
        self.X = np.asarray(X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError("X doit être une matrice (N, d)")
        n = self.X.shape[0]
        if len(image_ids) != n or len(categories) != n:
            raise ValueError("X, image_ids et categories doivent avoir la même longueur")

        self.image_ids = [int(i) for i in image_ids]
        self.categories = list(categories)
        self._id_to_index = {image_id: k for k, image_id in enumerate(self.image_ids)}
        if len(self._id_to_index) != n:
            raise ValueError("image_ids doit contenir des identifiants uniques")

        self.scope = scope
        self.max_iterations = max_iterations
        self.max_sessions = max_sessions
        self.max_scope = max_scope
        if blocks is None and self.X.shape[1] == config.N_DIMS:
            blocks = config.BLOCKS
        self.blocks = blocks

        self._sessions: "OrderedDict[str, SearchSession]" = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ API
    def start(self, query_image_id=None, query_vector=None,
              method=config.DEFAULT_METHOD, exclude_query=False, scope=None):
        """Lance une recherche initiale (itération 1, tous les poids égaux à 1).

        Donner soit `query_image_id` (image de la base), soit `query_vector`
        (image externe, déjà normalisée). `exclude_query=True` retire l'image
        requête des résultats : à utiliser en évaluation.
        """
        if method not in config.METHODS:
            raise ValueError(f"method doit être dans {config.METHODS}, reçu {method!r}")
        if (query_image_id is None) == (query_vector is None):
            raise ValueError("Donner exactement un de query_image_id ou query_vector")

        scope = self.scope if scope is None else scope
        if not isinstance(scope, (int, np.integer)) or not 1 <= scope <= self.max_scope:
            raise ValueError(f"scope doit être un entier entre 1 et {self.max_scope}")

        exclude_index = None
        if query_image_id is not None:
            idx = self._index_of(query_image_id)
            query = self.X[idx].copy()
            if exclude_query:
                exclude_index = idx
        else:
            query = np.asarray(query_vector, dtype=np.float64).copy()
            if query.shape != (self.X.shape[1],):
                raise ValueError(f"query_vector doit avoir la forme ({self.X.shape[1]},)")

        session = SearchSession(
            session_id=uuid.uuid4().hex,
            method=method,
            query=query,
            weights=initial_weights(self.X.shape[1]),
            exclude_index=exclude_index,
            scope=int(scope),
        )
        self._run_search(session)
        with self._lock:
            self._sessions[session.session_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)      # évince la plus ancienne
        return self._response(session)

    def feedback(self, session_id, relevant_ids):
        """Applique le retour de pertinence puis relance la recherche.

        `relevant_ids` : identifiants d'images parmi celles AFFICHÉES à
        l'itération courante. Liste vide autorisée (rien ne change).
        """
        with self._lock:
            session = self._get(session_id)
            if session.iteration >= self.max_iterations:
                raise ValueError(f"Nombre maximal d'itérations atteint ({self.max_iterations})")

            relevant_idx = np.unique([self._index_of(i) for i in relevant_ids]).astype(int)
            if not np.isin(relevant_idx, session.last_retrieved).all():
                raise ValueError("Seules les images affichées à cette itération peuvent être jugées")

            if session.method == "cumulative":
                session.relevant_all = np.union1d(session.relevant_all, relevant_idx).astype(int)
                retrieved, relevant = session.seen, session.relevant_all
            else:
                retrieved, relevant = session.last_retrieved, relevant_idx

            session.query, session.weights, info = feedback_step(
                self.X, retrieved, relevant, session.query, session.weights
            )
            session.iteration += 1
            self._run_search(session)
            return self._response(session, feedback=info)

    def has_image(self, image_id):
        """True si `image_id` fait partie de la base indexée."""
        try:
            return int(image_id) in self._id_to_index
        except (TypeError, ValueError):
            return False

    def get_session(self, session_id):
        with self._lock:
            return self._get(session_id)

    # ------------------------------------------------------------ internes
    def _run_search(self, session):
        idx, dist = search(
            session.query, self.X, scope=session.scope,
            exclude_index=session.exclude_index, weights=session.weights,
        )
        session.last_retrieved = idx
        session.last_distances = dist
        session.seen = np.union1d(session.seen, idx).astype(int)

    def _response(self, session, feedback=None):
        results = [
            {
                "image_id": self.image_ids[k],
                "distance": float(d),
                "category": self.categories[k],
            }
            for k, d in zip(session.last_retrieved, session.last_distances)
        ]
        response = {
            "session_id": session.session_id,
            "iteration": session.iteration,
            "method": session.method,
            "results": results,
            "block_weights": self._block_weights(session.weights),
        }
        if feedback is not None:
            response["feedback"] = feedback
        return response

    def _block_weights(self, weights):
        """Poids moyen par famille de descripteurs (pour l'affichage)."""
        if self.blocks is None:
            return None
        return {name: float(weights[s:e].mean()) for name, (s, e) in self.blocks.items()}

    def _index_of(self, image_id):
        try:
            return self._id_to_index[int(image_id)]
        except KeyError:
            raise ValueError(f"image_id inconnu : {image_id}") from None

    def _get(self, session_id):
        try:
            session = self._sessions[session_id]
        except KeyError:
            raise KeyError(f"Session inconnue ou expirée : {session_id}") from None
        self._sessions.move_to_end(session_id)           # marque comme récemment utilisée
        return session
