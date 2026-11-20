"""Blueprint Flask-RESTful autonome : POST /search et POST /feedback (Tâche 6).

Ce module ne dépend ni de la base de données ni de l'authentification de A.
A l'enregistre dans son application :

    from backend.services.loader import load_engine
    from backend.routes.search_routes import create_search_blueprint

    engine, normalizer = load_engine()
    bp = create_search_blueprint(
        engine,
        normalizer=normalizer,                   # requis si vector_provider
        vector_provider=my_vector_provider,      # optionnel : images uploadées
        auth_decorators=[login_required],        # authentification de A
    )
    app.register_blueprint(bp, url_prefix="/api")

`vector_provider(image_id)` retourne le vecteur BRUT (non normalisé, même
format que la matrice de B) d'une image qui n'est pas dans la base indexée,
ou None si elle est inconnue. Il sert aux images uploadées par l'utilisateur.

Codes de réponse : 200 succès ; 400 requête invalide ; 404 image ou session
inconnue ; 409 nombre maximal d'itérations atteint.
"""
import functools

from flask import Blueprint, request
from flask_restful import Api, Resource

from backend import cbir_config as config
from backend.services.normalization import transform


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _api_errors(method):
    """Transforme ApiError en réponse JSON {"error": ...} avec le bon code."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ApiError as e:
            return {"error": e.message}, e.status
    return wrapper


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError(400, "Corps JSON (objet) attendu")
    return data


class SearchResource(Resource):
    def __init__(self, engine, normalizer, vector_provider):
        self.engine = engine
        self.normalizer = normalizer
        self.vector_provider = vector_provider

    @_api_errors
    def post(self):
        data = _json_object()

        query_id = data.get("query_image_id")
        if not _is_int(query_id):
            raise ApiError(400, "query_image_id (entier) obligatoire")

        scope = data.get("scope")
        if scope is not None and not (_is_int(scope) and 1 <= scope <= self.engine.max_scope):
            raise ApiError(400, f"scope doit être un entier entre 1 et {self.engine.max_scope}")

        method = data.get("method", config.DEFAULT_METHOD)
        if method not in config.METHODS:
            raise ApiError(400, f"method doit être dans {list(config.METHODS)}")

        exclude_query = data.get("exclude_query", False)
        if not isinstance(exclude_query, bool):
            raise ApiError(400, "exclude_query doit être un booléen")

        options = {"method": method, "exclude_query": exclude_query, "scope": scope}
        try:
            if self.engine.has_image(query_id):
                return self.engine.start(query_image_id=query_id, **options)
            if self.vector_provider is not None:
                raw_vector = self.vector_provider(query_id)
                if raw_vector is not None:
                    vector = transform(raw_vector, self.normalizer)
                    return self.engine.start(query_vector=vector, **options)
        except ValueError as e:
            raise ApiError(400, str(e)) from e
        raise ApiError(404, f"Image inconnue : {query_id}")


class FeedbackResource(Resource):
    def __init__(self, engine):
        self.engine = engine

    @_api_errors
    def post(self):
        data = _json_object()

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ApiError(400, "session_id (chaîne) obligatoire")

        relevant_ids = data.get("relevant_ids")
        if not isinstance(relevant_ids, list) or not all(_is_int(i) for i in relevant_ids):
            raise ApiError(400, "relevant_ids doit être une liste d'entiers (éventuellement vide)")
        if len(relevant_ids) > self.engine.max_scope:
            raise ApiError(400, f"relevant_ids : {self.engine.max_scope} éléments au maximum")

        try:
            session = self.engine.get_session(session_id)
        except KeyError:
            raise ApiError(404, "Session inconnue ou expirée") from None
        if session.iteration >= self.engine.max_iterations:
            raise ApiError(409, f"Nombre maximal d'itérations atteint ({self.engine.max_iterations})")

        try:
            return self.engine.feedback(session_id, relevant_ids)
        except KeyError:
            raise ApiError(404, "Session inconnue ou expirée") from None
        except ValueError as e:
            raise ApiError(400, str(e)) from e


def create_search_blueprint(engine, *, normalizer=None, vector_provider=None,
                            auth_decorators=None, name="cbir_search"):
    """Crée le blueprint des endpoints de recherche.

    auth_decorators : liste de décorateurs appliqués à chaque endpoint (par
    exemple `flask_login.login_required`). Sans eux, les endpoints sont publics.
    """
    if vector_provider is not None and normalizer is None:
        raise ValueError("normalizer est obligatoire lorsque vector_provider est fourni")

    blueprint = Blueprint(name, __name__)
    api = Api(blueprint, decorators=list(auth_decorators or []))
    api.add_resource(
        SearchResource, "/search",
        resource_class_kwargs={
            "engine": engine, "normalizer": normalizer, "vector_provider": vector_provider,
        },
    )
    api.add_resource(FeedbackResource, "/feedback", resource_class_kwargs={"engine": engine})
    return blueprint
