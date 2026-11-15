import os
import logging
from flask import Flask
from backend.config import Config
from backend.extensions import db, login_manager
from backend.models import User

log = logging.getLogger(__name__)


def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    app.config.from_object(Config)

    for folder in (app.config["UPLOAD_FOLDER"], app.config["TRANSFORMED_FOLDER"]):
        os.makedirs(folder, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    register_search_api(app)

    @app.route("/")
    def index():
        return "VisualVault fonctionne"

    with app.app_context():
        db.create_all()
    return app


def register_search_api(app):
    """Monte l'API de recherche de C (contrat, section 7).
    L'app démarre même si le code de C ou les features de B sont absents."""
    from flask_login import login_required
    try:
        from backend.services.loader import load_engine
        from backend.routes.search_routes import create_search_blueprint

        engine, normalizer = load_engine(app.config["FEATURES_FOLDER"])
        bp = create_search_blueprint(
            engine, normalizer=normalizer, auth_decorators=[login_required]
        )
        app.register_blueprint(bp, url_prefix="/api")
    except (ImportError, FileNotFoundError, ValueError) as e:
        log.warning("API de recherche non montée : %s", e)


if __name__ == "__main__":
    create_app().run(debug=True)