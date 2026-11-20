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

    # Crée les dossiers de stockage s'ils n'existent pas
    for folder in (app.config["UPLOAD_FOLDER"], app.config["TRANSFORMED_FOLDER"]):
        os.makedirs(folder, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Versionnement automatique des fichiers statiques (anti-cache)
    @app.context_processor
    def asset_version():
        def asset_v(filename):
            path = os.path.join(app.static_folder, filename)
            try:
                return int(os.path.getmtime(path))
            except OSError:
                return 0
        return {"asset_v": asset_v}

    # Blueprints de A (auth, pages, galerie, catégories, transformations)
    from backend.routes.auth import auth_bp
    from backend.routes.main import main_bp
    from backend.routes.gallery import gallery_bp
    from backend.routes.categories import categories_bp
    from backend.routes.transform import transform_bp
    from backend.routes.dataset import dataset_bp
    from backend.routes.similarity import similarity_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(gallery_bp)
    app.register_blueprint(categories_bp)
    app.register_blueprint(transform_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(similarity_bp)

    from backend.services.descriptors import load_descriptor_catalog
    app.extensions["ghim_descriptors"] = load_descriptor_catalog(app.config["FEATURES_FOLDER"])

    # API de recherche de C (monte si le code de C et les features de B existent)
    register_search_api(app)

    with app.app_context():
        db.create_all()
    return app


def register_search_api(app):
    """Monte l'API de recherche de C (contrat, section 7).
    L'app démarre même si le code de C ou les features de B sont absents."""
    from flask import request
    from flask_login import current_user, login_required
    from sqlalchemy.dialects.sqlite import insert
    from backend.models import SearchCount

    try:
        from backend.services.loader import load_engine
        from backend.routes.search_routes import create_search_blueprint

        engine, normalizer = load_engine(app.config["FEATURES_FOLDER"])
        bp = create_search_blueprint(
            engine, normalizer=normalizer, auth_decorators=[login_required]
        )

        @bp.after_request
        def count_successful_search(response):
            if (
                request.endpoint == f"{bp.name}.searchresource"
                and request.method == "POST"
                and response.status_code == 200
                and current_user.is_authenticated
            ):
                # Une seule écriture SQLite, même si deux recherches aboutissent en parallèle.
                counter = insert(SearchCount).values(user_id=current_user.id, total=1)
                db.session.execute(counter.on_conflict_do_update(
                    index_elements=[SearchCount.user_id],
                    set_={"total": SearchCount.total + 1},
                ))
                db.session.commit()
            return response

        app.register_blueprint(bp, url_prefix="/api")
    except (ImportError, FileNotFoundError, ValueError) as e:
        log.warning("API de recherche non montée : %s", e)


if __name__ == "__main__":
    create_app().run(debug=True)