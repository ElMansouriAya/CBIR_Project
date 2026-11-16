"""Lance une API de démonstration (blueprint de recherche monté sur /api).

Sert à tester le module C seul, sans l'application de A :

    python scripts/make_synthetic_data.py            # une fois
    python scripts/run_demo_api.py                   # données synthétiques

    curl -s -X POST localhost:5001/api/search -H "Content-Type: application/json" \\
         -d '{"query_image_id": 1, "scope": 5}'

Avec les vraies données de B : python scripts/run_demo_api.py --features features
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask  # noqa: E402

from backend import cbir_config as config  # noqa: E402
from backend.routes.search_routes import create_search_blueprint  # noqa: E402
from backend.services.loader import load_engine  # noqa: E402


def create_demo_app(features_dir):
    engine, normalizer = load_engine(features_dir)
    app = Flask(__name__)
    app.register_blueprint(create_search_blueprint(engine, normalizer=normalizer),
                           url_prefix="/api")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.SYNTHETIC_DIR)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    create_demo_app(args.features).run(host="127.0.0.1", port=args.port, debug=False)
