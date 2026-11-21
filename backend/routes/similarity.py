from flask import Blueprint, abort, current_app, render_template, url_for
from flask_login import login_required

from backend.extensions import db
from backend.models import Image

similarity_bp = Blueprint("similarity", __name__, url_prefix="/recherche")


@similarity_bp.get("/")
@similarity_bp.get("/<int:image_id>")
@login_required
def index(image_id=None):
    image = None
    if image_id is not None:
        if not 0 < image_id < 2**63:
            abort(404)
        image = db.one_or_404(db.select(Image).filter_by(id=image_id, is_dataset=True))

    search_endpoint = "cbir_search.searchresource"
    search_url = (
        url_for(search_endpoint) if search_endpoint in current_app.view_functions else None
    )
    feedback_url = url_for("cbir_search.feedbackresource") if search_url else None
    return render_template("similarity.html", image=image,
                           search_url=search_url, feedback_url=feedback_url)
