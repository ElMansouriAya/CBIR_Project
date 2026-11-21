from flask import Blueprint, abort, current_app, render_template, request, send_file
from flask_login import login_required

from backend.dataset import resolve_dataset_jpeg
from backend.extensions import db
from backend.models import Image
from backend.services.descriptors import image_descriptors

dataset_bp = Blueprint("dataset", __name__, url_prefix="/dataset")
PAGE_SIZE = 25


def get_dataset_image_or_404(image_id):
    if not 0 < image_id < 2**63:
        abort(404)
    return db.one_or_404(db.select(Image).filter_by(id=image_id, is_dataset=True))


@dataset_bp.get("/")
@login_required
def index():
    catalog = Image.query.filter_by(is_dataset=True)
    category_rows = (
        catalog.with_entities(Image.dataset_category)
        .filter(Image.dataset_category.is_not(None))
        .distinct().order_by(Image.dataset_category)
    )
    category = request.args.get("category", "")
    query = catalog.filter_by(dataset_category=category) if category else catalog
    page = request.args.get("page", 1, type=int)
    if not 1 <= page <= 2**31 - 1:
        abort(404)
    pagination = query.order_by(Image.id).paginate(page=page, per_page=PAGE_SIZE)
    return render_template("dataset.html", pagination=pagination,
                           categories=[name for (name,) in category_rows], category=category)


@dataset_bp.get("/file/<int:image_id>")
@login_required
def serve(image_id):
    image = get_dataset_image_or_404(image_id)
    try:
        path = resolve_dataset_jpeg(current_app.config["DATASET_FOLDER"], image.filepath)
        response = send_file(path, mimetype="image/jpeg")
    except (OSError, RuntimeError, ValueError):
        abort(404)
    response.cache_control.private = True
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@dataset_bp.get("/<int:image_id>/descripteurs")
@login_required
def descriptors(image_id):
    image = get_dataset_image_or_404(image_id)
    values = image_descriptors(current_app.extensions["ghim_descriptors"], image)
    return render_template("descriptors.html", image=image, values=values)
