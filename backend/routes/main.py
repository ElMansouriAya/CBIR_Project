from io import BytesIO

from flask import Blueprint, abort, current_app, redirect, render_template, send_file, url_for
from flask_login import login_required, current_user
from PIL import Image as PILImage, ImageOps
from sqlalchemy import func
from backend.dataset import resolve_dataset_jpeg
from backend.extensions import db
from backend.models import Image, Category, SearchCount

main_bp = Blueprint("main", __name__)

LANDING_PHOTOS = {
    "falaise": ("sea shores/9_4501.jpg", 720, 420),
    "vagues": ("sea shores/9_4502.jpg", 360, 260),
    "crique": ("sea shores/9_4505.jpg", 360, 260),
    "plage": ("sea shores/9_4503.jpg", 360, 260),
    "montagne": ("mountains/5_2502.jpg", 560, 360),
    "voilier": ("yacht/15_7501.jpg", 420, 420),
    "coucher": ("sunset/13_6501.jpg", 420, 420),
}


def _landing_source(relative_path):
    try:
        return resolve_dataset_jpeg(current_app.config["DATASET_FOLDER"], relative_path)
    except (OSError, RuntimeError, ValueError):
        return None


@main_bp.get("/")
def landing():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    photos = {
        name: (width, height)
        for name, (relative_path, width, height) in LANDING_PHOTOS.items()
        if _landing_source(relative_path) is not None
    }
    return render_template("landing.html", landing_photos=photos)


@main_bp.get("/accueil/images/<name>.webp")
def landing_photo(name):
    # L'accueil ne publie que les aperçus choisis ci-dessus.
    if name not in LANDING_PHOTOS:
        abort(404)
    relative_path, width, height = LANDING_PHOTOS[name]
    source = _landing_source(relative_path)
    if source is None:
        abort(404)
    output = BytesIO()
    try:
        modified = source.stat().st_mtime
        with PILImage.open(source) as image:
            thumbnail = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (width, height))
            thumbnail.save(output, format="WEBP", quality=80)
    except (OSError, ValueError):
        abort(404)
    output.seek(0)
    return send_file(output, mimetype="image/webp", last_modified=modified, max_age=86400)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    personal_images = Image.query.filter_by(user_id=current_user.id)
    n_images = personal_images.count()
    n_categories = Category.query.filter_by(user_id=current_user.id).count()
    search_count = db.session.get(SearchCount, current_user.id)
    category_samples = db.session.execute(
        db.select(Image.dataset_category, func.min(Image.id))
        .where(Image.is_dataset.is_(True))
        .group_by(Image.dataset_category)
        .order_by(Image.dataset_category)
    ).all()
    by_category = dict(category_samples)
    preferred = ("sea shores", "mountains", "yacht", "sunset", "valleys")
    preview_ids = [by_category[name] for name in preferred if name in by_category]
    preview_ids.extend(image_id for _, image_id in category_samples if image_id not in preview_ids)
    preview_ids = preview_ids[:5]
    previews = {image.id: image for image in Image.query.filter(Image.id.in_(preview_ids))}
    ghim_previews = [previews[image_id] for image_id in preview_ids if image_id in previews]
    recent_images = personal_images.order_by(Image.created_at.desc(), Image.id.desc()).limit(12).all()
    return render_template("dashboard.html", n_images=n_images,
                           n_categories=n_categories,
                           n_searches=search_count.total if search_count else 0,
                           ghim_previews=ghim_previews, recent_images=recent_images)
