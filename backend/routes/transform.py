import os
import uuid
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, current_app)
from flask_login import login_required, current_user
from PIL import Image as PILImage, ImageOps
from backend.extensions import db
from backend.models import Image
from backend.routes.gallery import get_user_image_or_404

transform_bp = Blueprint("transform", __name__, url_prefix="/transform")

MAX_SIDE = 8000                       # taille maximale d'un côté (en pixels)
FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


def _full_path(img):
    return os.path.join(current_app.config["UPLOAD_FOLDER"], img.filepath)


def _check_size(w, h):
    if w < 1 or h < 1 or w > MAX_SIDE or h > MAX_SIDE:
        raise ValueError(f"Dimensions invalides (1 à {MAX_SIDE} px par côté).")


def _apply(im, op, form):
    """Applique une transformation Pillow. Retourne (image, suffixe de nom)."""
    w, h = im.size

    if op == "crop":
        x = int(float(form["x"]))
        y = int(float(form["y"]))
        cw = int(float(form["width"]))
        ch = int(float(form["height"]))
        # On borne la zone aux limites de l'image
        x, y = max(0, x), max(0, y)
        cw, ch = min(cw, w - x), min(ch, h - y)
        _check_size(cw, ch)
        return im.crop((x, y, x + cw, y + ch)), "crop"

    if op == "resize":
        if form.get("mode") == "percent":
            p = float(form["percent"])
            if not 1 <= p <= 500:
                raise ValueError("Le pourcentage doit être entre 1 et 500.")
            nw, nh = round(w * p / 100), round(h * p / 100)
        else:
            nw, nh = int(form["new_width"]), int(form["new_height"])
        _check_size(nw, nh)
        return im.resize((nw, nh), PILImage.LANCZOS), "resize"

    if op == "rotate":
        angle = float(form["angle"])
        if not -360 <= angle <= 360:
            raise ValueError("L'angle doit être entre -360 et 360.")
        # Angle positif = sens horaire (comme l'interface) ; expand agrandit le canevas
        fill = (255, 255, 255, 0) if im.mode == "RGBA" else (255, 255, 255)
        out = im.rotate(-angle, expand=True, resample=PILImage.BICUBIC, fillcolor=fill)
        _check_size(*out.size)
        return out, "rotate"

    if op == "flip":
        if form.get("direction") == "v":
            return ImageOps.flip(im), "flipV"
        return ImageOps.mirror(im), "flipH"

    raise ValueError("Opération inconnue.")


@transform_bp.route("/")
@login_required
def index():
    images = (Image.query.filter_by(user_id=current_user.id)
              .order_by(Image.created_at.desc()).all())
    return render_template("transform_index.html", images=images)


@transform_bp.route("/<int:image_id>")
@login_required
def editor(image_id):
    img = get_user_image_or_404(image_id)
    with PILImage.open(_full_path(img)) as im:
        width, height = ImageOps.exif_transpose(im).size
    parent = None
    if img.parent_image_id:
        p = db.session.get(Image, img.parent_image_id)
        parent = p if p and p.user_id == current_user.id else None
    versions = (Image.query.filter_by(parent_image_id=img.id,
                                      user_id=current_user.id)
                .order_by(Image.created_at.desc()).all())
    return render_template("transform_editor.html", img=img, width=width,
                           height=height, parent=parent, versions=versions)


@transform_bp.route("/<int:image_id>/apply", methods=["POST"])
@login_required
def apply(image_id):
    img = get_user_image_or_404(image_id)
    op = request.form.get("operation", "")

    try:
        with PILImage.open(_full_path(img)) as src:
            im = ImageOps.exif_transpose(src)      # même orientation que le navigateur
            im.load()
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        result, suffix = _apply(im, op, request.form)

        ext = img.filepath.rsplit(".", 1)[-1].lower()
        fmt = FORMATS.get(ext)
        if fmt is None:                            # ex. GIF : on enregistre en PNG
            ext, fmt = "png", "PNG"
        if fmt == "JPEG" and result.mode != "RGB":
            result = result.convert("RGB")

        user_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], str(current_user.id))
        os.makedirs(user_dir, exist_ok=True)
        stored = f"{uuid.uuid4().hex}.{ext}"
        result.save(os.path.join(user_dir, stored), format=fmt)

        stem = img.filename.rsplit(".", 1)[0]
        new_img = Image(
            filename=f"{stem}_{suffix}.{ext}",
            filepath=os.path.join(str(current_user.id), stored),
            user_id=current_user.id,
            category_id=img.category_id,           # hérite de la catégorie
            parent_image_id=img.id,                # lien vers l'originale
        )
        db.session.add(new_img)
        db.session.commit()
        flash(f"Nouvelle image créée : {new_img.filename}", "success")
        return redirect(url_for("transform.editor", image_id=new_img.id))

    except (ValueError, KeyError, TypeError) as e:
        flash(str(e) if isinstance(e, ValueError) else "Paramètres invalides.", "danger")
    except OSError:
        flash("Impossible de lire ou d'enregistrer l'image.", "danger")
    return redirect(url_for("transform.editor", image_id=img.id))