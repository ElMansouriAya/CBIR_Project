import os
import uuid
import io
import zipfile
from flask import (Blueprint, render_template, request, redirect, url_for,
                   flash, send_file, abort, current_app)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
from backend.extensions import db
from backend.models import Image

gallery_bp = Blueprint("gallery", __name__, url_prefix="/gallery")


def _allowed(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def _is_real_image(file_storage):
    """Vérifie que le contenu est vraiment une image (pas seulement l'extension)."""
    try:
        PILImage.open(file_storage.stream).verify()
        return True
    except Exception:
        return False
    finally:
        file_storage.stream.seek(0)


def get_user_image_or_404(image_id):
    """Retourne l'image seulement si elle appartient à l'utilisateur connecté."""
    img = db.session.get(Image, image_id)
    if img is None or img.user_id != current_user.id:
        abort(404)
    return img


@gallery_bp.route("/")
@login_required
def index():
    images = (Image.query.filter_by(user_id=current_user.id)
              .order_by(Image.created_at.desc()).all())
    return render_template("gallery.html", images=images)


@gallery_bp.route("/upload", methods=["POST"])
@login_required
def upload():
    files = request.files.getlist("images")
    saved, rejected = 0, []

    for f in files:
        if not f or not f.filename:
            continue
        original = secure_filename(f.filename)
        if not original or not _allowed(original) or not _is_real_image(f):
            rejected.append(f.filename)
            continue

        ext = original.rsplit(".", 1)[-1].lower()
        stored_name = f"{uuid.uuid4().hex}.{ext}"      # nom unique, évite les collisions
        user_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], str(current_user.id))
        os.makedirs(user_dir, exist_ok=True)
        f.save(os.path.join(user_dir, stored_name))

        db.session.add(Image(
            filename=original,
            filepath=os.path.join(str(current_user.id), stored_name),  # relatif à UPLOAD_FOLDER
            user_id=current_user.id,
        ))
        saved += 1

    db.session.commit()
    if saved:
        flash(f"{saved} image(s) ajoutée(s).", "success")
    if rejected:
        flash("Fichiers refusés : " + ", ".join(rejected), "warning")
    if not saved and not rejected:
        flash("Aucun fichier sélectionné.", "info")
    return redirect(url_for("gallery.index"))


@gallery_bp.route("/file/<int:image_id>")
@login_required
def serve(image_id):
    """Sert l'image à l'affichage (protégé : seul le propriétaire y accède)."""
    img = get_user_image_or_404(image_id)
    return send_file(os.path.join(current_app.config["UPLOAD_FOLDER"], img.filepath))


@gallery_bp.route("/download/<int:image_id>")
@login_required
def download(image_id):
    img = get_user_image_or_404(image_id)
    return send_file(os.path.join(current_app.config["UPLOAD_FOLDER"], img.filepath),
                     as_attachment=True, download_name=img.filename)


@gallery_bp.route("/download-multiple", methods=["POST"])
@login_required
def download_multiple():
    ids = request.form.getlist("selected", type=int)
    imgs = Image.query.filter(Image.id.in_(ids), Image.user_id == current_user.id).all()
    if not imgs:
        flash("Aucune image sélectionnée.", "info")
        return redirect(url_for("gallery.index"))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, img in enumerate(imgs):
            path = os.path.join(current_app.config["UPLOAD_FOLDER"], img.filepath)
            if os.path.exists(path):
                z.write(path, arcname=f"{i}_{img.filename}")   # préfixe : évite les doublons
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="images.zip",
                     mimetype="application/zip")


@gallery_bp.route("/delete", methods=["POST"])
@login_required
def delete():
    """Supprime une ou plusieurs images (cases cochées)."""
    ids = request.form.getlist("selected", type=int)
    imgs = Image.query.filter(Image.id.in_(ids), Image.user_id == current_user.id).all()

    for img in imgs:
        path = os.path.join(current_app.config["UPLOAD_FOLDER"], img.filepath)
        if os.path.exists(path):
            os.remove(path)                       # supprime le fichier sur disque
        # Les images transformées dérivées perdent leur lien avec l'originale
        Image.query.filter_by(parent_image_id=img.id).update({"parent_image_id": None})
        db.session.delete(img)

    db.session.commit()
    flash(f"{len(imgs)} image(s) supprimée(s).", "success" if imgs else "info")
    return redirect(url_for("gallery.index"))