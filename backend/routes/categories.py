from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, abort)
from flask_login import login_required, current_user
from sqlalchemy import func
from backend.extensions import db
from backend.models import Category, Image

categories_bp = Blueprint("categories", __name__, url_prefix="/categories")

MAX_DEPTH = 5   # profondeur maximale de la hiérarchie (racine = niveau 1)


# ---------- Fonctions utilitaires (réutilisées par gallery.py) ----------

def user_categories():
    """Toutes les catégories de l'utilisateur connecté, triées par nom."""
    return (Category.query.filter_by(user_id=current_user.id)
            .order_by(Category.name).all())


def get_category_or_404(cat_id):
    """Retourne la catégorie seulement si elle appartient à l'utilisateur."""
    cat = db.session.get(Category, cat_id)
    if cat is None or cat.user_id != current_user.id:
        abort(404)
    return cat


def flat_categories(cats):
    """Liste aplatie [(catégorie, profondeur)] dans l'ordre de l'arbre.
    Sert aux menus déroulants (indentation selon la profondeur)."""
    by_parent = {}
    for c in cats:
        by_parent.setdefault(c.parent_id, []).append(c)
    result = []

    def walk(parent_id, depth):
        for c in by_parent.get(parent_id, []):
            result.append((c, depth))
            walk(c.id, depth + 1)

    walk(None, 0)
    return result


def descendant_ids(root_id, cats):
    """Identifiants de tous les descendants d'une catégorie (sans elle-même)."""
    by_parent = {}
    for c in cats:
        by_parent.setdefault(c.parent_id, []).append(c.id)
    found, stack = [], [root_id]
    while stack:
        for child in by_parent.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def build_tree(cats, counts):
    """Construit l'arbre imbriqué pour l'affichage."""
    nodes = {c.id: {"cat": c, "count": counts.get(c.id, 0), "children": []}
             for c in cats}
    roots = []
    for c in cats:
        if c.parent_id in nodes:
            nodes[c.parent_id]["children"].append(nodes[c.id])
        else:
            roots.append(nodes[c.id])
    return roots


def _name_taken(name, parent_id, cats, exclude_id=None):
    """Vrai si une catégorie sœur porte déjà ce nom (insensible à la casse)."""
    return any(c.parent_id == parent_id and c.id != exclude_id
               and c.name.lower() == name.lower() for c in cats)


# ---------- Routes ----------

@categories_bp.route("/")
@login_required
def index():
    cats = user_categories()
    counts = dict(
        db.session.query(Image.category_id, func.count(Image.id))
        .filter(Image.user_id == current_user.id, Image.category_id.isnot(None))
        .group_by(Image.category_id).all()
    )
    uncategorized = Image.query.filter_by(
        user_id=current_user.id, category_id=None).count()
    return render_template(
        "categories.html",
        tree=build_tree(cats, counts),
        options=flat_categories(cats),
        uncategorized=uncategorized,
        max_depth=MAX_DEPTH,
    )


@categories_bp.route("/create", methods=["POST"])
@login_required
def create():
    name = request.form.get("name", "").strip()
    parent_id = request.form.get("parent_id", type=int)
    cats = user_categories()

    depth_of = {c.id: d for c, d in flat_categories(cats)}
    error = None
    if not name:
        error = "Le nom de la catégorie est obligatoire."
    elif len(name) > 100:
        error = "Le nom est trop long (100 caractères maximum)."
    elif parent_id is not None:
        get_category_or_404(parent_id)                 # appartient à l'utilisateur ?
        if depth_of.get(parent_id, 0) + 1 >= MAX_DEPTH:
            error = f"Profondeur maximale atteinte ({MAX_DEPTH} niveaux)."
    if not error and _name_taken(name, parent_id, cats):
        error = "Une catégorie portant ce nom existe déjà à cet endroit."

    if error:
        flash(error, "danger")
    else:
        db.session.add(Category(name=name, parent_id=parent_id,
                                user_id=current_user.id))
        db.session.commit()
        flash(f"Catégorie « {name} » créée.", "success")
    return redirect(url_for("categories.index"))


@categories_bp.route("/<int:cat_id>/rename", methods=["POST"])
@login_required
def rename(cat_id):
    cat = get_category_or_404(cat_id)
    name = request.form.get("name", "").strip()
    if not name or len(name) > 100:
        flash("Nom invalide.", "danger")
    elif _name_taken(name, cat.parent_id, user_categories(), exclude_id=cat.id):
        flash("Une catégorie portant ce nom existe déjà à cet endroit.", "danger")
    else:
        cat.name = name
        db.session.commit()
        flash("Catégorie renommée.", "success")
    return redirect(url_for("categories.index"))


@categories_bp.route("/<int:cat_id>/delete", methods=["POST"])
@login_required
def delete(cat_id):
    """Supprime la catégorie et ses sous-catégories.
    Les images concernées sont conservées (elles deviennent sans catégorie)."""
    cat = get_category_or_404(cat_id)
    ids = descendant_ids(cat.id, user_categories()) + [cat.id]

    Image.query.filter(Image.category_id.in_(ids)).update(
        {"category_id": None}, synchronize_session=False)
    Category.query.filter(Category.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    flash(f"Catégorie supprimée ({len(ids)} élément(s)). Les images sont conservées.",
          "success")
    return redirect(url_for("categories.index"))


@categories_bp.route("/assign", methods=["POST"])
@login_required
def assign():
    """Déplace les images cochées dans une catégorie (ou « sans catégorie »)."""
    ids = request.form.getlist("selected", type=int)
    target = request.form.get("category_id", "")
    back = request.form.get("current_filter", "")

    cat_id = None
    if target and target != "none":
        cat_id = get_category_or_404(int(target)).id

    imgs = Image.query.filter(Image.id.in_(ids),
                              Image.user_id == current_user.id).all()
    for img in imgs:
        img.category_id = cat_id
    db.session.commit()
    flash(f"{len(imgs)} image(s) déplacée(s).", "success" if imgs else "info")
    return redirect(url_for("gallery.index", category=back or None))