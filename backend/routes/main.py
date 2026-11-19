from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from backend.models import Image, Category

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("auth.login"))


@main_bp.route("/dashboard")
@login_required
def dashboard():
    n_images = Image.query.filter_by(user_id=current_user.id).count()
    n_categories = Category.query.filter_by(user_id=current_user.id).count()
    return render_template("dashboard.html", n_images=n_images,
                           n_categories=n_categories)