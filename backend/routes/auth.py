from urllib.parse import urlparse
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from backend.extensions import db
from backend.models import User

auth_bp = Blueprint("auth", __name__)


def _safe_next(target):
    """N'accepte que les redirections internes (évite l'open redirect)."""
    if target and target.startswith("/") and not urlparse(target).netloc:
        return target
    return url_for("main.dashboard")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        error = None
        if not username or not email or not password:
            error = "Tous les champs sont obligatoires."
        elif len(password) < 8:
            error = "Le mot de passe doit contenir au moins 8 caractères."
        elif password != confirm:
            error = "Les mots de passe ne correspondent pas."
        elif User.query.filter_by(username=username).first():
            error = "Ce nom d'utilisateur existe déjà."
        elif User.query.filter_by(email=email).first():
            error = "Cet email est déjà utilisé."

        if error:
            flash(error, "danger")
        else:
            user = User(username=username, email=email)
            user.set_password(password)   # mot de passe haché, jamais en clair
            db.session.add(user)
            db.session.commit()
            flash("Compte créé. Vous pouvez vous connecter.", "success")
            return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(_safe_next(request.args.get("next")))
        flash("Identifiants incorrects.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Vous êtes déconnecté.", "info")
    return redirect(url_for("auth.login"))