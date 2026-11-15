from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from backend.extensions import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey("category.id"), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    parent = db.relationship("Category", remote_side=[id], backref="children")


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # = image_id du contrat
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey("category.id"), nullable=True)
    # NULL pour les images du dataset GHIM (bibliothèque de référence)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    is_dataset = db.Column(db.Boolean, default=False, nullable=False)
    dataset_category = db.Column(db.String(50), nullable=True)  # ex. "Animals"
    parent_image_id = db.Column(db.Integer, db.ForeignKey("image.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)