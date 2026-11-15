import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "gallery.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "image_storage", "uploads")
    TRANSFORMED_FOLDER = os.path.join(BASE_DIR, "image_storage", "transformed")
    DATASET_FOLDER = os.path.join(BASE_DIR, "image_storage", "dataset")
    FEATURES_FOLDER = os.path.join(BASE_DIR, "features")
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024