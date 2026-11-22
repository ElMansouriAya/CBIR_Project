from io import BytesIO
import json
import re

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import text

from backend.app import create_app
from backend.cbir_config import N_DIMS
from backend.config import Config
from backend.extensions import db
from backend.models import Image, User
from scripts.import_dataset import main


@pytest.fixture
def storage(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    features = tmp_path / "features"
    features.mkdir()
    monkeypatch.setattr(Config, "DATASET_FOLDER", str(dataset))
    monkeypatch.setattr(Config, "FEATURES_FOLDER", str(features))
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(Config, "TRANSFORMED_FOLDER", str(tmp_path / "transformed"))
    monkeypatch.setattr(Config, "SQLALCHEMY_DATABASE_URI",
                        f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    return dataset, features / "image_mapping.json"


def write_mapping(storage, rows):
    _, mapping_path = storage
    mapping_path.write_text(json.dumps(rows), encoding="utf-8")


def make_mapping(storage, images):
    dataset, _ = storage
    rows = []
    for index, (image_id, category) in enumerate(images):
        filename = f"{image_id}.jpg"
        path = dataset / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.new("RGB", (16, 12)).save(path)
        rows.append({"index": index, "image_id": image_id, "filename": filename,
                     "category": category, "path": f"dataset/{category}/{filename}"})
    write_mapping(storage, rows)
    return rows


def login(client, app):
    with app.app_context():
        user = User(username="test-user", email="test@example.com")
        user.set_password("test-password")
        db.session.add(user)
        db.session.commit()
        user_id = user.id
    response = client.post("/login", data={"username": "test-user", "password": "test-password"})
    assert response.status_code == 302
    return user_id


def test_landing_uses_only_curated_previews_from_the_dataset(storage):
    dataset, _ = storage
    app = create_app()
    client = app.test_client()
    url = "/accueil/images/falaise.webp"
    empty = client.get("/")
    assert empty.status_code == 200
    assert url not in empty.get_data(as_text=True)
    assert client.get(url).status_code == 404

    source = dataset / "sea shores" / "9_4501.jpg"
    source.parent.mkdir(parents=True)
    PILImage.new("RGB", (16, 12)).save(source)
    assert url in client.get("/").get_data(as_text=True)
    response = client.get(url)
    assert response.status_code == 200
    assert response.mimetype == "image/webp"
    with PILImage.open(BytesIO(response.data)) as preview:
        assert preview.size == (720, 420)
    assert client.get("/accueil/images/9_4501.webp").status_code == 404

    login(client, app)
    assert client.get("/").location == "/dashboard"


def test_import_keeps_reference_ids_on_rerun(storage, capsys):
    make_mapping(storage, [(1, "aircraft"), (9525, "sea shores")])
    main([])
    assert "2 image(s) ajoutée(s)" in capsys.readouterr().out
    app = create_app()

    def snapshot():
        with app.app_context():
            images = db.session.scalars(db.select(Image).order_by(Image.id)).all()
            return [(im.id, im.is_dataset, im.user_id, im.filepath, im.created_at) for im in images]

    original = snapshot()
    assert [row[:4] for row in original] == [
        (1, True, None, "aircraft/1.jpg"),
        (9525, True, None, "sea shores/9525.jpg"),
    ]
    main(["--mapping", str(storage[1])])
    assert "0 image(s) ajoutée(s), 2 déjà présente(s)" in capsys.readouterr().out
    assert snapshot() == original


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"index": True}, "index attendu"),
        ({"image_id": 11}, "en double"),
        ({"image_id": "12"}, "entier positif"),
        ({"category": ""}, "doivent être renseignés"),
        ({"path": "dataset/../12.jpg"}, "chemin JPEG relatif"),
        ({"path": "dataset/C:/12.jpg"}, "chemin JPEG relatif"),
        ({"path": "dataset/introuvable/12.jpg"}, "image_id 12"),
    ],
)
def test_invalid_lot_leaves_database_empty(storage, capsys, changes, message):
    app = create_app()
    rows = make_mapping(storage, [(11, "aircraft"), (12, "aircraft")])
    rows[1].update(changes)
    write_mapping(storage, rows)
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 1
    assert message in capsys.readouterr().err
    with app.app_context():
        assert Image.query.count() == 0


def test_invalid_jpeg_is_refused_before_database_creation(storage, capsys):
    dataset, _ = storage
    make_mapping(storage, [(11, "aircraft")])
    PILImage.new("RGB", (16, 12)).save(dataset / "aircraft/11.jpg", format="PNG")
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 1
    assert "n'est pas un JPEG" in capsys.readouterr().err
    assert not (dataset.parent / "test.db").exists()


def test_collision_with_personal_image_aborts_whole_import(storage, capsys):
    make_mapping(storage, [(1, "aircraft"), (2, "aircraft")])
    app = create_app()
    user_id = login(app.test_client(), app)
    with app.app_context():
        db.session.add(Image(id=2, filename="perso.jpg", filepath="perso.jpg", user_id=user_id))
        db.session.commit()
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 1
    assert "image_id 2 est déjà attribué" in capsys.readouterr().err
    with app.app_context():
        assert [(im.id, im.user_id, im.is_dataset) for im in Image.query.all()] == [
            (2, user_id, False)
        ]


def test_sqlite_failure_rolls_back_previous_inserts(storage, capsys):
    make_mapping(storage, [(1, "aircraft"), (2, "aircraft")])
    app = create_app()
    with app.app_context():
        db.session.execute(text("""
            CREATE TRIGGER reject_second_image BEFORE INSERT ON image
            WHEN NEW.id = 2
            BEGIN
                SELECT RAISE(ABORT, 'insertion refusee');
            END
        """))
        db.session.commit()
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 1
    assert "insertion refusee" in capsys.readouterr().err
    with app.app_context():
        assert Image.query.count() == 0


def test_dataset_page_filters_and_serves_only_ghim_images(storage):
    images = [(image_id, "sea shores") for image_id in range(31, 5, -1)]
    make_mapping(storage, images + [(1, "aircraft")])
    app = create_app()
    client = app.test_client()
    for url in ("/dataset/", "/dataset/file/1"):
        response = client.get(url)
        assert response.status_code == 302
        assert response.location.startswith("/login?")

    user_id = login(client, app)
    assert "La bibliothèque GHIM est vide" in client.get("/dataset/").get_data(as_text=True)
    main([])
    with app.app_context():
        db.session.add(Image(id=2, filename="perso.jpg", filepath="perso.jpg", user_id=user_id))
        db.session.commit()

    def visible_ids(response):
        assert response.status_code == 200
        return [int(value) for value in re.findall(
            r'src="/dataset/file/(\d+)"', response.get_data(as_text=True)
        )]

    assert visible_ids(client.get("/dataset/")) == [1, *range(6, 30)]
    first = client.get("/dataset/", query_string={"category": "sea shores"})
    assert visible_ids(first) == list(range(6, 31))
    assert "category=sea+shores&amp;page=2" in first.get_data(as_text=True)
    assert visible_ids(client.get("/dataset/", query_string={"category": "sea shores", "page": 2})) == [31]
    ordered_ids = sorted(range(6, 32), key=lambda image_id: f"{image_id}.jpg", reverse=True)
    order = {"category": "sea shores", "sort": "name-desc"}
    sorted_page = client.get("/dataset/", query_string=order)
    assert visible_ids(sorted_page) == ordered_ids[:25]
    assert "category=sea+shores&amp;page=2&amp;sort=name-desc" in sorted_page.get_data(as_text=True)
    assert visible_ids(client.get("/dataset/", query_string={**order, "page": 2})) == ordered_ids[25:]
    assert "Aucune image dans cette catégorie" in client.get(
        "/dataset/?category=unknown"
    ).get_data(as_text=True)
    assert client.get("/dataset/?page=3").status_code == 404
    assert client.get("/dataset/file/2").status_code == 404

    image = client.get("/dataset/file/1")
    assert image.status_code == 200
    assert image.mimetype == "image/jpeg"
    assert image.cache_control.private
    assert image.data == (storage[0] / "aircraft/1.jpg").read_bytes()
    image.close()

    with app.app_context():
        db.session.get(Image, 1).filepath = "../aircraft/1.jpg"
        db.session.commit()
    assert client.get("/dataset/file/1").status_code == 404


def test_m3_result_resolves_to_imported_image(storage):
    dataset, mapping = storage
    rows = make_mapping(storage, [(401, "aircraft"), (9525, "yacht")])
    vectors = np.stack([np.zeros(N_DIMS), np.ones(N_DIMS)]).astype(np.float32)
    np.save(mapping.with_name("descriptors_raw.npy"), vectors)
    main([])
    app = create_app()
    client = app.test_client()
    login(client, app)
    response = client.post("/api/search", json={"query_image_id": rows[1]["image_id"], "scope": 1})
    assert response.status_code == 200
    result = response.json["results"][0]
    assert (result["image_id"], result["category"]) == (9525, "yacht")
    image = client.get(f"/dataset/file/{result['image_id']}")
    assert image.status_code == 200
    assert image.data == (dataset / "yacht/9525.jpg").read_bytes()
    image.close()
