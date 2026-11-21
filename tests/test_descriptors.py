import numpy as np
import pytest
from PIL import Image as PILImage

from backend.app import create_app
from backend.cbir_config import BLOCKS, N_DIMS
from backend.extensions import db
from backend.models import Image
from scripts.import_dataset import main

from test_dataset import login, make_mapping, storage, write_mapping


def make_descriptor_vector(hue):
    vector = np.zeros(N_DIMS, dtype=np.float32)
    start, _ = BLOCKS["hsv_hist"]
    vector[start + int(hue * 8) * 16 + 15] = 1
    start, end = BLOCKS["color_moments"]
    vector[start:end] = [hue, 0, 0, 1, 0, 0, 1, 0, 0]
    start, end = BLOCKS["dominant_colors"]
    vector[start:end] = [hue, 1, 1, hue, 0.9, 0.8, hue, 0.7, 0.6]
    start, end = BLOCKS["glcm"]
    vector[start:end] = [0] * 4 + [1] * 12
    start, _ = BLOCKS["lbp"]
    vector[start + 8] = 1
    return vector


def test_descriptors_require_login_and_match_ghim_id_not_row_number(storage):
    dataset, mapping = storage
    rows = make_mapping(storage, [(9525, "sea shores"), (7, "aircraft")])
    for row, color in zip(rows, ("blue", "red")):
        PILImage.new("RGB", (16, 12), color=color).save(dataset.parent / row["path"])
    np.save(mapping.with_name("descriptors_raw.npy"),
            np.stack([make_descriptor_vector(2 / 3), make_descriptor_vector(0)]))
    main([])
    app = create_app()
    client = app.test_client()
    url = "/dataset/9525/descripteurs"
    response = client.get(url)
    assert response.status_code == 302
    assert response.location.startswith("/login?")
    user_id = login(client, app)

    library = client.get("/dataset/").get_data(as_text=True)
    assert f'href="{url}"' in library
    assert "Voir les descripteurs" in library

    response = client.get(url)
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'src="/dataset/file/9525"' in page
    assert "sea shores" in page
    assert "#0000FF" in page
    assert "#FF0000" not in page
    assert "#FF0000" in client.get("/dataset/7/descripteurs").get_data(as_text=True)
    assert page.count('class="descriptor-bin"') == 128
    assert page.count("<meter ") == 10
    for heading in ("Histogramme HSV", "Couleurs dominantes", "Moments de couleur",
                    "GLCM par angle", "Histogramme LBP"):
        assert heading in page
    with client.get("/dataset/file/9525") as jpeg:
        assert jpeg.data == (dataset / "sea shores/9525.jpg").read_bytes()

    with app.app_context():
        db.session.add(Image(id=3, filename="private.jpg", filepath="private.jpg", user_id=user_id))
        db.session.commit()
    assert client.get("/dataset/3/descripteurs").status_code == 404


@pytest.mark.parametrize("reason", ["missing_matrix", "different_image", "invalid_histogram"])
def test_descriptors_unavailable_keep_image_summary(storage, reason):
    _, mapping = storage
    rows = make_mapping(storage, [(7, "aircraft")])
    main([])
    if reason != "missing_matrix":
        vector = make_descriptor_vector(0)
        if reason == "different_image":
            rows[0]["path"] = "dataset/aircraft/other.jpg"
            write_mapping(storage, rows)
        else:
            start, end = BLOCKS["hsv_hist"]
            vector[start:end] = 0
        np.save(mapping.with_name("descriptors_raw.npy"), vector[None, :])

    app = create_app()
    client = app.test_client()
    login(client, app)
    page = client.get("/dataset/7/descripteurs")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'src="/dataset/file/7"' in html
    assert "aircraft" in html
    assert "Les descripteurs sont indisponibles pour cette image" in html
    assert "Histogramme HSV" not in html
