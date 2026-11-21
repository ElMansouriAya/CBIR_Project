import numpy as np
from PIL import Image as PILImage

from backend.app import create_app
from backend.cbir_config import N_DIMS
from backend.extensions import db
from backend.models import Image
from scripts.import_dataset import main

from test_dataset import login, make_mapping, storage


def test_ghim_search_and_feedback_keep_result_ids_and_order(storage):
    dataset, mapping = storage
    rows = make_mapping(storage, [(401, "ships"), (20, "ships"), (9525, "yacht")])
    for row, color in zip(rows, ("blue", "red", "green")):
        PILImage.new("RGB", (16, 12), color=color).save(dataset.parent / row["path"])
    vectors = np.repeat(np.array([[0], [1], [5]], dtype=np.float32), N_DIMS, axis=1)
    np.save(mapping.with_name("descriptors_raw.npy"), vectors)
    main([])

    app = create_app()
    client = app.test_client()
    assert client.get("/recherche/").status_code == 302
    assert client.get("/recherche/401").status_code == 302
    user_id = login(client, app)
    with app.app_context():
        db.session.add(Image(id=3, filename="personal.jpg", filepath="personal.jpg", user_id=user_id))
        db.session.commit()
    assert client.get("/recherche/3").status_code == 404
    assert "Ouvrir la bibliothèque GHIM" in client.get("/recherche/").get_data(as_text=True)

    library = client.get("/dataset/").get_data(as_text=True)
    assert 'href="/recherche/401"' in library
    assert "Trouver des images similaires" in library
    page = client.get("/recherche/401").get_data(as_text=True)
    for expected in ('src="/dataset/file/401"', 'data-query-id="401"',
                     'data-search-url="/api/search"', 'data-feedback-url="/api/feedback"'):
        assert expected in page

    search = client.post("/api/search", json={"query_image_id": 401, "exclude_query": True})
    assert search.status_code == 200
    assert search.json["iteration"] == 1
    assert [item["image_id"] for item in search.json["results"]] == [20, 9525]
    feedback = client.post("/api/feedback", json={
        "session_id": search.json["session_id"],
        "relevant_ids": [9525],
    })
    assert feedback.status_code == 200
    assert feedback.json["session_id"] == search.json["session_id"]
    assert feedback.json["iteration"] == 2
    assert [item["image_id"] for item in feedback.json["results"]] == [9525, 20]
    result = feedback.json["results"][0]
    assert result["category"] == "yacht"
    with client.get(f"/dataset/file/{result['image_id']}") as thumbnail:
        assert thumbnail.status_code == 200
        assert thumbnail.data == (dataset / "yacht/9525.jpg").read_bytes()


def test_search_page_remains_accessible_without_descriptors(storage):
    make_mapping(storage, [(401, "ships")])
    main([])
    app = create_app()
    client = app.test_client()
    login(client, app)
    response = client.get("/recherche/401")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Le moteur de recherche est indisponible" in page
    assert 'src="/dataset/file/401"' in page
    assert "similarity.js" not in page
    assert client.post("/api/search", json={"query_image_id": 401}).status_code == 404
