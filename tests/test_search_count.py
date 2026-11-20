import re

import numpy as np

from backend.app import create_app
from backend.cbir_config import N_DIMS
from scripts.import_dataset import main

from test_dataset import login, make_mapping, storage


def dashboard_search_count(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    return int(re.search(r'class="stat-value">(\d+)</div>\s*'
                         r'<div class="stat-label">Recherches</div>', html)[1])


def test_only_successful_initial_searches_are_counted_per_user(storage):
    _, mapping = storage
    make_mapping(storage, [(7, "ships"), (41, "ships")])
    vectors = np.repeat(np.array([[0], [1]], dtype=np.float32), N_DIMS, axis=1)
    np.save(mapping.with_name("descriptors_raw.npy"), vectors)
    main([])

    app = create_app()
    first = app.test_client()
    second = app.test_client()
    query = {"query_image_id": 7, "exclude_query": True}
    assert second.post("/api/search", json=query).status_code == 302
    login(first, app)
    assert dashboard_search_count(first) == 0
    assert first.post("/api/search", json={"query_image_id": "7"}).status_code == 400
    assert first.post("/api/search", json={"query_image_id": 9999}).status_code == 404
    assert dashboard_search_count(first) == 0

    search = first.post("/api/search", json=query)
    assert search.status_code == 200
    assert search.json["iteration"] == 1
    assert dashboard_search_count(first) == 1
    assert dashboard_search_count(first) == 1
    feedback = first.post("/api/feedback", json={
        "session_id": search.json["session_id"],
        "relevant_ids": [41],
    })
    assert feedback.status_code == 200
    assert feedback.json["iteration"] == 2
    assert dashboard_search_count(first) == 1

    credentials = {"username": "another-user", "password": "another-password"}
    assert second.post("/register", data={
        **credentials, "email": "another@example.com", "confirm": credentials["password"],
    }).status_code == 302
    assert second.post("/login", data=credentials).status_code == 302
    assert dashboard_search_count(second) == 0
    assert second.post("/api/search", json=query).status_code == 200
    assert second.post("/api/search", json=query).status_code == 200
    assert dashboard_search_count(second) == 2
    assert dashboard_search_count(first) == 1

    restarted = create_app()
    first = restarted.test_client()
    second = restarted.test_client()
    assert first.post("/login", data={"username": "test-user", "password": "test-password"}).status_code == 302
    assert second.post("/login", data=credentials).status_code == 302
    assert dashboard_search_count(first) == 1
    assert dashboard_search_count(second) == 2
