import io
import os
import zipfile

from tests.conftest import make_image


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"] == "fake.pt"


def test_swagger_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/api/v1/images" in client.get("/openapi.json").json()["paths"]


def test_auth_required(client, png_bytes):
    files = {"file": ("a.png", png_bytes, "image/png")}
    assert client.post("/api/v1/images", files=files).status_code == 401
    assert client.post("/api/v1/images", files=files, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/api/v1/images").status_code == 401


def test_second_api_key_accepted(client):
    assert client.get("/api/v1/images", headers={"X-API-Key": "other-key"}).status_code == 200


def test_upload_saves_files_and_returns_detections(client, uploaded):
    assert uploaded["detections"][0]["class_name"] == "person"
    assert uploaded["width"] == 64 and uploaded["height"] == 48
    root = os.environ["STORAGE_DIR"]
    for key in ("input_file", "processed_file", "result_file"):
        assert os.path.isfile(os.path.join(root, uploaded[key])), key


def test_upload_rejects_non_image(client, auth):
    r = client.post("/api/v1/images", headers=auth, files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_upload_rejects_empty(client, auth):
    r = client.post("/api/v1/images", headers=auth, files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_upload_rejects_disallowed_format(client, auth):
    r = client.post("/api/v1/images", headers=auth, files={"file": ("x.png", make_image("GIF"), "image/png")})
    assert r.status_code == 415


def test_upload_rejects_oversized(client, auth):
    big = b"\x89PNG" + os.urandom(1024 * 1024 + 10)
    r = client.post("/api/v1/images", headers=auth, files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


def test_filename_cannot_escape_storage(client, auth, png_bytes):
    r = client.post(
        "/api/v1/images", headers=auth, files={"file": ("../../evil.png", png_bytes, "image/png")}
    )
    assert r.status_code == 201
    assert r.json()["original_filename"] == "evil.png"
    assert r.json()["input_file"].startswith("inputs/")


def test_get_list_download(client, auth, uploaded):
    iid = uploaded["id"]
    assert client.get(f"/api/v1/images/{iid}", headers=auth).json()["id"] == iid
    listing = client.get("/api/v1/images?limit=5", headers=auth).json()
    assert listing["total"] >= 1 and any(i["id"] == iid for i in listing["items"])
    assert client.get(f"/api/v1/images/{iid}/original", headers=auth).content.startswith(b"\x89PNG")
    proc = client.get(f"/api/v1/images/{iid}/processed", headers=auth)
    assert proc.headers["content-type"] == "image/jpeg"


def test_detect_returns_objects_without_saving(client, auth, png_bytes):
    before = client.get("/api/v1/images", headers=auth).json()["total"]
    r = client.post(
        "/api/v1/detect",
        headers=auth,
        files={"file": ("a.png", png_bytes, "image/png")},
        data={"include_image": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1 and body["summary"] == {"person": 1}
    assert body["annotated_image_base64"]
    assert client.get("/api/v1/images", headers=auth).json()["total"] == before


def test_detect_class_filter_and_auth(client, auth, png_bytes):
    files = {"file": ("a.png", png_bytes, "image/png")}
    r = client.post("/api/v1/detect", headers=auth, files=files, data={"classes": "car"})
    assert r.json()["count"] == 0 and r.json()["annotated_image_base64"] is None
    quoted = client.post("/api/v1/detect", headers=auth, files=files, data={"classes": "'person'"})
    assert quoted.json()["count"] == 1
    assert client.post("/api/v1/detect", files=files).status_code == 401
    bad = client.post("/api/v1/detect", headers=auth, files={"file": ("x.png", b"junk", "image/png")})
    assert bad.status_code == 400


def test_search_by_object(client, auth, uploaded):
    hit = client.get("/api/v1/images?object=Person&min_confidence=0.5", headers=auth).json()
    assert any(i["id"] == uploaded["id"] for i in hit["items"])
    assert hit["total"] == len(hit["items"])
    assert client.get("/api/v1/images?object=car", headers=auth).json()["total"] == 0
    miss = client.get("/api/v1/images?object=person&min_confidence=0.99", headers=auth).json()
    assert miss["total"] == 0


def test_web_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Find objects in images" in r.text


def test_invalid_and_unknown_ids(client, auth):
    assert client.get("/api/v1/images/..%2f..%2fetc", headers=auth).status_code in (404, 422)
    assert client.get("/api/v1/images/not-hex", headers=auth).status_code == 422
    assert client.get("/api/v1/images/" + "0" * 32, headers=auth).status_code == 404


def test_delete(client, auth, uploaded):
    iid = uploaded["id"]
    root = os.environ["STORAGE_DIR"]
    assert client.delete(f"/api/v1/images/{iid}", headers=auth).status_code == 204
    assert client.get(f"/api/v1/images/{iid}", headers=auth).status_code == 404
    assert not os.path.exists(os.path.join(root, uploaded["input_file"]))
    assert client.delete(f"/api/v1/images/{iid}", headers=auth).status_code == 404


def test_batch_reports_per_file_errors(client, auth, png_bytes):
    files = [
        ("files", ("ok.png", png_bytes, "image/png")),
        ("files", ("bad.png", b"junk", "image/png")),
    ]
    body = client.post("/api/v1/images/batch", headers=auth, files=files).json()
    assert len(body["results"]) == 1
    assert body["errors"][0]["filename"] == "bad.png"


def test_azure_sync_disabled(client, auth):
    body = client.post("/api/v1/sync/azure", headers=auth).json()
    assert body == {"uploaded": 0, "failed": 0, "azure_enabled": False}


def test_dataset_export_yolo_zip(client, auth, uploaded):
    r = client.get("/api/v1/dataset/export", headers=auth, params={"ids": uploaded["id"]})
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert {"obj.names", "obj.data", "train.txt"} <= set(names)
    label = zf.read(f"obj_train_data/{uploaded['id']}.txt").decode().split()
    assert label[0] == "0"
    assert [round(float(v), 2) for v in label[1:]] == [0.25, 0.25, 0.5, 0.5]
    assert zf.read("obj.names").decode().split() == ["person", "car"]
