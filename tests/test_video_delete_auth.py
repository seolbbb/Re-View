from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import src.process_api as process_api


class FakeAdapter:
    def __init__(self) -> None:
        self.s3_client = object()
        self.r2_only = True
        self.videos = {}
        self.preprocessing_jobs = {}
        self.processing_jobs = {}
        self.deleted_prefixes = []
        self.deleted_videos = []

    def get_video(self, video_id: str):
        return self.videos.get(video_id)

    def delete_video(self, video_id: str, user_id: str) -> bool:
        self.deleted_videos.append((video_id, user_id))
        return True

    def r2_delete_prefix(self, prefix: str):
        self.deleted_prefixes.append(prefix)
        return {"prefix": prefix, "total": 1, "deleted": 1, "errors": []}

    def get_preprocessing_job(self, job_id: str):
        return self.preprocessing_jobs.get(job_id)

    def get_processing_job(self, job_id: str):
        return self.processing_jobs.get(job_id)


@pytest.fixture
def client(monkeypatch):
    adapter = FakeAdapter()

    monkeypatch.setattr(process_api, "get_supabase_adapter", lambda: adapter)

    def fake_get_user_id(_adapter, request):
        auth = request.headers.get("Authorization") or request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        return None

    monkeypatch.setattr(process_api, "_get_user_id_from_request", fake_get_user_id)
    return TestClient(process_api.app), adapter


def test_delete_video_requires_owner(client):
    http, adapter = client
    adapter.videos["v1"] = {"id": "v1", "user_id": "owner"}

    res = http.delete("/api/videos/v1", headers={"Authorization": "Bearer other"})
    assert res.status_code == 403
    assert adapter.deleted_videos == []


def test_delete_video_owner_ok(client):
    http, adapter = client
    adapter.videos["v1"] = {"id": "v1", "user_id": "owner"}

    res = http.delete("/api/videos/v1", headers={"Authorization": "Bearer owner"})
    assert res.status_code == 204
    assert adapter.deleted_prefixes == ["v1/"]
    assert adapter.deleted_videos == [("v1", "owner")]


def test_delete_video_blocks_while_job_active(client):
    http, adapter = client
    adapter.videos["v1"] = {
        "id": "v1",
        "user_id": "owner",
        "current_processing_job_id": "job1",
    }
    adapter.processing_jobs["job1"] = {
        "id": "job1",
        "status": "VLM_RUNNING",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    res = http.delete("/api/videos/v1", headers={"Authorization": "Bearer owner"})
    assert res.status_code == 409
    assert adapter.deleted_videos == []


def test_delete_video_allows_stuck_job(client):
    http, adapter = client
    adapter.videos["v1"] = {
        "id": "v1",
        "user_id": "owner",
        "current_processing_job_id": "job1",
    }
    adapter.processing_jobs["job1"] = {
        "id": "job1",
        "status": "VLM_RUNNING",
        "updated_at": "2000-01-01T00:00:00+00:00",
    }

    res = http.delete("/api/videos/v1", headers={"Authorization": "Bearer owner"})
    assert res.status_code == 204
    assert adapter.deleted_videos == [("v1", "owner")]

