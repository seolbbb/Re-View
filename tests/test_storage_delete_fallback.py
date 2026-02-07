from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.process_api as process_api


class FakeStorageBucket:
    def __init__(self, bucket: str, recorder, *, should_fail: bool = False) -> None:
        self.bucket = bucket
        self.recorder = recorder
        self.should_fail = should_fail

    def remove(self, paths):
        if self.should_fail:
            raise RuntimeError(f"remove failed for bucket={self.bucket}")
        self.recorder.append((self.bucket, list(paths)))
        return {"data": [{"name": p} for p in paths]}


class FakeStorage:
    def __init__(self, recorder, *, fail_buckets=None) -> None:
        self.recorder = recorder
        self.fail_buckets = set(fail_buckets or [])

    def from_(self, bucket: str):
        return FakeStorageBucket(bucket, self.recorder, should_fail=bucket in self.fail_buckets)


class FakeTableQuery:
    def __init__(self, table: str, data_by_table: dict, *, video_id: str) -> None:
        self.table = table
        self.data_by_table = data_by_table
        self.video_id = video_id
        self._eq = {}

    def select(self, _fields: str):
        return self

    def eq(self, col: str, value: str):
        self._eq[col] = value
        return self

    def execute(self):
        if self._eq.get("video_id") != self.video_id:
            return SimpleNamespace(data=[])
        return SimpleNamespace(data=self.data_by_table.get(self.table, []))


class FakeClient:
    def __init__(self, recorder, data_by_table: dict, *, video_id: str, fail_buckets=None) -> None:
        self.storage = FakeStorage(recorder, fail_buckets=fail_buckets)
        self._data_by_table = data_by_table
        self._video_id = video_id

    def table(self, name: str):
        return FakeTableQuery(name, self._data_by_table, video_id=self._video_id)


class FakeAdapter:
    def __init__(self, video_id: str, *, fail_buckets=None, r2_only: bool = False) -> None:
        self.s3_client = None
        self.r2_only = r2_only
        self.deleted_videos = []
        self.removed = []
        self.videos = {
            video_id: {
                "id": video_id,
                "user_id": "owner",
                "video_storage_key": f"{video_id}/video.mp4",
            }
        }

        data_by_table = {
            "captures": [
                {"storage_path": f"{video_id}/cap_001.jpg"},
                {"storage_path": "other/ignore.jpg"},
            ],
            "preprocessing_jobs": [{"audio_storage_key": f"{video_id}/audio.wav"}],
        }
        self.client = FakeClient(self.removed, data_by_table, video_id=video_id, fail_buckets=fail_buckets)

    def get_video(self, video_id: str):
        return self.videos.get(video_id)

    def delete_video(self, video_id: str, user_id: str) -> bool:
        self.deleted_videos.append((video_id, user_id))
        return True

    # Methods referenced by _is_video_actively_processing (not used in these tests).
    def get_preprocessing_job(self, _job_id: str):
        return None

    def get_processing_job(self, _job_id: str):
        return None


@pytest.fixture
def client_and_adapter(monkeypatch):
    video_id = "v1"
    adapter = FakeAdapter(video_id)

    monkeypatch.setattr(process_api, "get_supabase_adapter", lambda: adapter)

    def fake_get_user_id(_adapter, request):
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        return None

    monkeypatch.setattr(process_api, "_get_user_id_from_request", fake_get_user_id)
    return TestClient(process_api.app), adapter, video_id


def test_delete_video_supabase_storage_fallback_ok(client_and_adapter):
    http, adapter, video_id = client_and_adapter
    res = http.delete(f"/api/videos/{video_id}", headers={"Authorization": "Bearer owner"})

    assert res.status_code == 204
    assert adapter.deleted_videos == [(video_id, "owner")]

    # remove() should be called per bucket with known paths.
    assert ("videos", [f"{video_id}/video.mp4"]) in adapter.removed
    assert ("captures", [f"{video_id}/cap_001.jpg"]) in adapter.removed
    assert ("audio", [f"{video_id}/audio.wav"]) in adapter.removed


def test_delete_video_supabase_storage_fallback_failure_blocks_db_delete(monkeypatch):
    video_id = "v1"
    adapter = FakeAdapter(video_id, fail_buckets={"captures"})

    monkeypatch.setattr(process_api, "get_supabase_adapter", lambda: adapter)
    monkeypatch.setattr(process_api, "_get_user_id_from_request", lambda _adapter, _request: "owner")

    http = TestClient(process_api.app)
    res = http.delete(f"/api/videos/{video_id}", headers={"Authorization": "Bearer owner"})

    assert res.status_code == 502
    assert adapter.deleted_videos == []


def test_delete_video_requires_r2_when_flagged(monkeypatch):
    video_id = "v1"
    adapter = FakeAdapter(video_id, r2_only=True)

    monkeypatch.setattr(process_api, "get_supabase_adapter", lambda: adapter)
    monkeypatch.setattr(process_api, "_get_user_id_from_request", lambda _adapter, _request: "owner")

    http = TestClient(process_api.app)
    res = http.delete(f"/api/videos/{video_id}", headers={"Authorization": "Bearer owner"})

    assert res.status_code == 503
    assert adapter.deleted_videos == []

