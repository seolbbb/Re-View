from fastapi.testclient import TestClient

import src.process_api as process_api


class FakeAdapter:
    def __init__(self) -> None:
        self.videos = {
            "v1": {
                "id": "v1",
                "user_id": "owner",
                "status": "FAILED",
                "error_message": "boom",
                "current_processing_job_id": "pj1",
            }
        }
        self.update_calls = []

    def get_video(self, video_id: str):
        return self.videos.get(video_id)

    def update_video_status(self, video_id: str, status: str, error=None):
        # Simulate DB behavior expected by the API layer.
        self.update_calls.append((video_id, status, error))
        row = self.videos[video_id]
        row["status"] = status
        if (status or "").upper() == "FAILED":
            if error is not None:
                row["error_message"] = error
        else:
            row["error_message"] = None
        return row


def test_process_restart_marks_processing_and_clears_error(monkeypatch):
    adapter = FakeAdapter()

    monkeypatch.setattr(process_api, "get_supabase_adapter", lambda: adapter)

    def fake_get_user_id(_adapter, request):
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        return None

    monkeypatch.setattr(process_api, "_get_user_id_from_request", fake_get_user_id)

    pipeline_calls = []
    monkeypatch.setattr(process_api, "run_processing_pipeline", lambda **kwargs: pipeline_calls.append(kwargs))

    http = TestClient(process_api.app)
    res = http.post("/process", json={"video_id": "v1"}, headers={"Authorization": "Bearer owner"})

    assert res.status_code == 200
    assert adapter.videos["v1"]["status"] == "PROCESSING"
    assert adapter.videos["v1"]["error_message"] is None
    assert pipeline_calls and pipeline_calls[0]["video_id"] == "v1"
    assert pipeline_calls[0]["sync_to_db"] is True
    assert pipeline_calls[0]["resume"] is True
    assert pipeline_calls[0]["existing_processing_job_id"] == "pj1"


def test_process_blocks_duplicate_run(monkeypatch):
    adapter = FakeAdapter()

    monkeypatch.setattr(process_api, "get_supabase_adapter", lambda: adapter)
    monkeypatch.setattr(process_api, "_get_user_id_from_request", lambda _adapter, _request: "owner")
    monkeypatch.setattr(process_api, "run_processing_pipeline", lambda **_kwargs: None)

    http = TestClient(process_api.app)
    first = http.post("/process", json={"video_id": "v1"}, headers={"Authorization": "Bearer owner"})
    assert first.status_code == 200

    second = http.post("/process", json={"video_id": "v1"}, headers={"Authorization": "Bearer owner"})
    assert second.status_code == 409
