from types import SimpleNamespace

from src.db.adapters.video_adapter import VideoAdapterMixin


class FakeTableQuery:
    def __init__(self, name: str, recorder) -> None:
        self.name = name
        self.recorder = recorder
        self._payload = None
        self._eq = {}

    def update(self, payload):
        self._payload = dict(payload)
        return self

    def eq(self, col: str, value: str):
        self._eq[col] = value
        return self

    def execute(self):
        assert self.name == "videos"
        self.recorder.append((dict(self._payload or {}), dict(self._eq)))
        video_id = self._eq.get("id")
        return SimpleNamespace(data=[{"id": video_id, **(self._payload or {})}])


class FakeClient:
    def __init__(self, recorder) -> None:
        self.recorder = recorder

    def table(self, name: str):
        return FakeTableQuery(name, self.recorder)


class FakeAdapter(VideoAdapterMixin):
    def __init__(self, client) -> None:
        self.client = client


def test_update_video_status_clears_error_message_on_non_failed():
    updates = []
    adapter = FakeAdapter(FakeClient(updates))

    adapter.update_video_status("v1", "PROCESSING")
    payload, where = updates[-1]
    assert where == {"id": "v1"}
    assert payload["status"] == "PROCESSING"
    assert payload["error_message"] is None


def test_update_video_status_sets_error_message_on_failed_when_provided():
    updates = []
    adapter = FakeAdapter(FakeClient(updates))

    adapter.update_video_status("v1", "FAILED", error="boom")
    payload, _where = updates[-1]
    assert payload["status"] == "FAILED"
    assert payload["error_message"] == "boom"


def test_update_video_status_does_not_overwrite_error_message_on_failed_without_error():
    updates = []
    adapter = FakeAdapter(FakeClient(updates))

    adapter.update_video_status("v1", "FAILED", error=None)
    payload, _where = updates[-1]
    assert payload["status"] == "FAILED"
    assert "error_message" not in payload

