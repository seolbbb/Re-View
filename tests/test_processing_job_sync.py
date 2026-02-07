from types import SimpleNamespace

from src.db.adapters.job_adapter import JobAdapterMixin


class FakeTableQuery:
    def __init__(self, name: str, client) -> None:
        self.name = name
        self.client = client
        self._op = None
        self._payload = None
        self._eq = {}

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def select(self, _fields: str):
        self._op = "select"
        return self

    def eq(self, col: str, value: str):
        self._eq[col] = value
        return self

    def execute(self):
        if self.name != "processing_jobs":
            raise AssertionError(f"unexpected table: {self.name}")

        job_id = self._eq.get("id")
        video_id = self.client.video_id_by_job_id.get(job_id)

        if self._op == "update":
            self.client.processing_job_updates.append((job_id, dict(self._payload or {})))
            row = {"id": job_id, "video_id": video_id}
            return SimpleNamespace(data=[row])

        if self._op == "select":
            if not video_id:
                return SimpleNamespace(data=[])
            return SimpleNamespace(data=[{"video_id": video_id}])

        raise AssertionError(f"unexpected op: {self._op}")


class FakeClient:
    def __init__(self) -> None:
        self.video_id_by_job_id = {"job1": "v1"}
        self.processing_job_updates = []

    def table(self, name: str):
        return FakeTableQuery(name, self)


class FakeAdapter(JobAdapterMixin):
    def __init__(self, client) -> None:
        self.client = client
        self.video_status_updates = []

    def update_video_status(self, video_id: str, status: str, error=None):
        self.video_status_updates.append((video_id, status, error))
        return {"id": video_id, "status": status}


def test_processing_job_done_syncs_video_status():
    client = FakeClient()
    adapter = FakeAdapter(client)

    adapter.update_processing_job_status("job1", "DONE")
    assert adapter.video_status_updates == [("v1", "DONE", None)]


def test_processing_job_failed_syncs_video_status():
    client = FakeClient()
    adapter = FakeAdapter(client)

    adapter.update_processing_job_status("job1", "FAILED", error_message="boom")
    assert adapter.video_status_updates == [("v1", "FAILED", "boom")]


def test_processing_job_non_terminal_does_not_sync_video_status():
    client = FakeClient()
    adapter = FakeAdapter(client)

    adapter.update_processing_job_status("job1", "VLM_RUNNING")
    assert adapter.video_status_updates == []

