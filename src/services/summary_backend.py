from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib import error as urllib_error
from urllib import request as urllib_request

State = Dict[str, Any]


@runtime_checkable
class SummaryBackend(Protocol):
    """Backend interface for summary chatbot data access."""

    def ensure_summary_exists(self, state: State) -> Dict[str, Any]:
        """Ensure summaries are available or started."""

    def get_summary_updates(self, state: State) -> Dict[str, Any]:
        """Fetch new summary updates and update state."""

    def get_summary_context(self, state: State, time_ms: Optional[int] = None) -> Dict[str, Any]:
        """Return summary cache and optional time-filtered matches."""

    def partial_not_implemented(self, state: State) -> Dict[str, Any]:
        """Placeholder for partial summary mode."""

    def get_evidence(
        self,
        state: State,
        *,
        stt_ids: List[str],
        cap_ids: List[str],
    ) -> Dict[str, Any]:
        """Fetch evidence records for given STT/VLM IDs."""


class ProcessApiBackend:
    def __init__(self, process_api_url: Optional[str] = None) -> None:
        base = process_api_url or os.environ.get("PROCESS_API_URL") or "http://localhost:8000"
        self._base_url = base.rstrip("/")

    def ensure_summary_exists(self, state: State) -> Dict[str, Any]:
        status_result = self._get_summary_status(state)
        if not status_result.get("success"):
            return status_result

        status = status_result.get("status")
        has_summary = status_result.get("has_summary", False)

        if has_summary or status in ("running", "summarizing", "in_progress"):
            return {
                "success": True,
                "action": "none",
                "status": status,
                "has_summary": has_summary,
                "message": "요약이 이미 존재하거나 진행 중입니다.",
            }

        if status == "not_started" or (not has_summary and status == "completed"):
            start_result = self._start_summary_job(state)
            if start_result.get("success"):
                return {
                    "success": True,
                    "action": "started",
                    "status": "running",
                    "has_summary": False,
                    "message": "요약 작업을 자동으로 시작했습니다.",
                }
            return {
                "success": False,
                "action": "failed",
                "error": start_result.get("error"),
                "message": "요약 작업 시작에 실패했습니다.",
            }

        return {
            "success": True,
            "action": "none",
            "status": status,
            "has_summary": has_summary,
            "message": f"현재 상태: {status}",
        }

    def get_summary_updates(self, state: State) -> Dict[str, Any]:
        video_id = state.get("video_id")
        if not video_id:
            if not state.get("video_name"):
                return {"success": False, "error": "video_name is not set"}
            return {"success": False, "error": "video_id is not set"}

        last_segment_id = state.get("last_segment_id", 0)
        try:
            last_segment_id = int(last_segment_id)
        except (TypeError, ValueError):
            last_segment_id = 0

        response = self._call_process_api("GET", f"/videos/{video_id}/summaries")
        if not response.get("success"):
            return {"success": False, "error": response.get("error")}

        items = response.get("payload", {}).get("items", [])
        new_updates = []
        max_segment_id = last_segment_id

        if isinstance(items, list):
            for item in items:
                try:
                    current_id = item.get("segment_index", -1)
                    if current_id > last_segment_id:
                        new_updates.append(item)
                        if current_id > max_segment_id:
                            max_segment_id = current_id
                except (TypeError, ValueError):
                    continue

        if new_updates:
            state["pending_updates"] = state.get("pending_updates", []) + new_updates
            state["last_segment_id"] = max_segment_id

        return {
            "success": True,
            "last_segment_id": max_segment_id,
            "new_count": len(new_updates),
            "source": "db",
        }

    def get_summary_context(self, state: State, time_ms: Optional[int] = None) -> Dict[str, Any]:
        summary_cache = state.get("summary_cache", [])
        if not isinstance(summary_cache, list):
            summary_cache = []
        pending_updates = state.get("pending_updates", [])
        if not isinstance(pending_updates, list):
            pending_updates = []

        if pending_updates:
            existing_ids = set()
            for record in summary_cache:
                try:
                    sid = record.get("segment_index") if "segment_index" in record else record.get("segment_id")
                    existing_ids.add(int(sid))
                except (TypeError, ValueError):
                    continue
            for record in pending_updates:
                try:
                    segment_id = int(
                        record.get("segment_index") if "segment_index" in record else record.get("segment_id")
                    )
                except (TypeError, ValueError):
                    segment_id = None
                if segment_id is not None and segment_id in existing_ids:
                    continue
                summary_cache.append(record)
                if segment_id is not None:
                    existing_ids.add(segment_id)

        state["summary_cache"] = summary_cache
        state["pending_updates"] = []

        if time_ms is None:
            return {"success": True, "summary_cache": summary_cache}

        try:
            time_ms = int(time_ms)
        except (TypeError, ValueError):
            return {"success": False, "error": "time_ms must be an integer"}
        if time_ms < 0:
            return {"success": False, "error": "time_ms must be >= 0"}

        matches = []
        max_end_ms = None
        for record in summary_cache:
            try:
                start_ms = int(record.get("start_ms"))
                end_ms = int(record.get("end_ms"))
            except (TypeError, ValueError):
                continue
            if max_end_ms is None or end_ms > max_end_ms:
                max_end_ms = end_ms
            if start_ms <= time_ms <= end_ms:
                matches.append(record)

        out_of_range = max_end_ms is not None and time_ms > max_end_ms
        return {
            "success": True,
            "summary_cache": summary_cache,
            "matches": matches,
            "out_of_range": out_of_range,
            "max_end_ms": max_end_ms,
        }

    def partial_not_implemented(self, state: State) -> Dict[str, Any]:
        return {"success": False, "error": "Partial summary chatbot is not implemented yet."}

    def get_evidence(
        self,
        state: State,
        *,
        stt_ids: List[str],
        cap_ids: List[str],
    ) -> Dict[str, Any]:
        video_id = state.get("video_id")
        if not video_id:
            return {"success": False, "error": "video_id is not set"}
        if not stt_ids and not cap_ids:
            return {"success": True, "stt": [], "vlm": []}

        params = []
        if stt_ids:
            params.append(("stt_ids", ",".join(stt_ids)))
        if cap_ids:
            params.append(("cap_ids", ",".join(cap_ids)))
        query = ""
        if params:
            from urllib.parse import urlencode

            query = f"?{urlencode(params)}"

        response = self._call_process_api("GET", f"/videos/{video_id}/evidence{query}")
        if not response.get("success"):
            return {"success": False, "error": response.get("error")}

        payload = response.get("payload", {})
        return {
            "success": True,
            "stt": payload.get("stt", []),
            "vlm": payload.get("vlm", []),
        }

    def _get_summary_status(self, state: State) -> Dict[str, Any]:
        video_id = state.get("video_id")
        if not video_id:
            return {"success": False, "error": "video_name or video_id is not set"}

        status_response = self._call_process_api("GET", f"/videos/{video_id}/status")
        if not status_response.get("success"):
            if status_response.get("status_code") == 404:
                return {"success": False, "error": "Video not found"}
            return {"success": False, "error": status_response.get("error", "status API failed")}

        status_data = status_response.get("payload", {})
        video_status = status_data.get("video_status")
        processing_job = status_data.get("processing_job", {})

        summary_response = self._call_process_api("GET", f"/videos/{video_id}/summary")
        summary_data = summary_response.get("payload", {}) if summary_response.get("success") else {}
        has_summary = summary_data.get("has_summary", False)
        summary_status = summary_data.get("status") if has_summary else None

        if processing_job:
            job_status = processing_job.get("status")
            if job_status == "VLM_RUNNING":
                normalized_status = "running"
            elif job_status == "SUMMARY_RUNNING":
                normalized_status = "summarizing"
            elif job_status == "DONE":
                normalized_status = "completed"
            elif job_status == "FAILED":
                normalized_status = "error"
            else:
                normalized_status = job_status.lower() if job_status else "unknown"
        elif has_summary and summary_status == "DONE":
            normalized_status = "completed"
        elif has_summary and summary_status == "IN_PROGRESS":
            normalized_status = "in_progress"
        elif video_status == "PREPROCESS_DONE":
            normalized_status = "not_started"
        else:
            normalized_status = video_status.lower() if video_status else "unknown"

        return {
            "success": True,
            "status": normalized_status,
            "video_status": video_status,
            "has_summary": has_summary,
            "summary_status": summary_status,
            "processing_job": processing_job,
        }

    def _start_summary_job(self, state: State) -> Dict[str, Any]:
        video_id = state.get("video_id")
        video_name = state.get("video_name")
        if not video_id and not video_name:
            return {"success": False, "error": "video_name or video_id is not set"}

        payload: Dict[str, Any] = {}
        if video_id:
            payload["video_id"] = video_id
        if video_name:
            payload["video_name"] = video_name

        response = self._call_process_api("POST", "/process", payload)
        if not response.get("success"):
            return {"success": False, "error": response.get("error", "process API failed")}

        api_payload = response.get("payload", {})
        return {
            "success": True,
            "status": api_payload.get("status", "started"),
            "message": api_payload.get("message"),
        }

    def _call_process_api(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib_request.Request(
            url,
            data=data,
            method=method.upper(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=5) as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
            return {
                "success": False,
                "status_code": exc.code,
                "error": body or exc.reason,
            }
        except urllib_error.URLError as exc:
            return {"success": False, "error": str(exc.reason)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        try:
            payload_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload_data = {"raw": body}
        return {"success": True, "status_code": status_code, "payload": payload_data}
