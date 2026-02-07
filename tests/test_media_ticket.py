import src.process_api as process_api


def test_media_ticket_roundtrip(monkeypatch):
    monkeypatch.setenv("MEDIA_TICKET_SECRET", "test-secret")
    monkeypatch.setattr(process_api.time, "time", lambda: 1_000)

    ticket = process_api._create_media_ticket("user-123", ttl_sec=60)
    assert process_api._verify_media_ticket(ticket) == "user-123"


def test_media_ticket_expired(monkeypatch):
    monkeypatch.setenv("MEDIA_TICKET_SECRET", "test-secret")
    monkeypatch.setattr(process_api.time, "time", lambda: 1_000)

    ticket = process_api._create_media_ticket("user-123", ttl_sec=10)

    monkeypatch.setattr(process_api.time, "time", lambda: 1_011)
    assert process_api._verify_media_ticket(ticket) is None


def test_media_ticket_tamper_rejected(monkeypatch):
    monkeypatch.setenv("MEDIA_TICKET_SECRET", "test-secret")
    monkeypatch.setattr(process_api.time, "time", lambda: 1_000)

    ticket = process_api._create_media_ticket("user-123", ttl_sec=60)
    payload_b64, sig_b64 = ticket.split(".", 1)

    # Flip a bit in the signature.
    tampered_sig = sig_b64[:-1] + ("A" if sig_b64[-1] != "A" else "B")
    tampered = f"{payload_b64}.{tampered_sig}"
    assert process_api._verify_media_ticket(tampered) is None

