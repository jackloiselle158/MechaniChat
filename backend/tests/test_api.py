import asyncio
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend import api, settings
from backend.image_store import TemporaryImageStore
from backend.schemas import Citation


client = TestClient(api.app)


def test_health_check_uses_mocked_ollama(monkeypatch):
    class FakeOllamaClient:
        def list(self):
            return {
                "models": [
                    {"model": settings.chat_model},
                    {"model": settings.router_model},
                    {"model": settings.vision_model},
                ]
            }

        def show(self, _):
            return {"capabilities": ["completion", "vision", "tools"]}

    monkeypatch.setattr(
        api.ollama,
        "Client",
        lambda host: FakeOllamaClient(),
    )
    monkeypatch.setattr(api.collection, "count", lambda: 42)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["capabilities"]["vision"] is True


def test_stream_returns_sse_tokens_and_citation(monkeypatch):
    async def fake_stream_chat(
        user_input,
        history,
        vehicle,
        cancel_event,
        image_paths=None,
    ):
        yield {"type": "delta", "text": "Check the brakes. "}
        yield {
            "type": "citation",
            "citation": Citation(
                id=1,
                title="NHTSA test source",
                source_type="nhtsa",
                detail="Mocked safety data",
                url="https://www.nhtsa.gov/",
            ),
        }

    monkeypatch.setattr(api, "stream_chat", fake_stream_chat)
    response = client.post(
        "/api/v1/chat/stream",
        json={
            "request_id": "stream-test",
            "conversation_id": "conversation-1",
            "message": "Check my brakes",
            "history": [],
            "attachments": [],
        },
    )

    assert response.status_code == 200
    assert "event: message.started" in response.text
    assert "event: message.delta" in response.text
    assert "event: citation" in response.text
    assert "event: message.completed" in response.text


def test_uploaded_image_reaches_chat_and_is_cleaned_up(monkeypatch, tmp_path):
    temporary_store = TemporaryImageStore(tmp_path)
    monkeypatch.setattr(api, "image_store", temporary_store)
    observed_paths = []

    async def fake_stream_chat(
        user_input,
        history,
        vehicle,
        cancel_event,
        image_paths=None,
    ):
        observed_paths.extend(image_paths or [])
        assert image_paths and image_paths[0].exists()
        yield {"type": "delta", "text": "The tire tread is visible."}

    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "black").save(image_buffer, format="PNG")
    upload_response = client.post(
        "/api/v1/attachments",
        files={"files": ("tire.png", image_buffer.getvalue(), "image/png")},
    )
    assert upload_response.status_code == 200
    attachment = upload_response.json()["attachments"][0]
    assert attachment["type"] == "image/jpeg"

    monkeypatch.setattr(api, "stream_chat", fake_stream_chat)
    chat_response = client.post(
        "/api/v1/chat/stream",
        json={
            "request_id": "image-test",
            "conversation_id": "conversation-1",
            "message": "What do you see in this tire photo?",
            "history": [],
            "attachments": [attachment],
        },
    )

    assert chat_response.status_code == 200
    assert "The tire tread is visible." in chat_response.text
    assert observed_paths
    assert not observed_paths[0].exists()


def test_upload_rejects_non_image_content(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "image_store", TemporaryImageStore(tmp_path))

    response = client.post(
        "/api/v1/attachments",
        files={"files": ("not-an-image.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a valid image."


def test_cancellation_endpoint_sets_request_event():
    cancel_event = asyncio.Event()
    api.request_controls["cancel-test"] = {
        "cancel_event": cancel_event,
        "task": None,
        "started_at": None,
    }

    response = client.delete("/api/v1/chat/cancel-test")

    assert response.status_code == 200
    assert response.json()["status"] == "cancellation_requested"
    assert cancel_event.is_set()
    api.request_controls.pop("cancel-test", None)


def test_vin_decoder_returns_vehicle_details(monkeypatch):
    monkeypatch.setattr(
        api,
        "decode_vin",
        lambda vin: {
            "vin": vin,
            "year": "2003",
            "make": "HONDA",
            "model": "Accord",
        },
    )

    response = client.post(
        "/api/v1/vehicles/decode-vin",
        json={"vin": "1HGCM82633A004352"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "year": "2003",
        "make": "HONDA",
        "model": "Accord",
    }


def test_vin_decoder_rejects_unidentified_vehicle(monkeypatch):
    monkeypatch.setattr(api, "decode_vin", lambda _: {})

    response = client.post(
        "/api/v1/vehicles/decode-vin",
        json={"vin": "1HGCM82633A004352"},
    )

    assert response.status_code == 422
