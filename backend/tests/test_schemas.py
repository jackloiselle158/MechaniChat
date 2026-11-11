import pytest
from pydantic import ValidationError

from backend.schemas import ChatStreamRequest


def make_request(**changes):
    request = {
        "request_id": "request-1",
        "conversation_id": "conversation-1",
        "message": "Why are my brakes squealing?",
        "history": [],
        "attachments": [],
    }
    request.update(changes)
    return request


def test_request_accepts_and_serializes_snake_case():
    request = ChatStreamRequest.model_validate(
        make_request(
            vehicle={
                "input_mode": "details",
                "year": "2011",
                "make": "GMC",
                "model": "Terrain",
            }
        )
    )

    body = request.model_dump()
    assert body["request_id"] == "request-1"
    assert body["vehicle"]["input_mode"] == "details"


def test_request_rejects_old_camel_case_fields():
    with pytest.raises(ValidationError):
        ChatStreamRequest.model_validate(
            {
                "conversationId": "conversation-1",
                "message": "Why are my brakes squealing?",
            }
        )


def test_request_rejects_more_than_five_images():
    attachments = [
        {
            "id": str(index),
            "name": f"photo-{index}.png",
            "type": "image/png",
            "size": 100,
        }
        for index in range(6)
    ]

    with pytest.raises(ValidationError):
        ChatStreamRequest.model_validate(make_request(attachments=attachments))


def test_request_rejects_incomplete_vehicle_details():
    with pytest.raises(ValidationError):
        ChatStreamRequest.model_validate(
            make_request(
                vehicle={
                    "input_mode": "details",
                    "year": "2011",
                    "make": "GMC",
                }
            )
        )
