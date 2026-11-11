import pytest
from PIL import Image

from backend import chat_service


def test_keyword_route_requires_an_explicit_tool_request():
    assert chat_service.get_keyword_route("Look up recalls for my car") == "get_recalls"
    assert chat_service.get_keyword_route("What does recall mean?") == "answer"
    assert chat_service.get_keyword_route("I can't recall when it started") == "answer"
    assert chat_service.get_keyword_route("Why are my brakes squealing?") == "answer"
    assert (
        chat_service.get_keyword_route("What is my NHTSA crash rating?")
        == "get_crash_ratings"
    )


@pytest.mark.asyncio
async def test_route_falls_back_safely_when_small_model_fails(monkeypatch):
    class FailingRouter:
        async def ainvoke(self, _):
            raise RuntimeError("router unavailable")

    monkeypatch.setattr(chat_service, "router", FailingRouter())

    assert await chat_service.get_route("Why is my engine shaking?") == "answer"
    assert await chat_service.get_route("Are there recalls for my car?") == "get_recalls"


@pytest.mark.asyncio
async def test_saved_vehicle_supplies_tool_arguments_without_extraction(monkeypatch):
    class FailingExtractor:
        async def ainvoke(self, _):
            raise AssertionError("saved vehicles should not be extracted by an LLM")

    monkeypatch.setattr(
        chat_service,
        "vehicle_argument_extractor",
        FailingExtractor(),
    )
    vehicle = {
        "input_mode": "details",
        "year": "2013",
        "make": "Dodge",
        "model": "Challenger",
    }

    arguments = await chat_service.get_route_arguments(
        "get_recalls",
        "What recalls apply?",
        [],
        vehicle,
    )

    assert arguments == {
        "year": "2013",
        "make": "Dodge",
        "model": "Challenger",
    }


@pytest.mark.asyncio
async def test_follow_up_retrieval_query_uses_history_and_vehicle(monkeypatch):
    class FakeRewriter:
        async def ainvoke(self, values):
            assert "Why is it doing that?" in values["conversation"]
            assert values["vehicle"] == "2011 GMC Terrain"
            return chat_service.RetrievalQuery(
                query="2011 GMC Terrain transmission hesitation"
            )

    monkeypatch.setattr(chat_service, "retrieval_query_rewriter", FakeRewriter())

    query = await chat_service.rewrite_retrieval_query(
        "Why is it doing that?",
        [{"role": "user", "content": "The transmission hesitates."}],
        {"year": "2011", "make": "GMC", "model": "Terrain"},
    )

    assert query == "2011 GMC Terrain transmission hesitation"


def test_image_message_contains_base64_image_context(tmp_path):
    image_path = tmp_path / "cleaned.jpg"
    Image.new("RGB", (10, 10), "red").save(image_path, format="JPEG")

    message = chat_service.create_user_message(
        "What warning light is this?",
        [image_path],
    )

    assert message.content[0]["type"] == "image_url"
    assert message.content[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert message.content[1] == {
        "type": "text",
        "text": "What warning light is this?",
    }
