from backend.tools import get_recalls


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {
                    "Component": "AIR BAGS",
                    "Summary": "The air bag may not deploy as intended.",
                }
            ]
        }


def test_recalls_uses_mocked_nhtsa_response(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("backend.tools.requests.get", fake_get)
    result = get_recalls.invoke(
        {"make": "GMC", "model": "Terrain", "year": "2011"}
    )

    assert "AIR BAGS" in result
    assert calls[0]["params"]["modelYear"] == "2011"
