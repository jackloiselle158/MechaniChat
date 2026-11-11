from backend import retrieval


class FakeCollection:
    def __init__(self, results, filtered_results=None):
        self.results = results
        self.filtered_results = filtered_results
        self.calls = []

    def query(self, **arguments):
        self.calls.append(arguments)
        if "where" in arguments and self.filtered_results is not None:
            return self.filtered_results
        return self.results


def make_results(ids, documents, distances, metadatas):
    return {
        "ids": [ids],
        "documents": [documents],
        "distances": [distances],
        "metadatas": [metadatas],
    }


def test_retrieval_filters_weak_matches_and_reranks_candidates(monkeypatch):
    results = make_results(
        ["closest", "useful", "irrelevant"],
        [
            "A vehicle had a minor unrelated concern.",
            "The 2010 Toyota Camry brake pedal went to the floor.",
            "A cake recipe used butter and sugar.",
        ],
        [0.20, 0.25, 0.90],
        [
            {},
            {"year": "2010", "make": "TOYOTA", "model": "CAMRY"},
            {},
        ],
    )
    fake_collection = FakeCollection(results)
    embedding_inputs = []

    def fake_embed(model, input):
        embedding_inputs.append(input)
        return {"embeddings": [[0.1, 0.2]]}

    monkeypatch.setattr(retrieval, "collection", fake_collection)
    monkeypatch.setattr(retrieval.ollama, "embed", fake_embed)
    monkeypatch.setattr(retrieval.settings, "embedding_model", "nomic-embed-text")
    monkeypatch.setattr(retrieval.settings, "retrieval_distance_threshold", 0.50)
    monkeypatch.setattr(retrieval.settings, "retrieval_result_count", 3)

    documents = retrieval.get_relevant_context(
        "Camry brake pedal failure",
        {"year": "2010", "make": "Toyota", "model": "Camry"},
    )

    assert [document.id for document in documents] == ["useful", "closest"]
    assert "irrelevant" not in [document.id for document in documents]
    assert embedding_inputs == ["search_query: Camry brake pedal failure"]
    assert "where" in fake_collection.calls[0]


def test_empty_metadata_filter_falls_back_to_unfiltered_query(monkeypatch):
    results = make_results(
        ["one"],
        ["Transmission hesitates while shifting."],
        [0.20],
        [{}],
    )
    fake_collection = FakeCollection(
        results,
        filtered_results=make_results([], [], [], []),
    )
    monkeypatch.setattr(retrieval, "collection", fake_collection)
    monkeypatch.setattr(
        retrieval.ollama,
        "embed",
        lambda **_: {"embeddings": [[0.1, 0.2]]},
    )
    monkeypatch.setattr(retrieval.settings, "retrieval_distance_threshold", 0.50)

    documents = retrieval.get_relevant_context(
        "transmission hesitation",
        {"year": "2013", "make": "Dodge", "model": "Challenger"},
    )

    assert [document.id for document in documents] == ["one"]
    assert len(fake_collection.calls) == 2
