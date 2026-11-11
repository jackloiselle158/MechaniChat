import re
from dataclasses import dataclass

import chromadb
import ollama

from backend import settings


client = chromadb.PersistentClient(path=settings.database_path)
collection = client.get_or_create_collection(name=settings.collection_name)


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    document: str
    distance: float
    metadata: dict
    rerank_score: float


stop_words = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "how",
    "i", "in", "is", "it", "my", "of", "on", "or", "the", "this", "to",
    "what", "when", "why", "with",
}


def get_search_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    }


def get_vehicle_filter(vehicle: dict | None) -> dict | None:
    if not vehicle or not all(vehicle.get(field) for field in ("year", "make", "model")):
        return None
    return {
        "$and": [
            {"year": str(vehicle["year"])},
            {"make": str(vehicle["make"]).upper()},
            {"model": str(vehicle["model"]).upper()},
        ]
    }


def calculate_rerank_score(
    query: str,
    document: str,
    distance: float,
    metadata: dict,
    vehicle: dict | None,
) -> float:
    query_tokens = get_search_tokens(query)
    document_tokens = get_search_tokens(document)
    lexical_score = len(query_tokens & document_tokens) / max(1, len(query_tokens))
    vector_score = 1 / (1 + max(distance, 0))
    vehicle_score = 0
    if vehicle:
        matches = [
            str(metadata.get(field, "")).lower() == str(vehicle.get(field, "")).lower()
            for field in ("year", "make", "model")
            if vehicle.get(field)
        ]
        vehicle_score = sum(matches) / max(1, len(matches))
    return vector_score * 0.65 + lexical_score * 0.25 + vehicle_score * 0.10


def query_collection(query_embedding: list, vehicle: dict | None) -> dict:
    query_arguments = {
        "query_embeddings": query_embedding,
        "n_results": settings.retrieval_candidate_count,
        "include": ["documents", "distances", "metadatas"],
    }
    vehicle_filter = get_vehicle_filter(vehicle)
    if vehicle_filter:
        try:
            filtered_results = collection.query(**query_arguments, where=vehicle_filter)
            if filtered_results.get("documents") and filtered_results["documents"][0]:
                return filtered_results
        except Exception:
            # Legacy records do not have vehicle metadata. The unfiltered query
            # keeps retrieval working until the versioned migration completes.
            pass
    return collection.query(**query_arguments)


def get_relevant_context(
    query: str,
    vehicle: dict | None = None,
) -> list[RetrievedDocument]:
    embedding_input = (
        f"search_query: {query}"
        if "nomic-embed-text" in settings.embedding_model
        else query
    )
    response = ollama.embed(
        model=settings.embedding_model,
        input=embedding_input,
    )
    results = query_collection(response["embeddings"], vehicle)
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    candidates = []
    ids = results.get("ids", [[]])[0]
    documents = results["documents"][0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for document_id, document, distance, metadata in zip(
        ids,
        documents,
        distances,
        metadatas,
    ):
        if not document or distance > settings.retrieval_distance_threshold:
            continue
        current_metadata = metadata or {}
        candidates.append(
            RetrievedDocument(
                id=document_id,
                document=document,
                distance=distance,
                metadata=current_metadata,
                rerank_score=calculate_rerank_score(
                    query,
                    document,
                    distance,
                    current_metadata,
                    vehicle,
                ),
            )
        )

    candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
    return candidates[:settings.retrieval_result_count]
