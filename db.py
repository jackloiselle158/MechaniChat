import chromadb
import ollama
import time
from tqdm import tqdm

from backend import settings
from backend.document_processing import create_document_metadata
from backend.document_processing import get_stable_document_id, normalize_document


def embed_documents(documents):
    """Embed a batch, splitting it when the local Ollama runner is overloaded."""

    try:
        response = ollama.embed(
            model=settings.migration_embedding_model,
            input=documents,
        )
        return response["embeddings"]
    except Exception:
        if len(documents) > 1:
            midpoint = len(documents) // 2
            time.sleep(2)
            return embed_documents(documents[:midpoint]) + embed_documents(
                documents[midpoint:]
            )

        # A runner can take a moment to reload after a failed larger batch.
        for attempt in range(3):
            time.sleep(2 ** (attempt + 1))
            try:
                response = ollama.embed(
                    model=settings.migration_embedding_model,
                    input=documents,
                )
                return response["embeddings"]
            except Exception:
                if attempt == 2:
                    raise


def migrate_database():
    client = chromadb.PersistentClient(path=settings.database_path)
    source_collection = client.get_collection(name=settings.source_collection_name)
    collection = client.get_or_create_collection(
        name=settings.migration_collection_name,
        metadata={
            "embedding_model": settings.migration_embedding_model,
            "schema_version": 2,
        },
    )

    document_count = source_collection.count()
    progress = tqdm(total=document_count)
    for offset in range(0, document_count, settings.embedding_batch_size):
        source_batch = source_collection.get(
            limit=settings.embedding_batch_size,
            offset=offset,
            include=["documents"],
        )
        prepared_documents = {}
        for source_id, document in zip(
            source_batch["ids"],
            source_batch["documents"],
        ):
            if not document:
                continue
            normalized_document = normalize_document(document)
            document_id = get_stable_document_id(normalized_document)
            prepared_documents[document_id] = {
                "document": normalized_document,
                "metadata": create_document_metadata(normalized_document, source_id),
            }

        stable_ids = list(prepared_documents)
        existing_ids = set(
            collection.get(ids=stable_ids, include=[])["ids"]
            if stable_ids
            else []
        )
        existing = [document_id for document_id in stable_ids if document_id in existing_ids]
        missing = [document_id for document_id in stable_ids if document_id not in existing_ids]

        if existing:
            collection.update(
                ids=existing,
                metadatas=[prepared_documents[document_id]["metadata"] for document_id in existing],
            )
        if missing:
            documents = [prepared_documents[document_id]["document"] for document_id in missing]
            collection.upsert(
                ids=missing,
                embeddings=embed_documents(documents),
                documents=documents,
                metadatas=[prepared_documents[document_id]["metadata"] for document_id in missing],
            )

        progress.update(len(source_batch["ids"]))

    progress.close()
    settings.migration_marker.write_text(
        settings.migration_embedding_model,
        encoding="utf-8",
    )
    print(
        f"Prepared {collection.count()} unique documents with "
        f"{settings.migration_embedding_model}. Restart the API to use them."
    )


if __name__ == "__main__":
    migrate_database()
