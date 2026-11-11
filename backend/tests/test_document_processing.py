from backend.document_processing import create_document_metadata
from backend.document_processing import get_stable_document_id


def test_stable_document_ids_ignore_spacing_and_case():
    first = "The contact owns a 2010 Toyota Camry. The brakes failed."
    duplicate = "  THE contact owns a 2010 Toyota Camry.   The brakes failed.  "

    assert get_stable_document_id(first) == get_stable_document_id(duplicate)


def test_document_metadata_extracts_vehicle_and_component():
    metadata = create_document_metadata(
        "The contact owns a 2010 Toyota Camry. The brake pedal went to the floor.",
        "42",
    )

    assert metadata == {
        "source": "nhtsa_complaint",
        "source_id": "42",
        "schema_version": 2,
        "year": "2010",
        "make": "TOYOTA",
        "model": "CAMRY",
        "component": "brakes",
    }
