from io import BytesIO

import pytest
from PIL import Image

from backend.image_store import ImageValidationError, TemporaryImageStore


def create_image(image_format="PNG", size=(40, 30)):
    image_buffer = BytesIO()
    Image.new("RGBA", size, (20, 80, 40, 128)).save(
        image_buffer,
        format=image_format,
    )
    return image_buffer.getvalue()


def test_store_normalizes_and_deletes_image(tmp_path):
    store = TemporaryImageStore(tmp_path, max_dimension=24)

    record = store.save("dashboard.png", "image/png", create_image())

    assert record.type == "image/jpeg"
    assert record.path.exists()
    with Image.open(record.path) as cleaned:
        assert cleaned.format == "JPEG"
        assert max(cleaned.size) <= 24
        assert cleaned.mode == "RGB"

    assert store.get_paths([record.id]) == [record.path]
    store.delete([record.id])
    assert not record.path.exists()


def test_store_rejects_file_type_mismatch(tmp_path):
    store = TemporaryImageStore(tmp_path)

    with pytest.raises(ImageValidationError, match="do not match"):
        store.save("dashboard.jpg", "image/jpeg", create_image())
