import threading
import time
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from backend import settings


class ImageValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StoredImage:
    id: str
    name: str
    type: str
    size: int
    path: Path
    created_at: float


class TemporaryImageStore:
    allowed_formats = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }

    def __init__(
        self,
        directory: Path = settings.temporary_image_directory,
        max_size: int = settings.max_image_size,
        max_pixels: int = settings.max_image_pixels,
        max_dimension: int = settings.max_image_dimension,
        ttl_seconds: int = settings.temporary_image_ttl_seconds,
    ):
        self.directory = Path(directory)
        self.max_size = max_size
        self.max_pixels = max_pixels
        self.max_dimension = max_dimension
        self.ttl_seconds = ttl_seconds
        self.records = {}
        self.lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, content_type: str, data: bytes) -> StoredImage:
        self.cleanup_stale()
        expected_format = self.allowed_formats.get(content_type)
        if expected_format is None:
            raise ImageValidationError("Only JPEG, PNG, and WebP images are supported.")
        if not data:
            raise ImageValidationError("The uploaded image is empty.")
        if len(data) > self.max_size:
            raise ImageValidationError("Each image must be no larger than 8 MB.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    source.load()
                    if source.format != expected_format:
                        raise ImageValidationError(
                            "The image contents do not match its file type."
                        )
                    if source.width * source.height > self.max_pixels:
                        raise ImageValidationError("The image dimensions are too large.")
                    cleaned = ImageOps.exif_transpose(source)
                    cleaned.thumbnail(
                        (self.max_dimension, self.max_dimension),
                        Image.Resampling.LANCZOS,
                    )
                    cleaned = self._flatten_to_rgb(cleaned)
        except ImageValidationError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ImageValidationError("The image dimensions are too large.")
        except (OSError, UnidentifiedImageError):
            raise ImageValidationError("The uploaded file is not a valid image.")

        attachment_id = str(uuid4())
        path = self.directory / f"{attachment_id}.jpg"
        cleaned.save(path, format="JPEG", quality=88, optimize=True)
        record = StoredImage(
            id=attachment_id,
            name=Path(name or "image.jpg").name[:255],
            type="image/jpeg",
            size=path.stat().st_size,
            path=path,
            created_at=time.time(),
        )
        with self.lock:
            self.records[attachment_id] = record
        return record

    def get_paths(self, attachment_ids: list[str]) -> list[Path]:
        paths = []
        with self.lock:
            for attachment_id in attachment_ids:
                record = self.records.get(attachment_id)
                if record is None or not record.path.exists():
                    raise KeyError(attachment_id)
                paths.append(record.path)
        return paths

    def delete(self, attachment_ids: list[str]):
        with self.lock:
            records = [self.records.pop(attachment_id, None) for attachment_id in attachment_ids]
        for record in records:
            if record:
                record.path.unlink(missing_ok=True)

    def clear(self):
        with self.lock:
            attachment_ids = list(self.records)
        self.delete(attachment_ids)

    def cleanup_stale(self):
        cutoff = time.time() - self.ttl_seconds
        with self.lock:
            expired_ids = [
                attachment_id
                for attachment_id, record in self.records.items()
                if record.created_at < cutoff
            ]
        self.delete(expired_ids)

        for path in self.directory.glob("*.jpg"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

    @staticmethod
    def _flatten_to_rgb(image: Image.Image) -> Image.Image:
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")


image_store = TemporaryImageStore()
