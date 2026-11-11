from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatHistoryMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str


class Vehicle(ApiModel):
    id: str | None = None
    nickname: str | None = None
    input_mode: Literal["details", "vin"]
    year: str | None = Field(default=None, pattern=r"^\d{4}$")
    make: str | None = None
    model: str | None = None
    vin: str | None = Field(default=None, min_length=17, max_length=17)

    @model_validator(mode="after")
    def validate_vehicle_input(self):
        if self.input_mode == "vin" and not self.vin:
            raise ValueError("A VIN is required when input_mode is 'vin'")
        if self.input_mode == "details" and not all(
            (self.year, self.make, self.model)
        ):
            raise ValueError(
                "Year, make, and model are required when input_mode is 'details'"
            )
        return self


class Attachment(ApiModel):
    id: str
    name: str = Field(min_length=1, max_length=255)
    type: Literal["image/jpeg", "image/png", "image/webp"]
    size: int | None = Field(default=None, ge=1, le=8 * 1024 * 1024)


class AttachmentUploadResponse(ApiModel):
    attachments: list[Attachment]


class ChatStreamRequest(ApiModel):
    request_id: str | None = None
    conversation_id: str
    message: str = Field(min_length=1)
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    vehicle: Vehicle | None = None
    attachments: list[Attachment] = Field(default_factory=list, max_length=5)


class VinDecodeRequest(ApiModel):
    vin: str = Field(min_length=17, max_length=17)


class VinDecodeResponse(ApiModel):
    year: str = Field(pattern=r"^\d{4}$")
    make: str
    model: str


class Citation(ApiModel):
    id: int
    title: str
    source_type: Literal["retrieval", "nhtsa"]
    detail: str
    url: str | None = None


class CancelResponse(ApiModel):
    request_id: str
    status: Literal["cancellation_requested", "not_found"]


class ComponentStatus(ApiModel):
    status: Literal["available", "unavailable"]


class CapabilityStatus(ApiModel):
    streaming: bool
    cancellation: bool
    vision: bool
    max_images: int


class HealthResponse(ApiModel):
    status: Literal["healthy", "degraded"]
    ollama: ComponentStatus
    chroma: ComponentStatus
    model: ComponentStatus
    capabilities: CapabilityStatus


class QueueStatus(ApiModel):
    request_id: str
    position: int
    estimated_wait_seconds: int
