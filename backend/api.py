import asyncio
import json
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import ollama
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import settings
from backend.chat_service import stream_chat
from backend.image_store import ImageValidationError, image_store
from backend.retrieval import collection
from backend.schemas import Attachment, AttachmentUploadResponse
from backend.schemas import CancelResponse, CapabilityStatus, ChatStreamRequest
from backend.schemas import ComponentStatus, HealthResponse, QueueStatus
from backend.schemas import VinDecodeRequest, VinDecodeResponse
from backend.tools import decode_vin


async def remove_stale_images():
    cleanup_interval = min(60, max(1, settings.temporary_image_ttl_seconds))
    while True:
        await asyncio.sleep(cleanup_interval)
        await asyncio.to_thread(image_store.cleanup_stale)


@asynccontextmanager
async def lifespan(_):
    await asyncio.to_thread(image_store.cleanup_stale)
    cleanup_task = asyncio.create_task(remove_stale_images())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await asyncio.to_thread(image_store.clear)


app = FastAPI(
    title="MechaniChat API",
    version="1.0.0",
    description="Local streaming API for the MechaniChat automotive assistant.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# Only one request can use Ollama at a time during the prototype.
generation_lock = asyncio.Semaphore(1)
queue_lock = asyncio.Lock()
queue_order = []
request_controls = {}
generation_durations = deque(maxlen=10)


def format_sse(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


def get_average_generation_seconds() -> int:
    if not generation_durations:
        return settings.default_generation_seconds
    return max(1, round(sum(generation_durations) / len(generation_durations)))


async def get_queue_status(request_id: str) -> QueueStatus:
    async with queue_lock:
        if request_id not in queue_order:
            position = 0
        else:
            position = queue_order.index(request_id) + 1
    return QueueStatus(
        request_id=request_id,
        position=position,
        estimated_wait_seconds=position * get_average_generation_seconds(),
    )


async def register_request(request_id: str):
    async with queue_lock:
        if len(request_controls) >= settings.max_queue_size + 1:
            raise HTTPException(
                status_code=503,
                detail="The local model queue is full. Please try again shortly.",
            )
        if request_id in request_controls:
            raise HTTPException(
                status_code=409,
                detail="That request ID is already active.",
            )

        cancel_event = asyncio.Event()
        request_controls[request_id] = {
            "cancel_event": cancel_event,
            "task": None,
            "started_at": None,
        }
        queue_order.append(request_id)
    return cancel_event


async def remove_request(request_id: str):
    async with queue_lock:
        if request_id in queue_order:
            queue_order.remove(request_id)
        request_controls.pop(request_id, None)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    ollama_available = False
    model_available = False
    vision_available = False
    chroma_available = False

    try:
        ollama_client = ollama.Client(host=settings.ollama_url)
        response = await asyncio.to_thread(ollama_client.list)
        ollama_available = True
        model_names = []
        if isinstance(response, dict):
            available_models = response.get("models", [])
        else:
            available_models = getattr(response, "models", [])
        for model in available_models:
            if isinstance(model, dict):
                model_names.append(model.get("model") or model.get("name"))
            else:
                model_names.append(getattr(model, "model", None))
        required_models = {
            settings.chat_model,
            settings.router_model,
            settings.vision_model,
        }
        model_available = required_models.issubset(set(model_names))
        if settings.vision_model in model_names:
            model_information = await asyncio.to_thread(
                ollama_client.show,
                settings.vision_model,
            )
            if isinstance(model_information, dict):
                capabilities = model_information.get("capabilities", [])
            else:
                capabilities = getattr(model_information, "capabilities", [])
            vision_available = "vision" in capabilities
    except Exception:
        pass

    try:
        await asyncio.to_thread(collection.count)
        chroma_available = True
    except Exception:
        pass

    overall_status = (
        "healthy"
        if ollama_available and model_available and vision_available and chroma_available
        else "degraded"
    )
    return HealthResponse(
        status=overall_status,
        ollama=ComponentStatus(
            status="available" if ollama_available else "unavailable"
        ),
        chroma=ComponentStatus(
            status="available" if chroma_available else "unavailable"
        ),
        model=ComponentStatus(
            status="available" if model_available else "unavailable"
        ),
        capabilities=CapabilityStatus(
            streaming=True,
            cancellation=True,
            vision=vision_available,
            max_images=settings.max_image_count,
        ),
    )


@app.get("/api/v1/queue/{request_id}", response_model=QueueStatus)
async def queue_status(request_id: str):
    if request_id not in request_controls:
        raise HTTPException(status_code=404, detail="Request not found")
    return await get_queue_status(request_id)


@app.post("/api/v1/vehicles/decode-vin", response_model=VinDecodeResponse)
async def decode_vehicle_vin(request: VinDecodeRequest):
    try:
        vehicle = await asyncio.to_thread(decode_vin, request.vin)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="The NHTSA VIN decoder is currently unavailable.",
        ) from error

    if not all(vehicle.get(field) for field in ("year", "make", "model")):
        raise HTTPException(
            status_code=422,
            detail="NHTSA could not identify a year, make, and model for that VIN.",
        )

    return VinDecodeResponse(
        year=vehicle["year"],
        make=vehicle["make"],
        model=vehicle["model"],
    )


@app.post("/api/v1/attachments", response_model=AttachmentUploadResponse)
async def upload_attachments(files: list[UploadFile] = File(...)):
    if not files or len(files) > settings.max_image_count:
        raise HTTPException(
            status_code=422,
            detail=f"Upload between one and {settings.max_image_count} images.",
        )

    saved_attachments = []
    try:
        for file in files:
            data = await file.read(settings.max_image_size + 1)
            record = await asyncio.to_thread(
                image_store.save,
                file.filename or "image",
                file.content_type or "",
                data,
            )
            saved_attachments.append(record)
    except ImageValidationError as error:
        image_store.delete([record.id for record in saved_attachments])
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        image_store.delete([record.id for record in saved_attachments])
        raise
    finally:
        for file in files:
            await file.close()

    return AttachmentUploadResponse(
        attachments=[
            Attachment(
                id=record.id,
                name=record.name,
                type=record.type,
                size=record.size,
            )
            for record in saved_attachments
        ]
    )


@app.delete("/api/v1/chat/{request_id}", response_model=CancelResponse)
async def cancel_chat(request_id: str):
    control = request_controls.get(request_id)
    if control is None:
        return CancelResponse(request_id=request_id, status="not_found")

    control["cancel_event"].set()
    task = control.get("task")
    if task and not task.done():
        task.cancel()
    return CancelResponse(
        request_id=request_id,
        status="cancellation_requested",
    )


@app.post(
    "/api/v1/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "A Server-Sent Events stream of queue, message, citation, completion, and error events.",
            "content": {
                "text/event-stream": {
                    "example": "event: message.delta\ndata: {\"request_id\":\"...\",\"text\":\"Hello\"}\n\n"
                }
            },
        },
        503: {"description": "The local Ollama queue is full."},
    },
)
async def chat_stream(chat_request: ChatStreamRequest, request: Request):
    request_id = chat_request.request_id or str(uuid4())
    attachment_ids = [attachment.id for attachment in chat_request.attachments]
    try:
        image_paths = image_store.get_paths(attachment_ids)
    except KeyError as error:
        image_store.delete(attachment_ids)
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded image {error.args[0]} is missing or expired.",
        ) from error

    try:
        cancel_event = await register_request(request_id)
    except Exception:
        image_store.delete(attachment_ids)
        raise

    async def generate_events():
        acquired_lock = False
        started_at = None
        try:
            # Wait for the Ollama generation slot and report queue progress.
            while not acquired_lock:
                if cancel_event.is_set() or await request.is_disconnected():
                    raise asyncio.CancelledError
                try:
                    await asyncio.wait_for(generation_lock.acquire(), timeout=1)
                    acquired_lock = True
                except asyncio.TimeoutError:
                    status = await get_queue_status(request_id)
                    yield format_sse(
                        "queue.updated",
                        status.model_dump(),
                    )

            async with queue_lock:
                if request_id in queue_order:
                    queue_order.remove(request_id)
                control = request_controls.get(request_id)
                if control:
                    control["task"] = asyncio.current_task()
                    control["started_at"] = time.monotonic()

            started_at = time.monotonic()
            yield format_sse("message.started", {"request_id": request_id})

            history = [
                message.model_dump()
                for message in chat_request.history
            ]
            vehicle = (
                chat_request.vehicle.model_dump()
                if chat_request.vehicle
                else None
            )

            async with asyncio.timeout(settings.request_timeout):
                async for event in stream_chat(
                    chat_request.message,
                    history,
                    vehicle,
                    cancel_event,
                    image_paths=image_paths,
                ):
                    if await request.is_disconnected():
                        cancel_event.set()
                        raise asyncio.CancelledError

                    if event["type"] == "delta":
                        yield format_sse(
                            "message.delta",
                            {"request_id": request_id, "text": event["text"]},
                        )
                    elif event["type"] == "citation":
                        citation = event["citation"].model_dump()
                        yield format_sse(
                            "citation",
                            {"request_id": request_id, "citation": citation},
                        )

            duration = round(time.monotonic() - started_at, 2)
            yield format_sse(
                "message.completed",
                {"request_id": request_id, "duration_seconds": duration},
            )
        except asyncio.CancelledError:
            yield format_sse("message.stopped", {"request_id": request_id})
        except TimeoutError:
            yield format_sse(
                "error",
                {
                    "request_id": request_id,
                    "code": "request_timeout",
                    "message": "The model did not finish within two minutes.",
                },
            )
        except Exception:
            yield format_sse(
                "error",
                {
                    "request_id": request_id,
                    "code": "generation_failed",
                    "message": "MechaniChat could not complete the response.",
                },
            )
        finally:
            if acquired_lock:
                generation_lock.release()
            if started_at is not None:
                generation_durations.append(time.monotonic() - started_at)
            image_store.delete(attachment_ids)
            await remove_request(request_id)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
