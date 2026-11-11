# MechaniChat

MechaniChat is a local automotive assistant with a React/Vite chat interface and a FastAPI backend. Conversations, saved vehicles, and uploaded photo blobs currently stay in the user's browser. The browser sends the relevant text history with each request, so the API remains stateless.

## How the pieces work together

- `frontend/` contains the React user interface. Vite runs its development server and proxies `/api` requests to FastAPI on port 8000.
- `backend/api.py` exposes the REST and Server-Sent Events endpoints, controls the one-request Ollama queue, reports queue estimates, handles cancellation, and accepts multipart image uploads.
- `backend/chat_service.py` contains the Ollama response and vision paths, validated tool routing, the tool loop, citations, and token streaming.
- `backend/image_store.py` validates uploaded images, strips metadata by re-encoding them, bounds their dimensions, and manages temporary server files.
- `backend/tools.py` contains the live NHTSA calls, including VIN decoding.
- `backend/retrieval.py` reads the local Chroma collection.
- `main.py` is the temporary Streamlit interface. It calls the same backend chat service as the API.

All API request bodies, response bodies, and SSE event data use Python-style `snake_case` field names. For example: `request_id`, `conversation_id`, `input_mode`, and `estimated_wait_seconds`.

## Run locally

Prerequisites: Python 3.11 or newer, Node.js/npm, Ollama, `qwen3.5:4b`, `huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest`, and `embeddinggemma-2:270m`.

The 4B Qwen model generates chat responses and analyzes uploaded images because the installed model supports vision and tools. `MECHANICHAT_VISION_MODEL` can select a different vision-capable Ollama model later. A separate 3B Llama model performs only the small route-classification and missing-argument extraction tasks. It runs with temperature zero, a 2K context window, and CPU-only inference so it does not displace the response model from limited GPU memory. The router model can be changed with `MECHANICHAT_ROUTER_MODEL`.

## Image request flow

1. The browser keeps the original JPEG, PNG, or WebP in IndexedDB so it remains visible in conversation history.
2. When the user sends the message, the frontend uploads up to five images through `POST /api/v1/attachments` using multipart form data.
3. The API validates the real image format and size, removes metadata, limits the longest edge to 2048 pixels, and stores a temporary JPEG under a random server ID.
4. The SSE chat request contains only those server-issued IDs. The backend converts the cleaned images to Ollama-compatible image content and selects the configured vision model.
5. The temporary files are removed when generation completes, fails, times out, or is cancelled. A background sweeper removes abandoned uploads after 15 minutes, and shutdown cleanup removes the remaining files.

Image limits and temporary storage can be changed with the `MECHANICHAT_MAX_IMAGE_*`, `MECHANICHAT_TEMPORARY_IMAGE_TTL_SECONDS`, and `MECHANICHAT_TEMPORARY_IMAGE_DIRECTORY` environment variables. Run the API with one Uvicorn worker while its temporary upload registry is stored in memory.

From the project directory, create the Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Start Ollama, then start FastAPI in the first terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. FastAPI's interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`, and health information is at `http://127.0.0.1:8000/api/v1/health`.

When changing embedding models, rebuild the vectors before starting the API:

```powershell
python db.py
```

The migration reads the documents already stored in the old `docs` collection, normalizes and deduplicates them with stable content-based IDs, adds inferred vehicle/component metadata, embeds them in batches, and upserts them into `docs_retrieval_v2`. It can safely resume after an interruption. Until the migration finishes, the API continues using the complete `nomic-embed-text` collection. Only a completed run writes the readiness marker; the API selects EmbeddingGemma 2 after its next restart.

Retrieval embeds a standalone version of the question, asks Chroma for eight candidates, discards candidates outside the configured distance threshold, reranks the survivors using vector similarity, keyword overlap, and selected-vehicle metadata, and supplies the best three to the answer model. Each retained complaint appears as a separate source. Follow-up query rewriting uses the smaller router model, while the main chat model remains responsible for the final answer.

To use the frontend without the API, copy `frontend/.env.example` to `frontend/.env` and set `VITE_CHAT_MODE=mock`.

## Test

```powershell
python -m pytest backend/tests
cd frontend
npm test
```

The backend tests replace Ollama and NHTSA responses with local fakes. They do not call either service.

## Proxmox deployment note

The local default deliberately binds FastAPI to `127.0.0.1`. When the app is containerized on Proxmox, the API process will need to listen on `0.0.0.0` inside its container, while access is restricted at the container, firewall, and tunnel layers. The Chroma `mydb` directory should use persistent storage. If Ollama runs in another container, set `MECHANICHAT_OLLAMA_URL` to that container's private network address. Authentication and the Cloudflare Tunnel can be added at deployment time without changing the browser-owned conversation design.

The API health response reports whether Ollama confirms the configured vision model has the `vision` capability.
