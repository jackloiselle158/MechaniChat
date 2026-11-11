import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv


# Application paths
project_directory = Path(__file__).resolve().parent.parent
load_dotenv(project_directory / ".env")
database_path = os.getenv("MECHANICHAT_DATABASE_PATH", str(project_directory / "mydb"))

# Ollama settings
ollama_url = os.getenv("MECHANICHAT_OLLAMA_URL", "http://127.0.0.1:11434")
chat_model = os.getenv(
    "MECHANICHAT_CHAT_MODEL",
    "qwen3.5:4b",
)
vision_model = os.getenv(
    "MECHANICHAT_VISION_MODEL",
    chat_model,
)
router_model = os.getenv(
    "MECHANICHAT_ROUTER_MODEL",
    "huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest",
)
reasoning_setting = os.getenv("MECHANICHAT_MODEL_REASONING", "false").lower()
if reasoning_setting == "true":
    model_reasoning = True
elif reasoning_setting == "false":
    model_reasoning = False
else:
    model_reasoning = reasoning_setting
model_context_length = int(os.getenv("MECHANICHAT_MODEL_CONTEXT_LENGTH", "8192"))
router_context_length = int(os.getenv("MECHANICHAT_ROUTER_CONTEXT_LENGTH", "2048"))
router_num_gpu = int(os.getenv("MECHANICHAT_ROUTER_NUM_GPU", "0"))

# Each embedding model needs its own Chroma collection.
source_collection_name = os.getenv(
    "MECHANICHAT_SOURCE_COLLECTION_NAME",
    "docs",
)
migration_embedding_model = os.getenv(
    "MECHANICHAT_MIGRATION_EMBEDDING_MODEL",
    "embeddinggemma-2:270m",
)
migration_collection_name = os.getenv(
    "MECHANICHAT_MIGRATION_COLLECTION_NAME",
    "docs_retrieval_v2",
)
migration_marker = Path(database_path) / ".retrieval_v2_ready"

# Keep using the complete legacy collection until the batch migration finishes.
if migration_marker.exists():
    default_embedding_model = migration_embedding_model
    default_collection_name = migration_collection_name
else:
    default_embedding_model = "nomic-embed-text"
    default_collection_name = source_collection_name

embedding_model = os.getenv(
    "MECHANICHAT_EMBEDDING_MODEL",
    default_embedding_model,
)
collection_name = os.getenv(
    "MECHANICHAT_COLLECTION_NAME",
    default_collection_name,
)

# Retrieval settings. Chroma distances are lower for more similar documents.
retrieval_candidate_count = int(
    os.getenv("MECHANICHAT_RETRIEVAL_CANDIDATE_COUNT", "8")
)
retrieval_result_count = int(
    os.getenv("MECHANICHAT_RETRIEVAL_RESULT_COUNT", "3")
)
default_distance_threshold = (
    "0.50" if embedding_model == migration_embedding_model else "0.78"
)
retrieval_distance_threshold = float(
    os.getenv("MECHANICHAT_RETRIEVAL_DISTANCE_THRESHOLD", default_distance_threshold)
)
embedding_batch_size = int(os.getenv("MECHANICHAT_EMBEDDING_BATCH_SIZE", "64"))

# API settings
api_host = os.getenv("MECHANICHAT_API_HOST", "127.0.0.1")
api_port = int(os.getenv("MECHANICHAT_API_PORT", "8000"))
request_timeout = int(os.getenv("MECHANICHAT_REQUEST_TIMEOUT", "120"))
nhtsa_timeout = int(os.getenv("MECHANICHAT_NHTSA_TIMEOUT", "15"))
max_queue_size = int(os.getenv("MECHANICHAT_MAX_QUEUE_SIZE", "5"))
default_generation_seconds = int(
    os.getenv("MECHANICHAT_DEFAULT_GENERATION_SECONDS", "30")
)

# Uploaded images are short-lived server data. They are re-encoded before use
# and removed after the chat request finishes.
temporary_image_directory = Path(
    os.getenv(
        "MECHANICHAT_TEMPORARY_IMAGE_DIRECTORY",
        str(Path(tempfile.gettempdir()) / "mechanichat_uploads"),
    )
)
max_image_size = int(os.getenv("MECHANICHAT_MAX_IMAGE_SIZE", str(8 * 1024 * 1024)))
max_image_count = int(os.getenv("MECHANICHAT_MAX_IMAGE_COUNT", "5"))
max_image_pixels = int(os.getenv("MECHANICHAT_MAX_IMAGE_PIXELS", "25000000"))
max_image_dimension = int(os.getenv("MECHANICHAT_MAX_IMAGE_DIMENSION", "2048"))
temporary_image_ttl_seconds = int(
    os.getenv("MECHANICHAT_TEMPORARY_IMAGE_TTL_SECONDS", "900")
)

# Only the local React development servers are accepted during the prototype.
default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("MECHANICHAT_ALLOWED_ORIGINS", default_origins).split(",")
    if origin.strip()
]
