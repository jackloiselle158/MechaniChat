import asyncio
import base64
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import AsyncGenerator, Literal

from backend import settings
from backend.retrieval import get_relevant_context
from backend.schemas import Citation
from backend.tools import decode_vin, tools, tools_by_name


# Initialize one model that is configured in different ways for routing and responses
llm = ChatOllama(
    model=settings.chat_model,
    temperature=0.4,
    reasoning=settings.model_reasoning,
    num_ctx=settings.model_context_length,
    base_url=settings.ollama_url,
)

# Image turns can use a different model without changing ordinary text chat.
# By default this points to the same vision-capable qwen3.5 model.
vision_llm = ChatOllama(
    model=settings.vision_model,
    temperature=0.3,
    reasoning=settings.model_reasoning,
    num_ctx=settings.model_context_length,
    base_url=settings.ollama_url,
)

# Routing is a small, deterministic classification task. Keep it on the CPU so
# it does not compete with the response model for limited GPU memory.
router_llm = ChatOllama(
    model=settings.router_model,
    temperature=0,
    reasoning=False,
    num_ctx=settings.router_context_length,
    num_gpu=settings.router_num_gpu,
    num_predict=128,
    keep_alive="5m",
    base_url=settings.ollama_url,
)


class RouteDecision(BaseModel):
    """The single action needed for the user's latest request."""

    route: Literal[
        "answer",
        "get_recalls",
        "get_crash_ratings",
        "get_seat_stations",
    ] = Field(description="The tool to run, or 'answer' when no tool is needed.")


class VehicleArguments(BaseModel):
    """Vehicle fields extracted only when no complete saved vehicle exists."""

    year: str = Field(description="Four-digit model year, or an empty string.")
    make: str = Field(description="Vehicle manufacturer, or an empty string.")
    model: str = Field(description="Vehicle model, or an empty string.")


class StateArguments(BaseModel):
    """Two-letter state needed by the car-seat station tool."""

    state: str = Field(description="Two-letter US state abbreviation, or an empty string.")


class RetrievalQuery(BaseModel):
    """A standalone search query for the complaint database."""

    query: str = Field(description="A concise standalone automotive search query.")


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user's latest request into exactly one route.
Use get_recalls only when the user explicitly asks to look up vehicle recalls.
Use get_crash_ratings only when the user explicitly asks for NHTSA crash or safety ratings.
Use get_seat_stations only when the user explicitly asks for child car-seat inspection stations.
Use answer for everything else, including general questions about what a recall means.
Do not extract tool arguments. Return only the route.""",
        ),
        ("human", "{input}"),
    ]
)
router = router_prompt | router_llm.with_structured_output(
    RouteDecision,
    method="json_schema",
)

vehicle_argument_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract a vehicle's four-digit year, make, and model from the text.
Use an empty string for any field that is not explicitly present.
Never guess or substitute unrelated words.""",
        ),
        ("human", "{text}"),
    ]
)
vehicle_argument_extractor = vehicle_argument_prompt | router_llm.with_structured_output(
    VehicleArguments,
    method="json_schema",
)

state_argument_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract the US state from the request and return its two-letter abbreviation.
Use an empty string when no state is explicitly present. Never guess.""",
        ),
        ("human", "{text}"),
    ]
)
state_argument_extractor = state_argument_prompt | router_llm.with_structured_output(
    StateArguments,
    method="json_schema",
)

retrieval_query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the latest request as a concise, standalone automotive complaint search query.
Resolve references such as 'it', 'that', or 'the transmission' using the recent conversation.
Include the selected vehicle when it helps retrieval. Do not invent symptoms or vehicle details.
Return only the rewritten query.""",
        ),
        (
            "human",
            "Selected vehicle: {vehicle}\n\nRecent conversation:\n{conversation}",
        ),
    ]
)
retrieval_query_rewriter = retrieval_query_prompt | router_llm.with_structured_output(
    RetrievalQuery,
    method="json_schema",
)


# System prompt used for both ordinary responses and responses after tool calls
system_prompt = """
    You are a well-experienced mechanic. Your job is to help everyday drivers identify and solve car problems using clear language.
    CRITICAL TOOL USAGE RULES:
    - You have access to three tools: get_recalls, get_crash_ratings, and get_seat_stations
    - ONLY use get_recalls when the user EXPLICITLY asks about recalls for a specific vehicle
    - ONLY use get_crash_ratings when the user EXPLICITLY asks about NHTSA crash or safety ratings for a specific vehicle
    - ONLY use get_seat_stations when the user EXPLICITLY asks for child car-seat inspection stations in a specific state
    - DO NOT use these tools for general diagnosis or troubleshooting
    - If a user asks about a car problem, sound, or issue, answer from your knowledge and relevant documents

    RESPONSE RULES:
    1. Always consider the conversation history and selected vehicle.
    2. For a diagnostic answer, organize each likely problem under Cause, Likelihood, How to check, and Urgency.
    3. Give three likely causes when enough information is available.
    4. Ask for missing vehicle information when it is important to the diagnosis.
    5. Clearly recommend an in-person inspection for safety-critical brake, steering, tire, fuel, or airbag concerns.
    6. Cite supplied sources inline using their bracketed number, for example [1].
    7. Clearly label statements based only on general model knowledge when no supplied source supports them.
    8. If the prompt is unrelated to automobiles, answer it normally.
    9. NHTSA complaint records describe other owners' vehicles. Treat them as supporting examples, not proof that the selected vehicle has the same fault.

    IMAGE RULES:
    - Treat uploaded photos as supporting evidence, not definitive proof of a mechanical fault.
    - State what is visibly observable before drawing conclusions.
    - Say when an image is unclear, incomplete, or cannot confirm the suspected problem.
    - Never tell the user an unsafe vehicle is safe based only on a photo.

    Selected vehicle:
    {vehicle}

    Relevant documents and tool sources:
    {documents}
"""

conversation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
agent = conversation_prompt | llm.bind_tools(tools)
final_agent = conversation_prompt | llm
vision_agent = conversation_prompt | vision_llm.bind_tools(tools)
vision_final_agent = conversation_prompt | vision_llm


def convert_history(history: list[dict]) -> list:
    messages = []
    for message in history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
    return messages


def create_user_message(user_input: str, image_paths: list[Path] | None = None):
    if not image_paths:
        return HumanMessage(content=user_input)

    content = []
    for image_path in image_paths:
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_image}",
            }
        )
    content.append({"type": "text", "text": user_input})
    return HumanMessage(content=content)


def get_keyword_route(user_input: str) -> str:
    """Conservative fallback for explicit tool requests."""

    text = user_input.lower()
    recall_definition = any(
        phrase in text
        for phrase in (
            "what does recall mean",
            "what do recalls mean",
            "what is a recall",
            "define recall",
        )
    )
    explicit_recall = "recalls" in text or any(
        phrase in text
        for phrase in (
            "vehicle recall",
            "safety recall",
            "open recall",
            "recall for",
            "recall on",
            "recall apply",
            "recall status",
            "check recall",
            "look up recall",
            "is there a recall",
        )
    )
    if explicit_recall and not recall_definition:
        return "get_recalls"
    if "rating" in text and any(word in text for word in ("crash", "safety", "nhtsa")):
        return "get_crash_ratings"
    if (
        any(phrase in text for phrase in ("car seat", "child seat", "booster seat"))
        and any(word in text for word in ("station", "inspect", "inspection"))
    ):
        return "get_seat_stations"
    return "answer"


async def get_route(user_input: str) -> str:
    """Use the small model, with a safe keyword fallback on errors or misses."""

    keyword_route = get_keyword_route(user_input)
    if keyword_route != "answer":
        return keyword_route
    try:
        decision = await router.ainvoke({"input": user_input})
        return decision.route
    except Exception:
        return "answer"


def get_history_text(history: list[dict], user_input: str) -> str:
    recent_history = history[-6:]
    lines = [f"{message['role']}: {message['content']}" for message in recent_history]
    lines.append(f"user: {user_input}")
    return "\n".join(lines)


async def get_route_arguments(
    route: str,
    user_input: str,
    history: list[dict],
    vehicle: dict | None,
) -> dict | None:
    """Prefer validated app data and extract only genuinely missing fields."""

    if route in {"get_recalls", "get_crash_ratings"}:
        if vehicle and all(vehicle.get(name) for name in ("make", "model", "year")):
            return {
                "make": vehicle["make"],
                "model": vehicle["model"],
                "year": vehicle["year"],
            }
        try:
            arguments = await vehicle_argument_extractor.ainvoke(
                {"text": get_history_text(history, user_input)}
            )
        except Exception:
            return None
        if not (
            arguments.year.isdigit()
            and len(arguments.year) == 4
            and arguments.make.strip()
            and arguments.model.strip()
        ):
            return None
        return {
            "make": arguments.make.strip(),
            "model": arguments.model.strip(),
            "year": arguments.year,
        }

    if route == "get_seat_stations":
        try:
            arguments = await state_argument_extractor.ainvoke(
                {"text": get_history_text(history, user_input)}
            )
        except Exception:
            return None
        state = arguments.state.strip().upper()
        if len(state) != 2 or not state.isalpha():
            return None
        return {"state": state}

    return {}


def get_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)
    return str(content or "")


def get_vehicle_text(vehicle: dict | None) -> str:
    if not vehicle:
        return "No vehicle selected"
    if vehicle.get("year") and vehicle.get("make") and vehicle.get("model"):
        return f"{vehicle['year']} {vehicle['make']} {vehicle['model']}"
    if vehicle.get("vin"):
        return f"VIN {vehicle['vin']}"
    return "Vehicle details are incomplete"


async def rewrite_retrieval_query(
    user_input: str,
    history: list[dict],
    vehicle: dict | None,
) -> str:
    """Make follow-up questions searchable without spending the response model on it."""

    vehicle_text = get_vehicle_text(vehicle)
    if not history:
        if vehicle and vehicle_text != "Vehicle details are incomplete":
            return f"{vehicle_text} {user_input}"
        return user_input

    try:
        rewritten = await retrieval_query_rewriter.ainvoke(
            {
                "vehicle": vehicle_text,
                "conversation": get_history_text(history, user_input),
            }
        )
        if rewritten.query.strip():
            return rewritten.query.strip()
    except Exception:
        pass

    recent_user_messages = [
        message["content"]
        for message in history[-4:]
        if message["role"] == "user"
    ]
    parts = [vehicle_text] if vehicle else []
    parts.extend(recent_user_messages[-1:])
    parts.append(user_input)
    return " ".join(parts)


def get_retrieval_citation_title(metadata: dict) -> str:
    vehicle_parts = [metadata.get(field) for field in ("year", "make", "model")]
    vehicle_name = " ".join(str(part) for part in vehicle_parts if part)
    if vehicle_name:
        return f"NHTSA complaint — {vehicle_name}"
    return "NHTSA complaint record"


async def prepare_vehicle(vehicle: dict | None, citation_id: int):
    if not vehicle or vehicle.get("input_mode") != "vin" or not vehicle.get("vin"):
        return vehicle, None

    try:
        decoded = await asyncio.to_thread(decode_vin, vehicle["vin"])
    except Exception:
        return vehicle, None
    if not decoded.get("make"):
        return vehicle, None

    prepared_vehicle = {**vehicle, **decoded}
    citation = Citation(
        id=citation_id,
        title="NHTSA vPIC VIN Decoder",
        source_type="nhtsa",
        detail=f"VIN decoded as {decoded.get('year')} {decoded.get('make')} {decoded.get('model')}",
        url=f"https://vpic.nhtsa.dot.gov/decoder/Decoder?VIN={quote(vehicle['vin'])}",
    )
    return prepared_vehicle, citation


def create_tool_citation(tool_name: str, tool_args: dict, result: str, citation_id: int):
    if tool_name == "get_recalls":
        url = "https://www.nhtsa.gov/recalls"
        title = "NHTSA Vehicle Recalls"
    elif tool_name == "get_crash_ratings":
        url = "https://www.nhtsa.gov/ratings"
        title = "NHTSA Vehicle Safety Ratings"
    elif tool_name == "get_seat_stations":
        url = "https://www.nhtsa.gov/equipment/car-seats-and-booster-seats"
        title = "NHTSA Car Seat Inspection Stations"
    else:
        return None

    return Citation(
        id=citation_id,
        title=title,
        source_type="nhtsa",
        detail=result[:240],
        url=url,
    )


async def execute_tool_call(tool_call: dict, citation_id: int):
    selected_tool = tools_by_name.get(tool_call["name"])
    if selected_tool is None:
        result = f"Unknown tool: {tool_call['name']}"
        citation = None
    else:
        try:
            result = await asyncio.to_thread(
                selected_tool.invoke,
                tool_call["args"],
            )
            citation = create_tool_citation(
                tool_call["name"],
                tool_call["args"],
                str(result),
                citation_id,
            )
        except Exception as error:
            result = f"The tool failed: {error}"
            citation = None

    if citation:
        result = f"[{citation.id}] {result}"
    message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )
    return message, citation


# Stream one response while allowing the API to cancel between model chunks
async def stream_chat(
    user_input: str,
    history: list[dict],
    vehicle: dict | None,
    cancel_event: asyncio.Event,
    image_paths: list[Path] | None = None,
    max_tool_rounds: int = 5,
) -> AsyncGenerator[dict, None]:
    prior_messages = convert_history(history)
    citations = []
    citation_id = 1

    vehicle, vehicle_citation = await prepare_vehicle(vehicle, citation_id)
    if vehicle_citation:
        citations.append(vehicle_citation)
        citation_id += 1
        yield {"type": "citation", "citation": vehicle_citation}
    vehicle_text = get_vehicle_text(vehicle)

    search_query = await rewrite_retrieval_query(user_input, history, vehicle)
    try:
        contexts = await asyncio.to_thread(
            get_relevant_context,
            search_query,
            vehicle,
        )
    except Exception:
        contexts = []

    document_text = "No relevant context found"
    if contexts:
        document_parts = []
        for context in contexts:
            context_citation = Citation(
                id=citation_id,
                title=get_retrieval_citation_title(context.metadata),
                source_type="retrieval",
                detail=context.document[:240],
            )
            citations.append(context_citation)
            citation_id += 1
            document_parts.append(f"[{context_citation.id}] {context.document}")
            yield {"type": "citation", "citation": context_citation}
        document_text = "\n\n".join(document_parts)

    if cancel_event.is_set():
        raise asyncio.CancelledError

    route = await get_route(user_input)
    tool_arguments = await get_route_arguments(
        route,
        user_input,
        history,
        vehicle,
    )
    turn_messages = [create_user_message(user_input, image_paths)]
    selected_agent = vision_agent if image_paths else agent
    selected_final_agent = vision_final_agent if image_paths else final_agent

    if route != "answer" and tool_arguments is not None:
        initial_call = {
            "name": route,
            "args": tool_arguments,
            "id": f"route_{uuid4().hex}",
            "type": "tool_call",
        }
        routed_call = AIMessage(content="", tool_calls=[initial_call])
        tool_message, tool_citation = await execute_tool_call(initial_call, citation_id)
        turn_messages.extend([routed_call, tool_message])
        if tool_citation:
            citations.append(tool_citation)
            citation_id += 1
            document_text += f"\n[{tool_citation.id}] {tool_citation.title}"
            yield {"type": "citation", "citation": tool_citation}

    for _ in range(max_tool_rounds):
        if cancel_event.is_set():
            raise asyncio.CancelledError

        full_response = None
        async for chunk in selected_agent.astream(
            {
                "messages": prior_messages + turn_messages,
                "documents": document_text,
                "vehicle": vehicle_text,
            }
        ):
            if cancel_event.is_set():
                raise asyncio.CancelledError
            full_response = chunk if full_response is None else full_response + chunk
            text = get_text_from_content(chunk.content)
            if text:
                yield {"type": "delta", "text": text}

        if full_response is None:
            raise RuntimeError("The model returned an empty response")

        response = AIMessage(
            content=full_response.content,
            tool_calls=full_response.tool_calls,
        )
        turn_messages.append(response)
        if not response.tool_calls:
            return

        for tool_call in response.tool_calls:
            tool_message, tool_citation = await execute_tool_call(tool_call, citation_id)
            turn_messages.append(tool_message)
            if tool_citation:
                citations.append(tool_citation)
                citation_id += 1
                document_text += f"\n[{tool_citation.id}] {tool_citation.title}"
                yield {"type": "citation", "citation": tool_citation}

    # Stop an accidental tool cycle and stream a final response without tools.
    async for chunk in selected_final_agent.astream(
        {
            "messages": prior_messages + turn_messages,
            "documents": document_text,
            "vehicle": vehicle_text,
        }
    ):
        if cancel_event.is_set():
            raise asyncio.CancelledError
        text = get_text_from_content(chunk.content)
        if text:
            yield {"type": "delta", "text": text}


# Collect the stream for the temporary Streamlit interface
async def get_complete_response(user_input: str, history: list[dict]):
    content = ""
    citations = []
    cancel_event = asyncio.Event()
    async for event in stream_chat(user_input, history, None, cancel_event):
        if event["type"] == "delta":
            content += event["text"]
        elif event["type"] == "citation":
            citations.append(event["citation"])
    return content, citations
