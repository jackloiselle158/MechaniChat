import { streamMockResponse } from "./mockChat";
import type { Attachment, ChatRequest, ChatStreamEvent, DecodedVehicle, HealthStatus, Source } from "../types";


const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";
export const isMockMode =
  import.meta.env.MODE === "test" || import.meta.env.VITE_CHAT_MODE === "mock";


function parseEvent(eventName: string, data: Record<string, unknown>): ChatStreamEvent | null {
  if (eventName === "queue.updated") {
    return {
      type: eventName,
      position: Number(data.position),
      estimated_wait_seconds: Number(data.estimated_wait_seconds),
    };
  }
  if (eventName === "message.started") return { type: eventName };
  if (eventName === "message.delta") {
    return { type: eventName, text: String(data.text ?? "") };
  }
  if (eventName === "citation") {
    return { type: eventName, citation: data.citation as Source };
  }
  if (eventName === "message.completed") {
    return { type: eventName, duration_seconds: Number(data.duration_seconds ?? 0) };
  }
  if (eventName === "message.stopped") return { type: eventName };
  if (eventName === "error") {
    return {
      type: eventName,
      code: String(data.code ?? "unknown_error"),
      message: String(data.message ?? "The request failed."),
    };
  }
  return null;
}


async function* streamApiResponse(
  chatRequest: ChatRequest,
  signal: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${apiBaseUrl}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(chatRequest),
    signal,
  });
  if (!response.ok || !response.body) {
    const message = response.status === 503
      ? "The local model queue is full. Please try again shortly."
      : "MechaniChat could not connect to the local API.";
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      let eventName = "message";
      let dataText = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataText += line.slice(5).trim();
      }
      if (!dataText) continue;
      const event = parseEvent(eventName, JSON.parse(dataText));
      if (event) yield event;
    }
    if (done) break;
  }
}


export async function* streamChatResponse(
  chatRequest: ChatRequest,
  signal: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  if (isMockMode) {
    for await (const event of streamMockResponse(
      chatRequest.message,
      chatRequest.vehicle,
      signal,
    )) {
      if (event.type === "delta") {
        yield { type: "message.delta", text: event.text };
      } else if (event.type === "sources") {
        for (const citation of event.sources) {
          yield { type: "citation", citation };
        }
      } else {
        yield { type: "message.completed" };
      }
    }
    return;
  }

  yield* streamApiResponse(chatRequest, signal);
}


export async function cancelChatResponse(request_id: string): Promise<void> {
  if (isMockMode) return;
  await fetch(`${apiBaseUrl}/api/v1/chat/${request_id}`, { method: "DELETE" });
}


export async function getApiHealth(): Promise<HealthStatus | null> {
  if (isMockMode) return null;
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/health`);
    if (!response.ok) return null;
    return await response.json() as HealthStatus;
  } catch {
    return null;
  }
}


export async function uploadAttachments(
  files: Array<{ attachment: Attachment; blob: Blob }>,
): Promise<Attachment[]> {
  if (isMockMode) return files.map(({ attachment }) => attachment);

  const formData = new FormData();
  for (const { attachment, blob } of files) {
    formData.append("files", blob, attachment.name);
  }
  const response = await fetch(`${apiBaseUrl}/api/v1/attachments`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    let message = "MechaniChat could not upload the selected photos.";
    try {
      const body = await response.json() as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the useful fallback when the API does not return JSON.
    }
    throw new Error(message);
  }
  const body = await response.json() as { attachments: Attachment[] };
  return body.attachments;
}


export async function decodeVin(vin: string): Promise<DecodedVehicle> {
  if (isMockMode) {
    throw new Error("VIN decoding requires the local API.");
  }
  const response = await fetch(`${apiBaseUrl}/api/v1/vehicles/decode-vin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vin }),
  });
  if (!response.ok) {
    let message = "NHTSA could not identify that VIN. Check it and try again.";
    try {
      const body = await response.json() as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the user-friendly fallback when the API does not return JSON.
    }
    throw new Error(message);
  }
  return await response.json() as DecodedVehicle;
}
