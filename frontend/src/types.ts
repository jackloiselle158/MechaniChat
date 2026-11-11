export type Source = {
  id: number;
  title: string;
  detail: string;
  source_type?: "retrieval" | "nhtsa";
  url?: string;
};

export type Attachment = {
  id: string;
  name: string;
  type: string;
  size?: number;
  previewUrl?: string;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  attachments?: Attachment[];
  sources?: Source[];
  status?: "streaming" | "complete" | "error";
};

export type Conversation = {
  id: string;
  title: string;
  customTitle?: boolean;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
  vehicleId?: string;
};

export type Vehicle = {
  id: string;
  nickname: string;
  input_mode: "details" | "vin";
  year?: string;
  make?: string;
  model?: string;
  vin?: string;
};

export type DecodedVehicle = {
  year: string;
  make: string;
  model: string;
};

export type AppData = {
  conversations: Conversation[];
  vehicles: Vehicle[];
  activeConversationId: string;
};

export type ChatRequest = {
  request_id: string;
  conversation_id: string;
  message: string;
  history: Array<{ role: "user" | "assistant"; content: string }>;
  vehicle?: Vehicle;
  attachments: Array<{ id: string; name: string; type: string; size?: number }>;
};

export type ChatStreamEvent =
  | { type: "queue.updated"; position: number; estimated_wait_seconds: number }
  | { type: "message.started" }
  | { type: "message.delta"; text: string }
  | { type: "citation"; citation: Source }
  | { type: "message.completed"; duration_seconds?: number }
  | { type: "message.stopped" }
  | { type: "error"; code: string; message: string };

export type HealthStatus = {
  status: "healthy" | "degraded";
  ollama: { status: "available" | "unavailable" };
  chroma: { status: "available" | "unavailable" };
  model: { status: "available" | "unavailable" };
  capabilities: {
    streaming: boolean;
    cancellation: boolean;
    vision: boolean;
    max_images: number;
  };
};
