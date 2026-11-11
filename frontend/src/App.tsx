import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import {
  cancelChatResponse,
  decodeVin,
  getApiHealth,
  isMockMode,
  streamChatResponse,
  uploadAttachments,
} from "./api/chat";
import {
  clearLocalData,
  deleteAttachment,
  loadAppData,
  loadAttachmentBlob,
  loadAttachmentUrl,
  saveAppData,
  saveAttachment,
} from "./storage/localData";
import type {
  AppData,
  Attachment,
  ChatRequest,
  Conversation,
  Message,
  Vehicle,
} from "./types";

const uid = () => crypto.randomUUID();

const getVehicleDescription = (vehicle: Vehicle): string => {
  if (vehicle.year && vehicle.make && vehicle.model) {
    return `${vehicle.year} ${vehicle.make} ${vehicle.model}`;
  }
  return "Vehicle details unavailable";
};

const summarizeRequest = (request: string): string => {
  const normalized = request.replace(/\s+/g, " ").trim();
  if (!normalized) return "Photo diagnosis";

  const concise = normalized
    .replace(/^(?:please\s+)?(?:can|could|would)\s+you\s+/i, "")
    .replace(/^please\s+/i, "")
    .replace(/^i(?:'d| would) like (?:you )?to\s+/i, "")
    .trim();
  const sentence = concise.split(/(?<=[.!?])\s/)[0];
  const firstSentence = `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}`;
  const maximumLength = 64;
  if (firstSentence.length <= maximumLength) return firstSentence;

  const shortened = firstSentence.slice(0, maximumLength + 1);
  const lastSpace = shortened.lastIndexOf(" ");
  return `${shortened.slice(0, lastSpace > 36 ? lastSpace : maximumLength).trim()}…`;
};

const resizeChatInput = (textArea: HTMLTextAreaElement | null) => {
  if (!textArea) return;
  const styles = window.getComputedStyle(textArea);
  const fontSize = Number.parseFloat(styles.fontSize) || 14;
  const measuredLineHeight = Number.parseFloat(styles.lineHeight);
  const lineHeight = Number.isFinite(measuredLineHeight) && measuredLineHeight > 3
    ? measuredLineHeight
    : fontSize * 1.4;
  const verticalPadding = (Number.parseFloat(styles.paddingTop) || 0)
    + (Number.parseFloat(styles.paddingBottom) || 0);
  const maximumHeight = lineHeight * 8 + verticalPadding;

  textArea.style.height = "auto";
  const contentHeight = textArea.scrollHeight;
  textArea.style.height = `${Math.max(44, Math.min(contentHeight, maximumHeight))}px`;
  textArea.style.overflowY = contentHeight > maximumHeight ? "auto" : "hidden";
};

const createConversation = (): Conversation => ({
  id: uid(),
  title: "New conversation",
  createdAt: Date.now(),
  updatedAt: Date.now(),
  messages: [],
});

function initialData(): AppData {
  const stored = loadAppData();
  if (stored?.conversations.length) {
    return {
      ...stored,
      conversations: stored.conversations.map((conversation) => ({
        ...conversation,
        ...(() => {
          const firstUserMessage = conversation.messages.find((message) => message.role === "user");
          const oldGeneratedTitle = firstUserMessage
            ? firstUserMessage.content.slice(0, 42) || "Photo diagnosis"
            : "New conversation";
          const isGeneratedTitle = conversation.customTitle === false
            || conversation.title === oldGeneratedTitle
            || conversation.title === "Photo diagnosis"
            || conversation.title === "New conversation";
          return isGeneratedTitle && firstUserMessage
            ? { title: summarizeRequest(firstUserMessage.content), customTitle: false }
            : {};
        })(),
        messages: conversation.messages.map((message) =>
          message.status === "streaming" ? { ...message, status: "error" } : message,
        ),
      })),
    };
  }
  const conversation = createConversation();
  return { conversations: [conversation], vehicles: [], activeConversationId: conversation.id };
}

const Icon = ({ name, size = 20 }: { name: string; size?: number }) => {
  const paths: Record<string, ReactNode> = {
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    plus: <path d="M12 5v14M5 12h14" />,
    chat: <path d="M5 17.5 3.8 21l4.3-1.8c1.2.5 2.5.8 3.9.8 5 0 9-3.6 9-8s-4-8-9-8-9 3.6-9 8c0 2.1.8 4 2 5.5Z" />,
    edit: <path d="m14.7 5.3 4 4M4 20l3.8-.8L19 8a2.1 2.1 0 0 0-3-3L4.8 16.2 4 20Z" />,
    trash: <path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" />,
    copy: <path d="M8 8h11v12H8zM5 16H4V4h11v1" />,
    car: <><path d="m5 11 1.8-4h10.4l1.8 4M4 11h16v7H4z" /><path d="M7 18v2M17 18v2M7.5 14h.01M16.5 14h.01" /></>,
    camera: <><path d="M4 8h3l1.5-2h7L17 8h3v11H4z" /><circle cx="12" cy="13" r="3" /></>,
    send: <path d="m4 4 17 8-17 8 3-8-3-8Zm3 8h14" />,
    stop: <rect x="7" y="7" width="10" height="10" rx="1" />,
    wrench: <><path d="M14.7 6.3a4 4 0 0 0-5-5l2.1 2.1-2.4 2.4-2.1-2.1a4 4 0 0 0 5 5L19 15.4a2.5 2.5 0 1 1-3.5 3.5l-6.7-6.7" /><circle cx="17.2" cy="17.2" r=".6" fill="currentColor" /></>,
    chevron: <path d="m8 10 4 4 4-4" />,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M5 15v5h14v-5" /></>,
    retry: <path d="M20 7v5h-5M4 17v-5h5M6.1 8A7 7 0 0 1 18 6l2 6M18 16a7 7 0 0 1-11.9 2L4 12" />,
    check: <path d="m5 12 4 4L19 6" />,
    external: <path d="M14 4h6v6M20 4l-9 9M18 13v7H4V6h7" />,
  };
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {paths[name]}
    </svg>
  );
};

function VehicleModal({ vehicles, selectedVehicleId, onClose, onSave, onSelect, onDelete }: {
  vehicles: Vehicle[];
  selectedVehicleId?: string;
  onClose: () => void;
  onSave: (vehicle: Vehicle) => Promise<void>;
  onSelect: (id?: string) => void;
  onDelete: (id: string) => void;
}) {
  const [mode, setMode] = useState<"details" | "vin">("details");
  const [nickname, setNickname] = useState("");
  const [year, setYear] = useState("");
  const [makeChoice, setMakeChoice] = useState("");
  const [customMake, setCustomMake] = useState("");
  const [modelChoice, setModelChoice] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [vin, setVin] = useState("");
  const [editingVehicleId, setEditingVehicleId] = useState<string>();
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string>();
  const years = Array.from({ length: 47 }, (_, index) => String(2026 - index));
  const makes = ["Chevrolet", "Dodge", "Ford", "GMC", "Honda", "Subaru", "Toyota"];
  const models: Record<string, string[]> = {
    Chevrolet: ["Equinox", "Malibu", "Silverado"], Dodge: ["Challenger", "Charger", "Durango"],
    Ford: ["Escape", "F-150", "Mustang"], GMC: ["Acadia", "Sierra", "Terrain"],
    Honda: ["Accord", "Civic", "CR-V"], Subaru: ["Forester", "Impreza", "Outback"],
    Toyota: ["Camry", "Corolla", "RAV4"],
  };

  const resetForm = () => {
    setMode("details"); setNickname(""); setYear(""); setMakeChoice("");
    setCustomMake(""); setModelChoice(""); setCustomModel(""); setVin("");
    setEditingVehicleId(undefined); setSaveError(undefined);
  };

  const startEditing = (vehicle: Vehicle) => {
    const knownMake = makes.find((item) => item.toLowerCase() === vehicle.make?.toLowerCase());
    const knownModel = knownMake
      ? models[knownMake]?.find((item) => item.toLowerCase() === vehicle.model?.toLowerCase())
      : undefined;
    setEditingVehicleId(vehicle.id);
    setMode(vehicle.input_mode);
    setNickname(vehicle.nickname);
    setYear(vehicle.year ?? "");
    setMakeChoice(knownMake ?? (vehicle.make ? "Other" : ""));
    setCustomMake(knownMake ? "" : vehicle.make ?? "");
    setModelChoice(knownModel ?? (vehicle.model ? "Other" : ""));
    setCustomModel(knownModel ? "" : vehicle.model ?? "");
    setVin(vehicle.vin ?? "");
    setSaveError(undefined);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const make = makeChoice === "Other" ? customMake.trim() : makeChoice;
    const model = modelChoice === "Other" ? customModel.trim() : modelChoice;
    if (mode === "details" && (!year || !make || !model)) return;
    if (mode === "vin" && vin.trim().length !== 17) return;
    setIsSaving(true);
    setSaveError(undefined);
    try {
      await onSave({
        id: editingVehicleId ?? uid(), nickname: nickname.trim() || "My car", input_mode: mode,
        ...(mode === "details" ? { year, make, model } : { vin: vin.trim().toUpperCase() }),
      });
      onClose();
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "The vehicle could not be saved.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="vehicle-modal" role="dialog" aria-modal="true" aria-labelledby="vehicle-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div><p className="eyebrow">Garage</p><h2 id="vehicle-title">Your vehicles</h2></div>
          <button className="icon-button" onClick={onClose} aria-label="Close vehicle manager"><Icon name="close" /></button>
        </div>

        {vehicles.length > 0 && (
          <div className="saved-vehicle-list">
            <div className="saved-vehicle-heading">
              <span>Choose a vehicle for this conversation</span>
              {selectedVehicleId && <button type="button" onClick={() => { onSelect(); onClose(); }}>Use no vehicle</button>}
            </div>
            {vehicles.map((vehicle) => (
              <div className={`saved-vehicle ${selectedVehicleId === vehicle.id ? "selected" : ""}`} key={vehicle.id}>
                <button
                  type="button"
                  className="saved-vehicle-choice"
                  onClick={() => { onSelect(vehicle.id); onClose(); }}
                  aria-label={`${selectedVehicleId === vehicle.id ? "Selected" : "Use"} ${vehicle.nickname} for this conversation`}
                >
                  <span className="vehicle-avatar"><Icon name="car" /></span>
                  <span className="saved-vehicle-copy"><strong>{vehicle.nickname}</strong><span>{getVehicleDescription(vehicle)}</span></span>
                  <span className="vehicle-choice-state">{selectedVehicleId === vehicle.id ? <><Icon name="check" size={15} /> Selected</> : "Use vehicle"}</span>
                </button>
                <button className="icon-button subtle" type="button" onClick={() => startEditing(vehicle)} aria-label={`Edit ${vehicle.nickname}`}><Icon name="edit" size={17} /></button>
                <button className="icon-button subtle" onClick={() => onDelete(vehicle.id)} aria-label={`Delete ${vehicle.nickname}`}><Icon name="trash" size={17} /></button>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={submit}>
          {editingVehicleId && <div className="vehicle-form-heading"><span><Icon name="edit" size={15} /> Editing saved vehicle</span><button type="button" onClick={resetForm}>Cancel edit</button></div>}
          <div className="segmented" aria-label="Vehicle entry method">
            <button type="button" className={mode === "details" ? "active" : ""} onClick={() => setMode("details")}>Year, make & model</button>
            <button type="button" className={mode === "vin" ? "active" : ""} onClick={() => setMode("vin")}>VIN</button>
          </div>
          <label className="field"><span>Nickname <small>Optional</small></span><input value={nickname} onChange={(event) => setNickname(event.target.value)} placeholder="My car" /></label>
          {mode === "details" ? (
            <div className="vehicle-fields">
              <label className="field"><span>Year</span><select required value={year} onChange={(event) => setYear(event.target.value)}><option value="">Select year</option>{years.map((item) => <option key={item}>{item}</option>)}</select></label>
              <label className="field"><span>Make</span><select required value={makeChoice} onChange={(event) => { setMakeChoice(event.target.value); setModelChoice(""); }}><option value="">Select make</option>{makes.map((item) => <option key={item}>{item}</option>)}<option>Other</option></select></label>
              {makeChoice === "Other" && <label className="field field-wide"><span>Enter make</span><input required value={customMake} onChange={(event) => setCustomMake(event.target.value)} placeholder="Vehicle make" /></label>}
              <label className="field"><span>Model</span><select required value={modelChoice} onChange={(event) => setModelChoice(event.target.value)}><option value="">Select model</option>{(models[makeChoice] ?? []).map((item) => <option key={item}>{item}</option>)}<option>Other</option></select></label>
              {modelChoice === "Other" && <label className="field field-wide"><span>Enter model</span><input required value={customModel} onChange={(event) => setCustomModel(event.target.value)} placeholder="Vehicle model" /></label>}
            </div>
          ) : (
            <label className="field"><span>Vehicle identification number</span><input required minLength={17} maxLength={17} value={vin} onChange={(event) => setVin(event.target.value)} placeholder="Enter the 17-character VIN" autoCapitalize="characters" /><small>Usually found on the dashboard near the windshield or inside the driver’s door.</small></label>
          )}
          {saveError && <div className="vehicle-save-error" role="alert">{saveError}</div>}
          <button className="primary-button modal-submit" type="submit" disabled={isSaving}><Icon name={editingVehicleId ? "check" : "plus"} size={18} /> {isSaving ? "Identifying vehicle…" : editingVehicleId ? "Update vehicle" : "Save vehicle"}</button>
        </form>
      </section>
    </div>
  );
}

function App() {
  const [data, setData] = useState<AppData>(initialData);
  const [draft, setDraft] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string>();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [vehicleModalOpen, setVehicleModalOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [copiedId, setCopiedId] = useState<string>();
  const [uploadError, setUploadError] = useState<string>();
  const [queueStatus, setQueueStatus] = useState<{ position: number; estimated_wait_seconds: number }>();
  const [apiStatus, setApiStatus] = useState<"mock" | "checking" | "healthy" | "degraded" | "offline">(isMockMode ? "mock" : "checking");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<{ controller: AbortController; request_id: string } | undefined>(undefined);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const vinDecodeAttemptsRef = useRef(new Set<string>());

  const activeConversation = data.conversations.find((item) => item.id === data.activeConversationId) ?? data.conversations[0];
  const activeVehicle = data.vehicles.find((item) => item.id === activeConversation?.vehicleId);

  useEffect(() => saveAppData(data), [data]);

  useEffect(() => {
    for (const vehicle of data.vehicles) {
      const vehicleVin = vehicle.vin;
      if (
        vehicle.input_mode !== "vin"
        || !vehicleVin
        || (vehicle.year && vehicle.make && vehicle.model)
        || vinDecodeAttemptsRef.current.has(vehicle.id)
      ) continue;

      vinDecodeAttemptsRef.current.add(vehicle.id);
      void decodeVin(vehicleVin).then((details) => {
        setData((current) => ({
          ...current,
          vehicles: current.vehicles.map((item) => (
            item.id === vehicle.id ? { ...item, ...details } : item
          )),
        }));
      }).catch(() => {
        // Keep the VIN private and show a neutral fallback until decoding succeeds later.
      });
    }
  }, [data.vehicles]);

  useEffect(() => {
    if (isMockMode) return;
    void getApiHealth().then((health) => {
      setApiStatus(health?.status ?? "offline");
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const hydrateImages = async () => {
      const updates = new Map<string, string>();
      for (const conversation of data.conversations) {
        for (const message of conversation.messages) {
          for (const attachment of message.attachments ?? []) {
            if (!attachment.previewUrl) {
              const url = await loadAttachmentUrl(attachment.id);
              if (url) updates.set(attachment.id, url);
            }
          }
        }
      }
      if (!cancelled && updates.size) {
        setData((current) => ({ ...current, conversations: current.conversations.map((conversation) => ({ ...conversation, messages: conversation.messages.map((message) => ({ ...message, attachments: message.attachments?.map((attachment) => ({ ...attachment, previewUrl: attachment.previewUrl ?? updates.get(attachment.id) })) })) })) }));
      }
    };
    void hydrateImages();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversation?.messages, isStreaming]);

  useEffect(() => {
    resizeChatInput(chatInputRef.current);
  }, [draft]);

  const sortedConversations = useMemo(() => [...data.conversations].sort((a, b) => b.updatedAt - a.updatedAt), [data.conversations]);

  const updateConversation = (id: string, update: (conversation: Conversation) => Conversation) => {
    setData((current) => ({ ...current, conversations: current.conversations.map((conversation) => conversation.id === id ? update(conversation) : conversation) }));
  };

  const stopResponse = () => {
    const activeRequest = abortRef.current;
    if (!activeRequest) return;
    void cancelChatResponse(activeRequest.request_id);
    activeRequest.controller.abort();
  };

  const newConversation = () => {
    if (isStreaming) stopResponse();
    const conversation = createConversation();
    setData((current) => ({ ...current, conversations: [conversation, ...current.conversations], activeConversationId: conversation.id }));
    setDraft(""); setPendingAttachments([]); setSidebarOpen(false);
  };

  const deleteConversation = (id: string) => {
    const target = data.conversations.find((item) => item.id === id);
    target?.messages.flatMap((message) => message.attachments ?? []).forEach((attachment) => void deleteAttachment(attachment.id));
    setData((current) => {
      const remaining = current.conversations.filter((item) => item.id !== id);
      if (remaining.length) return { ...current, conversations: remaining, activeConversationId: current.activeConversationId === id ? remaining[0].id : current.activeConversationId };
      const replacement = createConversation();
      return { ...current, conversations: [replacement], activeConversationId: replacement.id };
    });
  };

  const renameConversation = (conversation: Conversation) => {
    const title = window.prompt("Rename conversation", conversation.title)?.trim();
    if (title) updateConversation(conversation.id, (item) => ({ ...item, title, customTitle: true, updatedAt: Date.now() }));
  };

  const addFiles = async (files: FileList | File[]) => {
    const acceptedTypes = ["image/jpeg", "image/png", "image/webp"];
    const selectedFiles = Array.from(files);
    const images = selectedFiles
      .filter((file) => acceptedTypes.includes(file.type) && file.size <= 8 * 1024 * 1024)
      .slice(0, 5 - pendingAttachments.length);
    if (images.length !== selectedFiles.length) {
      setUploadError("Use up to five JPEG, PNG, or WebP images no larger than 8 MB each.");
    } else {
      setUploadError(undefined);
    }
    const saved = await Promise.all(images.map(saveAttachment));
    setPendingAttachments((current) => [...current, ...saved].slice(0, 5));
  };

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) void addFiles(event.target.files);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault(); setDragActive(false); void addFiles(event.dataTransfer.files);
  };

  const removePendingAttachment = (attachment: Attachment) => {
    setPendingAttachments((current) => current.filter((item) => item.id !== attachment.id));
    void deleteAttachment(attachment.id);
  };

  const runResponse = async (
    conversation_id: string,
    prompt: string,
    history: Message[],
    attachments: Attachment[] = [],
  ) => {
    const assistantId = uid();
    const request_id = uid();
    const controller = new AbortController();
    abortRef.current = { controller, request_id };
    setIsStreaming(true);
    setQueueStatus(undefined);
    updateConversation(conversation_id, (conversation) => ({ ...conversation, messages: [...conversation.messages, { id: assistantId, role: "assistant", content: "", createdAt: Date.now(), status: "streaming" }] }));

    const chatRequest: ChatRequest = {
      request_id,
      conversation_id,
      message: prompt,
      history: history.map((message) => ({ role: message.role, content: message.content })),
      vehicle: activeVehicle,
      attachments: [],
    };

    try {
      if (attachments.length) {
        const files = await Promise.all(attachments.map(async (attachment) => {
          const blob = await loadAttachmentBlob(attachment.id);
          if (!blob) {
            throw new Error(`The photo ${attachment.name} is no longer available in this browser.`);
          }
          return { attachment, blob };
        }));
        chatRequest.attachments = await uploadAttachments(files);
      }

      for await (const event of streamChatResponse(chatRequest, controller.signal)) {
        if (event.type === "queue.updated") {
          setQueueStatus({ position: event.position, estimated_wait_seconds: event.estimated_wait_seconds });
          continue;
        }
        if (event.type === "message.started") {
          setQueueStatus(undefined);
          continue;
        }
        updateConversation(conversation_id, (conversation) => ({ ...conversation, updatedAt: Date.now(), messages: conversation.messages.map((message) => {
          if (message.id !== assistantId) return message;
          if (event.type === "message.delta") return { ...message, content: message.content + event.text };
          if (event.type === "citation") {
            const sources = message.sources ?? [];
            return sources.some((source) => source.id === event.citation.id)
              ? message
              : { ...message, sources: [...sources, event.citation] };
          }
          if (event.type === "error") return { ...message, status: "error", content: message.content || event.message };
          if (event.type === "message.stopped") return { ...message, status: "complete", content: message.content || "Response stopped." };
          return { ...message, status: "complete" };
        }) }));
        if (event.type === "error" || event.type === "message.stopped") return;
      }
    } catch (error) {
      const stopped = error instanceof DOMException && error.name === "AbortError";
      const errorMessage = error instanceof Error ? error.message : "I couldn’t complete that response.";
      updateConversation(conversation_id, (conversation) => ({ ...conversation, messages: conversation.messages.map((message) => message.id === assistantId ? { ...message, status: stopped ? "complete" : "error", content: message.content || (stopped ? "Response stopped." : errorMessage) } : message) }));
    } finally {
      setIsStreaming(false);
      setQueueStatus(undefined);
      if (abortRef.current?.request_id === request_id) abortRef.current = undefined;
    }
  };

  const submitMessage = (event?: FormEvent) => {
    event?.preventDefault();
    const prompt = draft.trim();
    if ((!prompt && !pendingAttachments.length) || isStreaming || !activeConversation) return;
    const userMessage: Message = { id: uid(), role: "user", content: prompt || "Please examine the attached image.", createdAt: Date.now(), attachments: pendingAttachments };
    const editingIndex = editingMessageId ? activeConversation.messages.findIndex((message) => message.id === editingMessageId) : -1;
    const priorMessages = editingIndex >= 0 ? activeConversation.messages.slice(0, editingIndex) : activeConversation.messages;
    updateConversation(activeConversation.id, (conversation) => {
      const prior = editingIndex >= 0 ? conversation.messages.slice(0, editingIndex) : conversation.messages;
      const firstUserMessage = prior.every((message) => message.role !== "user");
      return {
        ...conversation,
        title: firstUserMessage ? summarizeRequest(userMessage.content) : conversation.title,
        customTitle: firstUserMessage ? false : conversation.customTitle,
        updatedAt: Date.now(),
        messages: [...prior, userMessage],
      };
    });
    setDraft(""); setPendingAttachments([]); setEditingMessageId(undefined);
    void runResponse(activeConversation.id, userMessage.content, priorMessages, userMessage.attachments);
  };

  const editMessage = (message: Message) => {
    setDraft(message.content); setPendingAttachments(message.attachments ?? []); setEditingMessageId(message.id);
    document.querySelector<HTMLTextAreaElement>("#chat-input")?.focus();
  };

  const retryMessage = (messageId: string) => {
    if (!activeConversation || isStreaming) return;
    const index = activeConversation.messages.findIndex((message) => message.id === messageId);
    let previousUserIndex = -1;
    for (let messageIndex = index - 1; messageIndex >= 0; messageIndex -= 1) {
      if (activeConversation.messages[messageIndex].role === "user") {
        previousUserIndex = messageIndex;
        break;
      }
    }
    const previousUser = activeConversation.messages[previousUserIndex];
    if (!previousUser) return;
    updateConversation(activeConversation.id, (conversation) => ({ ...conversation, messages: conversation.messages.slice(0, index) }));
    void runResponse(activeConversation.id, previousUser.content, activeConversation.messages.slice(0, previousUserIndex), previousUser.attachments);
  };

  const saveVehicle = async (vehicle: Vehicle) => {
    const existingVehicle = data.vehicles.find((item) => item.id === vehicle.id);
    let savedVehicle = vehicle;

    if (vehicle.input_mode === "vin" && vehicle.vin) {
      const canReuseDetails = existingVehicle?.input_mode === "vin"
        && existingVehicle.vin === vehicle.vin
        && existingVehicle.year
        && existingVehicle.make
        && existingVehicle.model;
      const details = canReuseDetails
        ? {
          year: existingVehicle.year as string,
          make: existingVehicle.make as string,
          model: existingVehicle.model as string,
        }
        : await decodeVin(vehicle.vin);
      savedVehicle = { ...vehicle, ...details };
    }

    setData((current) => {
      const isExistingVehicle = current.vehicles.some((item) => item.id === savedVehicle.id);
      return {
        ...current,
        vehicles: isExistingVehicle
          ? current.vehicles.map((item) => item.id === savedVehicle.id ? savedVehicle : item)
          : [...current.vehicles, savedVehicle],
        conversations: isExistingVehicle
          ? current.conversations
          : current.conversations.map((conversation) => (
            conversation.id === current.activeConversationId
              ? { ...conversation, vehicleId: savedVehicle.id }
              : conversation
          )),
      };
    });
  };
  const selectVehicle = (id?: string) => setData((current) => ({ ...current, conversations: current.conversations.map((conversation) => conversation.id === current.activeConversationId ? { ...conversation, vehicleId: id } : conversation) }));
  const deleteVehicle = (id: string) => setData((current) => ({ ...current, vehicles: current.vehicles.filter((item) => item.id !== id), conversations: current.conversations.map((conversation) => conversation.vehicleId === id ? { ...conversation, vehicleId: undefined } : conversation) }));

  const clearEverything = async () => {
    if (!window.confirm("Clear every locally saved conversation, vehicle, and photo? This cannot be undone.")) return;
    await clearLocalData();
    setData(initialData()); setPendingAttachments([]); setVehicleModalOpen(false);
  };

  const copyMessage = async (message: Message) => {
    await navigator.clipboard.writeText(message.content); setCopiedId(message.id);
    window.setTimeout(() => setCopiedId(undefined), 1600);
  };

  return (
    <div className="app-shell">
      <button className="mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open conversations"><Icon name="menu" /></button>
      {sidebarOpen && <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="Close conversations" />}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand-row"><div className="brand-mark"><Icon name="wrench" size={22} /></div><div><strong>MechaniChat</strong><span>Automotive assistant</span></div><button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><Icon name="close" /></button></div>
        <button className="new-chat" onClick={newConversation} aria-label="Start a new conversation"><Icon name="plus" size={18} /> New conversation</button>
        <div className="conversation-heading"><span>Recent</span><span>{data.conversations.length}</span></div>
        <nav className="conversation-list" aria-label="Conversations">
          {sortedConversations.map((conversation) => {
            const conversationVehicle = data.vehicles.find((vehicle) => vehicle.id === conversation.vehicleId);
            return (
              <div className={`conversation-item ${conversation.id === activeConversation.id ? "active" : ""}`} key={conversation.id}>
                <button className="conversation-select" onClick={() => { setData((current) => ({ ...current, activeConversationId: conversation.id })); setSidebarOpen(false); }}>
                  <Icon name="chat" size={18} />
                  <span className="conversation-copy">
                    <strong>{conversation.title}</strong>
                    {conversationVehicle && <small><Icon name="car" size={12} /> {conversationVehicle.nickname} · {getVehicleDescription(conversationVehicle)}</small>}
                  </span>
                </button>
                <div className="conversation-actions"><button onClick={() => renameConversation(conversation)} aria-label={`Rename ${conversation.title}`}><Icon name="edit" size={15} /></button><button onClick={() => deleteConversation(conversation.id)} aria-label={`Delete ${conversation.title}`}><Icon name="trash" size={15} /></button></div>
              </div>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <div className="demo-status"><span className={`status-dot ${apiStatus === "degraded" || apiStatus === "checking" ? "amber" : apiStatus === "offline" ? "red" : ""}`} /><div><strong>{apiStatus === "mock" ? "Frontend preview" : "Local API"}</strong><span>{apiStatus === "mock" ? "Mock responses enabled" : apiStatus === "checking" ? "Checking connection…" : apiStatus === "offline" ? "API is offline" : `${apiStatus[0].toUpperCase()}${apiStatus.slice(1)}`}</span></div></div>
          <button className="clear-data" onClick={() => void clearEverything()}><Icon name="trash" size={17} /> Clear local data</button>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div><p className="eyebrow">Active vehicle</p><button className="vehicle-trigger" aria-label={activeVehicle ? `Manage active vehicle ${activeVehicle.nickname} ${getVehicleDescription(activeVehicle)}` : "Add your vehicle"} onClick={() => setVehicleModalOpen(true)}><span className="mini-car"><Icon name="car" size={18} /></span><span>{activeVehicle ? activeVehicle.nickname : "Add your vehicle"}<small>{activeVehicle && getVehicleDescription(activeVehicle)}</small></span><Icon name="chevron" size={17} /></button></div>
          <div className="connection-pill"><span className={`status-dot ${apiStatus === "healthy" ? "" : apiStatus === "offline" ? "red" : "amber"}`} /> {apiStatus === "mock" ? "Demo mode" : apiStatus === "healthy" ? "API connected" : apiStatus === "offline" ? "API offline" : "API degraded"}</div>
        </header>

        <section className={`message-area ${activeConversation.messages.length ? "has-messages" : ""}`} aria-live="polite">
          {!activeConversation.messages.length ? (
            <div className="empty-state">
              <div className="hero-illustration" aria-hidden="true"><div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" /><div className="hero-car"><Icon name="car" size={72} /></div><div className="spark spark-one">✦</div><div className="spark spark-two">✦</div></div>
              <p className="eyebrow">Your garage, decoded</p>
              <h1>Clear answers for<br /><em>the road ahead.</em></h1>
              <p className="hero-copy">Describe what your car is doing, add a photo if it helps, and get practical guidance in plain language.</p>
              <div className="trust-row"><span><Icon name="check" size={15} /> Vehicle-aware</span><span><Icon name="check" size={15} /> Source-backed</span><span><Icon name="check" size={15} /> Private by default</span></div>
            </div>
          ) : (
            <div className="messages">
              {activeConversation.messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <div className="message-identity">{message.role === "assistant" ? <div className="assistant-avatar"><Icon name="wrench" size={17} /></div> : <span>You</span>}</div>
                  <div className="message-body">
                    {!!message.attachments?.length && <div className="message-images">{message.attachments.map((attachment) => attachment.previewUrl ? <img src={attachment.previewUrl} alt={attachment.name} key={attachment.id} /> : <div className="image-placeholder" key={attachment.id}><Icon name="camera" /></div>)}</div>}
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                    {message.status === "streaming" && <span className="typing-cursor" aria-label="Generating response" />}
                    {!!message.sources?.length && <details className="sources" open><summary>Sources <span>{message.sources.length}</span></summary>{message.sources.map((source) => <div className="source-card" key={source.id}><span className="source-number">{source.id}</span><div><strong>{source.title}</strong><p>{source.detail}</p></div>{source.url && <a href={source.url} target="_blank" rel="noreferrer" aria-label={`Open ${source.title}`}><Icon name="external" size={16} /></a>}</div>)}</details>}
                    <div className="message-tools">
                      {message.role === "assistant" && message.status !== "streaming" && <button onClick={() => void copyMessage(message)}><Icon name={copiedId === message.id ? "check" : "copy"} size={15} />{copiedId === message.id ? "Copied" : "Copy"}</button>}
                      {message.role === "assistant" && message.status === "error" && <button onClick={() => retryMessage(message.id)}><Icon name="retry" size={15} />Retry</button>}
                      {message.role === "user" && <button onClick={() => editMessage(message)}><Icon name="edit" size={15} />Edit</button>}
                    </div>
                  </div>
                </article>
              ))}
              <div ref={messageEndRef} />
            </div>
          )}
        </section>

        <div className="composer-wrap">
          {queueStatus && <div className="queue-banner"><span className="status-dot amber" /><span>Queued at position {queueStatus.position}. Estimated wait: about {queueStatus.estimated_wait_seconds} seconds.</span></div>}
          {!!pendingAttachments.length && <div className="attachment-strip">{pendingAttachments.map((attachment) => <div className="attachment-preview" key={attachment.id}>{attachment.previewUrl && <img src={attachment.previewUrl} alt={attachment.name} />}<button onClick={() => removePendingAttachment(attachment)} aria-label={`Remove ${attachment.name}`}><Icon name="close" size={14} /></button></div>)}</div>}
          {uploadError && <div className="upload-error" role="alert">{uploadError}</div>}
          {editingMessageId && <div className="editing-banner"><span><Icon name="edit" size={15} /> Editing your message</span><button onClick={() => { setEditingMessageId(undefined); setDraft(""); setPendingAttachments([]); }}>Cancel</button></div>}
          <form className="composer" onSubmit={submitMessage}>
            <div className="text-composer">
              <textarea ref={chatInputRef} id="chat-input" rows={1} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitMessage(); } }} placeholder="Describe what’s happening with your car…" aria-label="Message MechaniChat" />
              {isStreaming ? <button className="send-button stop-button" type="button" onClick={stopResponse} aria-label="Stop response"><Icon name="stop" /></button> : <button className="send-button" type="submit" disabled={!draft.trim() && !pendingAttachments.length} aria-label="Send message"><Icon name="send" /></button>}
            </div>
            <button type="button" className={`upload-box ${dragActive ? "dragging" : ""}`} onClick={() => fileInputRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)} onDrop={handleDrop}>
              <span className="upload-icon"><Icon name="camera" /></span><span><strong>Add photos</strong><small>Dash lights, tires, leaks</small></span>
            </button>
            <input ref={fileInputRef} type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" multiple hidden onChange={handleFileInput} />
          </form>
          <p className="disclaimer">MechaniChat can make mistakes. For safety-critical issues, consult a qualified mechanic.</p>
        </div>
      </main>
      {vehicleModalOpen && <VehicleModal vehicles={data.vehicles} selectedVehicleId={activeConversation?.vehicleId} onClose={() => setVehicleModalOpen(false)} onSave={saveVehicle} onSelect={selectVehicle} onDelete={deleteVehicle} />}
    </div>
  );
}

export default App;
