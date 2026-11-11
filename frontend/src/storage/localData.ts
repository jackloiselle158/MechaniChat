import type { AppData, Attachment, Source, Vehicle } from "../types";

const APP_KEY = "mechanichat.app-data.v1";
const DB_NAME = "mechanichat-media";
const STORE_NAME = "attachments";

export function loadAppData(): AppData | null {
  try {
    const value = localStorage.getItem(APP_KEY);
    if (!value) return null;

    const stored = JSON.parse(value) as AppData;
    return {
      ...stored,
      conversations: stored.conversations.map((conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) => ({
          ...message,
          sources: message.sources?.map((source) => {
            const legacy_source = source as Source & {
              sourceType?: "retrieval" | "nhtsa";
            };
            if (source.source_type) return source;
            const { sourceType, ...current_source } = legacy_source;
            return {
              ...current_source,
              source_type: sourceType,
            };
          }),
        })),
      })),
      vehicles: stored.vehicles.map((vehicle) => {
        const legacy_vehicle = vehicle as Vehicle & {
          inputMode?: "details" | "vin";
        };
        if (vehicle.input_mode) return vehicle;
        const { inputMode, ...current_vehicle } = legacy_vehicle;
        return {
          ...current_vehicle,
          input_mode: inputMode ?? "details",
        };
      }),
    };
  } catch {
    return null;
  }
}

export function saveAppData(data: AppData): void {
  localStorage.setItem(
    APP_KEY,
    JSON.stringify(data, (key, value) => (key === "previewUrl" ? undefined : value)),
  );
}

function openMediaDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveAttachment(file: File): Promise<Attachment> {
  const id = crypto.randomUUID();
  const db = await openMediaDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(file, id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
  return {
    id,
    name: file.name,
    type: file.type,
    size: file.size,
    previewUrl: URL.createObjectURL(file),
  };
}

export async function loadAttachmentBlob(id: string): Promise<Blob | undefined> {
  const db = await openMediaDb();
  const blob = await new Promise<Blob | undefined>((resolve, reject) => {
    const request = db.transaction(STORE_NAME).objectStore(STORE_NAME).get(id);
    request.onsuccess = () => resolve(request.result as Blob | undefined);
    request.onerror = () => reject(request.error);
  });
  db.close();
  return blob;
}

export async function loadAttachmentUrl(id: string): Promise<string | undefined> {
  const blob = await loadAttachmentBlob(id);
  return blob ? URL.createObjectURL(blob) : undefined;
}

export async function deleteAttachment(id: string): Promise<void> {
  const db = await openMediaDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

export async function clearLocalData(): Promise<void> {
  localStorage.removeItem(APP_KEY);
  await new Promise<void>((resolve) => {
    const request = indexedDB.deleteDatabase(DB_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () => resolve();
    request.onblocked = () => resolve();
  });
}
