/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CHAT_MODE?: "api" | "mock";
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
