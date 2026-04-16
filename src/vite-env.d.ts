/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Google Gemini API key — add when you wire up AI features. */
  readonly VITE_GEMINI_API_KEY?: string
}

declare module '*.m?raw' {
  const src: string
  export default src
}
