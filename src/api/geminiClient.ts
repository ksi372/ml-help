/** In-browser Gemini REST calls. Prefer env in production; fallback below if unset. */
const HARDCODED_KEY = 'AIzaSyBhYgANQlY-MHlzbfemDh8qeyxOBdsm4O8'

export function getGeminiApiKey(): string {
  const fromEnv = import.meta.env.VITE_GEMINI_API_KEY
  if (typeof fromEnv === 'string' && fromEnv.length > 0) return fromEnv
  return HARDCODED_KEY
}

/** Swap to e.g. `gemini-1.5-flash` if your key rejects this model. */
const MODEL = 'gemini-2.0-flash'

function apiBase(): string {
  return import.meta.env.DEV
    ? '/api/gemini'
    : 'https://generativelanguage.googleapis.com'
}

export async function generateGeminiText(userMessage: string): Promise<string> {
  const key = getGeminiApiKey()
  const url = `${apiBase()}/v1beta/models/${MODEL}:generateContent?key=${encodeURIComponent(key)}`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: userMessage }] }],
      /** Shorter max reply = less generation time than an unbounded answer. */
      generationConfig: {
        maxOutputTokens: 2048,
        temperature: 0.4,
      },
    }),
  })
  if (!res.ok) {
    const errBody = await res.text()
    throw new Error(`Gemini ${res.status}: ${errBody.slice(0, 200)}`)
  }
  const data = (await res.json()) as {
    candidates?: { content?: { parts?: { text?: string }[] } }[]
  }
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text
  if (!text) throw new Error('No text in Gemini response')
  return text
}
