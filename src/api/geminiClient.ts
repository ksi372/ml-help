/** Gemini REST — API key must come from env, never from source control. */

export function getGeminiApiKey(): string {
  const raw = import.meta.env.VITE_GEMINI_API_KEY
  const key = typeof raw === 'string' ? raw.trim() : ''
  if (!key) {
    throw new Error(
      'Missing VITE_GEMINI_API_KEY. Add it to .env (local) or Render → Environment (production).',
    )
  }
  return key
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
