/** Gemini REST — API key must come from env, never from source control. */

const PYTHON_REPLY_RULES = `You answer ML / coding questions with **Python 3** only (numpy, pandas, scikit-learn, matplotlib, etc.). Never use MATLAB or other languages unless the user explicitly forbids Python.

**Output format (mandatory, every time):**
1) Start with a markdown heading exactly: ## Code
2) Under it, put **one** fenced code block with language tag python. That block must contain **all executable code** and may include # comments inside. Do not put prose sentences inside the code block except short # comments.
3) Then a markdown heading exactly: ## Explanation
4) Under it, write plain explanation only: no fenced code blocks, no indented code, no triple backticks. Do not mix explanation into the code section or code into the explanation section.`

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

/** See https://ai.google.dev/gemini-api/docs/models for current IDs. */
const MODEL = 'gemini-3-flash-preview'

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
      systemInstruction: {
        parts: [{ text: PYTHON_REPLY_RULES }],
      },
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: userMessage,
            },
          ],
        },
      ],
      generationConfig: {
        maxOutputTokens: 4096,
        temperature: 0.35,
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
