import { useCallback, useEffect, useRef, useState } from 'react'
import { generateGeminiText } from '../api/geminiClient'

export function GeminiStealthPanel() {
  const [value, setValue] = useState('')
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [reply, setReply] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const blankPanelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (reply && blankPanelRef.current) {
      blankPanelRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [reply])

  const copyReply = useCallback(async () => {
    if (!reply) return
    try {
      await navigator.clipboard.writeText(reply)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopied(false)
    }
  }, [reply])

  const send = useCallback(async () => {
    const q = value.trim()
    if (!q) return
    setErr(null)
    setReply(null)
    setLoading(true)
    try {
      const text = await generateGeminiText(q)
      setReply(text)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [value])

  return (
    <div className="relative min-h-[100px] rounded-md border border-dashed border-slate-200/80 bg-[#f8fafc]">
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder=" "
        spellCheck={false}
        className="min-h-[88px] w-full resize-y bg-transparent px-2 py-2 text-[13px] text-slate-200 placeholder:text-transparent focus:text-slate-700 focus:outline-none focus:placeholder:text-slate-400"
        aria-label="Input"
      />
      <div className="flex min-h-[28px] items-center justify-between gap-2 border-t border-transparent px-2 pb-2 pt-0">
        {loading ? (
          <span className="text-[11px] text-slate-500">Working…</span>
        ) : null}
        {!loading ? (
          <button
            type="button"
            onClick={send}
            disabled={loading}
            className="ml-auto rounded px-2 py-0.5 text-[10px] font-medium text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
          >
            Continue
          </button>
        ) : null}
      </div>
      {err ? (
        <p className="px-2 pb-2 text-[11px] text-red-600/90">{err}</p>
      ) : null}

      {/* Looks blank; full text only lives in memory and is copied on button press */}
      {reply ? (
        <div
          ref={blankPanelRef}
          className="relative mx-2 mb-2 min-h-[min(40vh,240px)] rounded-md border border-slate-100 bg-white"
        >
          <button
            type="button"
            onClick={copyReply}
            className="absolute left-3 top-3 z-10 rounded border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-500 shadow-sm hover:border-slate-300 hover:bg-slate-50 hover:text-slate-700"
            title="Copy"
          >
            {copied ? '✓' : 'Copy'}
          </button>
          {/* Screen-reader + layout only — not visible */}
          <span className="sr-only" aria-live="polite">
            Response ready. Use Copy to copy to clipboard.
          </span>
        </div>
      ) : null}
    </div>
  )
}
