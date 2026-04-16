import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useCallback, useState } from 'react'

type Props = {
  code: string
  id: string
}

export function CodeBlock({ code, id }: Props) {
  const [copied, setCopied] = useState(false)

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopied(false)
    }
  }, [code])

  return (
    <div className="group relative overflow-hidden rounded-lg border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between gap-2 border-b border-slate-700/80 bg-[#1e293b] px-3 py-2 text-left text-xs text-slate-300">
        <span className="font-mono text-[11px] tracking-wide text-slate-400">
          MATLAB · {id}
        </span>
        <button
          type="button"
          onClick={copy}
          className="rounded border border-slate-500/80 bg-slate-800/80 px-2 py-1 text-[11px] font-medium text-slate-200 transition hover:bg-slate-700"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <SyntaxHighlighter
        language="matlab"
        style={oneDark}
        customStyle={{
          margin: 0,
          padding: '1rem 1rem 1.25rem',
          fontSize: '13px',
          lineHeight: 1.55,
          borderRadius: 0,
          maxHeight: 'min(70vh, 720px)',
        }}
        showLineNumbers
        wrapLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}
