import { lazy, Suspense, useCallback, useState } from 'react'
import type { Topic } from '../types'
import { GeminiStealthPanel } from './GeminiStealthPanel'

const CodeBlock = lazy(async () => {
  const { CodeBlock: C } = await import('./CodeBlock')
  return { default: C }
})

type Props = {
  sections: Topic[]
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      className={`h-5 w-5 shrink-0 text-slate-500 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={2}
      stroke="currentColor"
      aria-hidden
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
    </svg>
  )
}

export function SectionAccordion({ sections }: Props) {
  const [open, setOpen] = useState<Set<string>>(() => new Set())

  const toggle = useCallback((id: string) => {
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  return (
    <div className="space-y-3">
      {sections.map((s) => {
        const isOpen = open.has(s.id)
        const panelId = `${s.id}-panel`
        const headerId = `${s.id}-header`
        return (
          <section
            key={s.id}
            id={s.id}
            className="scroll-mt-24 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm"
          >
            <button
              type="button"
              id={headerId}
              className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-slate-50 focus-visible:relative focus-visible:z-10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2f8d46]"
              onClick={() => toggle(s.id)}
              aria-expanded={isOpen}
              aria-controls={panelId}
            >
              <span className="min-w-0">
                <span className="block font-semibold text-[#1b3d2c]">{s.title}</span>
                {s.hint ? (
                  <span className="mt-0.5 block text-sm text-slate-600">{s.hint}</span>
                ) : null}
              </span>
              <Chevron open={isOpen} />
            </button>

            {isOpen ? (
              <div
                id={panelId}
                role="region"
                aria-labelledby={headerId}
                className="border-t border-slate-200 bg-[#f8fafc] p-3 sm:p-4"
              >
                {s.kind === 'stealth-gemini' ? (
                  <GeminiStealthPanel />
                ) : (
                  <Suspense
                    fallback={
                      <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-900 p-4 font-mono text-[13px] text-slate-300">
                        Loading highlighter…
                      </pre>
                    }
                  >
                    <CodeBlock id={s.id} code={s.code ?? ''} />
                  </Suspense>
                )}
              </div>
            ) : null}
          </section>
        )
      })}
    </div>
  )
}
