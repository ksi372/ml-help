import { SITE } from './config/site'
import { TOPICS } from './data/sections'
import { SectionAccordion } from './components/SectionAccordion'

function App() {
  return (
    <div className="min-h-svh bg-[#f6f7f9] text-slate-800">
      <header className="sticky top-0 z-40 border-b border-[#1b6b52] bg-[#2f8d46] shadow-sm">
        <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-3 sm:px-6">
          <span
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white/15 text-sm font-bold text-white ring-1 ring-white/30"
            aria-hidden
          >
            {SITE.brandShort}
          </span>
          <div className="min-w-0 text-left">
            <h1 className="truncate text-lg font-semibold tracking-tight text-white sm:text-xl">
              {SITE.title}
            </h1>
            <p className="truncate text-xs text-white/90 sm:text-sm">{SITE.tagline}</p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 pb-16 pt-6 text-left sm:px-6">
        <SectionAccordion sections={TOPICS} />
      </main>

      <footer className="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-500">
        Local notes · offline-friendly
      </footer>
    </div>
  )
}

export default App
