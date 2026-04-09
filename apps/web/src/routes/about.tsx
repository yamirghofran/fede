import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <main className="page-wrap px-4 py-12">
      <section className="island-shell rounded-2xl p-6 sm:p-8">
        <p className="island-kicker mb-2">About</p>
        <h1 className="display-title mb-3 text-4xl font-bold text-[var(--sea-ink)] sm:text-5xl">
          FEDE query modes in one place.
        </h1>
        <p className="m-0 max-w-3xl text-base leading-8 text-[var(--sea-ink-soft)]">
          This demo frontend talks to the FastAPI backend and lets you compare
          pure semantic retrieval, graph-driven narrative search, and the
          hybrid pipeline that combines both. It is meant for inspecting how
          the same natural-language question behaves across retrieval modes.
        </p>
      </section>
    </main>
  )
}
