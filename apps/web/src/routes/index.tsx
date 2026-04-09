import { useMemo, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useMutation } from '@tanstack/react-query'

import {
  searchHybrid,
  searchKnowledgeGraph,
  searchSemantic,
  type HybridResponse,
  type QueryMode,
  type SemanticResponse,
} from '#/lib/api'

export const Route = createFileRoute('/')({ component: App })

const MODE_COPY: Record<
  QueryMode,
  {
    label: string
    kicker: string
    description: string
    placeholder: string
  }
> = {
  semantic: {
    label: 'Semantic',
    kicker: 'Scene-Level Similarity',
    description:
      'Search the vector index for scripts and scenes that feel semantically close to the query, even when the relation vocabulary is implicit.',
    placeholder: 'A prodigy pushed toward obsession by a brutal mentor',
  },
  kb: {
    label: 'Knowledge Graph',
    kicker: 'Narrative Motif Search',
    description:
      'Use the LLM-to-graph translator to map natural language into story predicates like TEACHES, BETRAYS, SAVES, or CONFRONTS.',
    placeholder: 'Mentor teaches student, then turns against them',
  },
  hybrid: {
    label: 'Hybrid',
    kicker: 'Best Of Both',
    description:
      'Combine motif extraction from the knowledge graph with semantic recall from the script embeddings and rank by both signals.',
    placeholder: 'Movies where a mentor molds someone and later betrays them',
  },
}

type DemoState =
  | { mode: 'semantic'; payload: SemanticResponse }
  | { mode: 'kb' | 'hybrid'; payload: HybridResponse }

function App() {
  const [mode, setMode] = useState<QueryMode>('hybrid')
  const [query, setQuery] = useState(MODE_COPY.hybrid.placeholder)
  const [topK, setTopK] = useState(5)
  const [graphLimit, setGraphLimit] = useState(5)

  const queryMutation = useMutation<DemoState, Error, void>({
    mutationFn: async () => {
      if (mode === 'semantic') {
        const payload = await searchSemantic({ query, top_k: topK })
        return { mode, payload }
      }

      if (mode === 'kb') {
        const payload = await searchKnowledgeGraph({
          query,
          top_k: topK,
          graph_limit: graphLimit,
        })
        return { mode, payload }
      }

      const payload = await searchHybrid({
        query,
        top_k: topK,
        graph_limit: graphLimit,
      })
      return { mode, payload }
    },
  })

  const activeCopy = MODE_COPY[mode]
  const modeSummary = useMemo(() => {
    if (!queryMutation.data || queryMutation.data.mode === 'semantic') {
      return null
    }
    return queryMutation.data.payload.translation
  }, [queryMutation.data])

  return (
    <main className="page-wrap px-4 pb-10 pt-10">
      <section className="island-shell rise-in relative overflow-hidden rounded-[2rem] px-6 py-8 sm:px-10 sm:py-10">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-[linear-gradient(90deg,transparent,rgba(79,184,178,0.8),transparent)]" />
        <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <p className="island-kicker mb-3">FEDE Retrieval Console</p>
            <h1 className="display-title mb-4 max-w-4xl text-4xl leading-[1] font-bold tracking-tight text-[var(--sea-ink)] sm:text-6xl">
              Ask for a story pattern and inspect how each retrieval layer responds.
            </h1>
            <p className="max-w-2xl text-base leading-8 text-[var(--sea-ink-soft)] sm:text-lg">
              Switch between semantic recall, knowledge-graph motif search, and
              hybrid retrieval. The same prompt can retrieve scenes, graph
              paths, or both.
            </p>
          </div>

          <div className="rounded-[1.7rem] border border-[var(--line)] bg-[color:var(--surface)] p-4 shadow-[0_18px_38px_rgba(23,58,64,0.1)]">
            <div className="mb-4 flex flex-wrap gap-2">
              {(Object.keys(MODE_COPY) as QueryMode[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => {
                    setMode(item)
                    setQuery(MODE_COPY[item].placeholder)
                  }}
                  className={
                    item === mode
                      ? 'rounded-full border border-[rgba(50,143,151,0.3)] bg-[rgba(79,184,178,0.16)] px-4 py-2 text-sm font-semibold text-[var(--lagoon-deep)]'
                      : 'rounded-full border border-[var(--line)] bg-white/60 px-4 py-2 text-sm font-semibold text-[var(--sea-ink-soft)] transition hover:border-[rgba(50,143,151,0.3)] hover:text-[var(--sea-ink)]'
                  }
                >
                  {MODE_COPY[item].label}
                </button>
              ))}
            </div>

            <p className="mb-1 text-xs font-semibold tracking-[0.14em] text-[var(--kicker)] uppercase">
              {activeCopy.kicker}
            </p>
            <p className="mb-5 text-sm leading-7 text-[var(--sea-ink-soft)]">
              {activeCopy.description}
            </p>

            <form
              className="space-y-4"
              onSubmit={(event) => {
                event.preventDefault()
                queryMutation.mutate()
              }}
            >
              <label className="block">
                <span className="mb-2 block text-sm font-semibold text-[var(--sea-ink)]">
                  Query
                </span>
                <textarea
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  rows={4}
                  placeholder={activeCopy.placeholder}
                  className="w-full rounded-2xl border border-[var(--line)] bg-white/80 px-4 py-3 text-sm text-[var(--sea-ink)] outline-none transition focus:border-[rgba(50,143,151,0.5)] focus:ring-4 focus:ring-[rgba(79,184,178,0.14)]"
                />
              </label>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="block">
                  <span className="mb-2 block text-sm font-semibold text-[var(--sea-ink)]">
                    Top results
                  </span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                    className="w-full rounded-2xl border border-[var(--line)] bg-white/80 px-4 py-3 text-sm text-[var(--sea-ink)] outline-none transition focus:border-[rgba(50,143,151,0.5)] focus:ring-4 focus:ring-[rgba(79,184,178,0.14)]"
                  />
                </label>
                <label className="block">
                  <span className="mb-2 block text-sm font-semibold text-[var(--sea-ink)]">
                    Graph limit
                  </span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={graphLimit}
                    onChange={(event) => setGraphLimit(Number(event.target.value))}
                    disabled={mode === 'semantic'}
                    className="w-full rounded-2xl border border-[var(--line)] bg-white/80 px-4 py-3 text-sm text-[var(--sea-ink)] outline-none transition disabled:cursor-not-allowed disabled:opacity-50 focus:border-[rgba(50,143,151,0.5)] focus:ring-4 focus:ring-[rgba(79,184,178,0.14)]"
                  />
                </label>
              </div>

              <button
                type="submit"
                disabled={queryMutation.isPending || !query.trim()}
                className="inline-flex items-center justify-center rounded-full border border-[rgba(50,143,151,0.26)] bg-[linear-gradient(135deg,rgba(79,184,178,0.22),rgba(47,106,74,0.18))] px-5 py-3 text-sm font-semibold text-[var(--lagoon-deep)] transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {queryMutation.isPending ? 'Running query...' : `Run ${activeCopy.label} search`}
              </button>
            </form>
          </div>
        </div>
      </section>

      {queryMutation.error ? (
        <section className="mt-8 rounded-3xl border border-[rgba(182,71,71,0.24)] bg-[rgba(255,244,244,0.86)] px-5 py-4 text-sm text-[#8a2d2d] shadow-[0_12px_28px_rgba(138,45,45,0.08)]">
          <p className="m-0 font-semibold">Request failed</p>
          <p className="mt-1 mb-0">{queryMutation.error.message}</p>
        </section>
      ) : null}

      {queryMutation.data ? (
        <>
          {modeSummary ? (
            <section className="island-shell mt-8 rounded-3xl px-6 py-5">
              <div className="flex flex-wrap items-center gap-3">
                <p className="island-kicker m-0">Graph Translation</p>
                <span className="rounded-full border border-[var(--line)] bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--sea-ink-soft)]">
                  {modeSummary.status}
                </span>
              </div>
              {modeSummary.pattern ? (
                <div className="mt-4 flex flex-wrap gap-2">
                  {modeSummary.pattern.predicates.map((predicate) => (
                    <span
                      key={predicate}
                      className="rounded-full border border-[rgba(50,143,151,0.28)] bg-[rgba(79,184,178,0.14)] px-3 py-1 text-xs font-semibold tracking-[0.12em] text-[var(--lagoon-deep)] uppercase"
                    >
                      {predicate}
                    </span>
                  ))}
                </div>
              ) : null}
              {modeSummary.error ? (
                <p className="mt-3 mb-0 text-sm text-[var(--sea-ink-soft)]">
                  {modeSummary.error}
                </p>
              ) : null}
            </section>
          ) : null}

          <section className="mt-8 grid gap-4">
            {queryMutation.data.mode === 'semantic'
              ? queryMutation.data.payload.results.map((result) => (
                  <SemanticCard key={result.movie_id} result={result} />
                ))
              : queryMutation.data.payload.results.map((result) => (
                  <HybridCard key={result.movie_id} result={result} mode={queryMutation.data.mode} />
                ))}
          </section>
        </>
      ) : (
        <section className="mt-8 grid gap-4 lg:grid-cols-3">
          {[
            [
              'Semantic mode',
              'Best when the relation is implied in tone, dialogue, or scene structure rather than in a clean predicate chain.',
            ],
            [
              'KB mode',
              'Best when the user is really asking for a motif, such as who teaches, betrays, protects, or confronts whom.',
            ],
            [
              'Hybrid mode',
              'Best default. It keeps semantic recall but upgrades ranking when the knowledge graph can anchor a narrative pattern.',
            ],
          ].map(([title, body]) => (
            <article key={title} className="island-shell rounded-3xl p-5">
              <h2 className="mb-2 text-lg font-semibold text-[var(--sea-ink)]">
                {title}
              </h2>
              <p className="m-0 text-sm leading-7 text-[var(--sea-ink-soft)]">
                {body}
              </p>
            </article>
          ))}
        </section>
      )}
    </main>
  )
}

function SemanticCard({ result }: { result: SemanticResponse['results'][number] }) {
  return (
    <article className="island-shell rounded-3xl px-6 py-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="island-kicker mb-2">Semantic Match #{result.rank}</p>
          <h2 className="display-title m-0 text-3xl text-[var(--sea-ink)]">
            {result.movie_title}
          </h2>
          <p className="mt-2 mb-0 text-sm text-[var(--sea-ink-soft)]">
            `{result.movie_id}` · score {result.score.toFixed(3)}
          </p>
        </div>
        <div className="rounded-2xl border border-[var(--line)] bg-white/60 px-4 py-3 text-sm text-[var(--sea-ink-soft)]">
          Scene {result.best_scene.scene_index}
          {result.best_scene.scene_title ? ` · ${result.best_scene.scene_title}` : ''}
        </div>
      </div>
      <p className="mt-5 mb-0 text-sm leading-7 text-[var(--sea-ink-soft)]">
        {result.best_scene.text}
      </p>
      {result.best_scene.character_names.length ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {result.best_scene.character_names.map((name) => (
            <span
              key={name}
              className="rounded-full border border-[var(--line)] bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--sea-ink-soft)]"
            >
              {name}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  )
}

function HybridCard({
  result,
  mode,
}: {
  result: HybridResponse['results'][number]
  mode: 'kb' | 'hybrid'
}) {
  return (
    <article className="island-shell rounded-3xl px-6 py-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="island-kicker mb-2">
            {mode === 'kb' ? 'Knowledge Graph Match' : 'Hybrid Match'} #{result.rank}
          </p>
          <h2 className="display-title m-0 text-3xl text-[var(--sea-ink)]">
            {result.movie_title}
          </h2>
          <p className="mt-2 mb-0 text-sm text-[var(--sea-ink-soft)]">
            `{result.movie_id}` · combined score {result.score.toFixed(3)}
          </p>
        </div>
        <div className="grid gap-2 text-right text-sm text-[var(--sea-ink-soft)]">
          <span>graph {result.graph_score?.toFixed(3) ?? 'n/a'}</span>
          <span>semantic {result.semantic_score?.toFixed(3) ?? 'n/a'}</span>
        </div>
      </div>

      {result.best_scene?.text ? (
        <div className="mt-5 rounded-2xl border border-[var(--line)] bg-white/60 px-4 py-4">
          <p className="mb-2 text-xs font-semibold tracking-[0.14em] text-[var(--kicker)] uppercase">
            Best semantic scene
          </p>
          <p className="m-0 text-sm leading-7 text-[var(--sea-ink-soft)]">
            {result.best_scene.text}
          </p>
        </div>
      ) : null}

      <div className="mt-5 grid gap-3">
        {result.graph_matches.map((match, index) => (
          <div
            key={`${result.movie_id}-${index}`}
            className="rounded-2xl border border-[rgba(50,143,151,0.22)] bg-[rgba(79,184,178,0.1)] px-4 py-4"
          >
            <div className="flex flex-wrap items-center gap-2">
              {match.path_entities.map((entity, entityIndex) => (
                <div key={`${entity}-${entityIndex}`} className="flex items-center gap-2">
                  <span className="rounded-full bg-white/75 px-3 py-1 text-xs font-semibold text-[var(--sea-ink)]">
                    {entity}
                  </span>
                  {entityIndex < match.predicates.length ? (
                    <span className="text-xs font-semibold tracking-[0.12em] text-[var(--lagoon-deep)] uppercase">
                      {match.predicates[entityIndex]}
                    </span>
                  ) : null}
                </div>
              ))}
            </div>
            <div className="mt-3 grid gap-2">
              {match.evidences.map((evidence, evidenceIndex) => (
                <p key={`${evidenceIndex}-${evidence}`} className="m-0 text-sm leading-7 text-[var(--sea-ink-soft)]">
                  "{evidence}"
                </p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </article>
  )
}
