import { useState } from 'react'
import { Link } from 'react-router-dom'
import { DEPLOYED_KEY, type Product } from '../api'
import { ErrorBox, KindBadge, ModelBadge, Spinner } from '../components/ui'
import { fmt, useProducts } from '../hooks'

const FILTERS = [
  ['all', 'All'], ['object', 'Objects'], ['texture', 'Textures'], ['logical', 'Logical constraints'], ['custom', 'Custom'],
] as const

function thumb(p: Product) {
  return (p.samples.find((s) => !s.label) ?? p.samples[0])?.url
}

export default function Products() {
  const { data, error, loading } = useProducts()
  const [filter, setFilter] = useState<string>('all')
  const shown = (data ?? []).filter((p) => filter === 'all' || p.kind === filter)
  const counts = (data ?? []).reduce<Record<string, number>>((a, p) => ({ ...a, [p.model]: (a[p.model] ?? 0) + 1 }), {})
  const customCount = (data ?? []).filter((p) => p.custom).length

  return (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:items-end">
        <div>
          <p className="text-sm font-medium text-teal-700 dark:text-teal-400">Visual quality inspection</p>
          <h1 className="mt-1 text-3xl font-bold sm:text-4xl">Every product gets the model that inspects it best</h1>
          <p className="mt-3 max-w-2xl muted">
            AnoDe benchmarks unsupervised anomaly detectors on MVTec AD and MVTec LOCO AD and deploys, per product family,
            the best model whose threshold can be set honestly: PatchCore for surface and structural defects, and CSAD for
            logical anomalies such as missing or misplaced parts. Inspect a batch, review the heatmaps, and track the defect
            rate on a P-chart.
          </p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Link to="/inspect" className="btn-primary">Start an inspection</Link>
            <Link to="/benchmark" className="btn-ghost">See the benchmark</Link>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3 text-center">
          {(['patchcore', 'csad'] as const).map((m) => (
            <div key={m} className="card p-4">
              <div className="text-2xl font-semibold tabular-nums">{counts[m] ?? 0}</div>
              <div className="mt-1"><ModelBadge model={m} /></div>
              <div className="mt-1 text-xs muted">products</div>
            </div>
          ))}
          <div className="card p-4">
            <div className="text-2xl font-semibold tabular-nums">{customCount}</div>
            <div className="mt-1 text-xs font-semibold">Custom</div>
            <div className="mt-1 text-xs muted">in this browser</div>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-4 flex flex-wrap items-center gap-2">
          {FILTERS.map(([k, label]) => (
            <button key={k} onClick={() => setFilter(k)}
              className={`rounded-full px-3 py-1 text-sm ${filter === k ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900'
                : 'bg-white text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-300 dark:ring-slate-800'}`}>
              {label}
            </button>
          ))}
          <Link to="/new" className="ml-auto text-sm font-medium text-teal-700 hover:underline dark:text-teal-400">+ Add your own product</Link>
        </div>
        {loading && <Spinner label="Loading products" />}
        <ErrorBox error={error} />
        {!loading && filter === 'custom' && !shown.length && (
          <div className="card p-8 text-center muted">
            No custom products in this browser yet. <Link to="/new" className="text-teal-700 underline dark:text-teal-400">Create one</Link> from 10+ photos of good parts.
          </div>
        )}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {shown.map((p) => {
            const auroc = p.benchmark[DEPLOYED_KEY[p.model]]
            return (
              <Link key={p.id} to={`/products/${p.id}`} className="card group overflow-hidden transition hover:-translate-y-0.5 hover:shadow-md">
                <div className="aspect-square overflow-hidden bg-slate-100 dark:bg-slate-800">
                  {thumb(p)
                    ? <img src={thumb(p)} alt="" loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" />
                    : <div className="flex h-full items-center justify-center text-4xl muted">⌖</div>}
                </div>
                <div className="space-y-2 p-3">
                  <div className="font-medium">{p.name}</div>
                  <div className="flex flex-wrap gap-1.5"><ModelBadge model={p.model} /><KindBadge kind={p.kind} /></div>
                  <div className="text-xs muted">
                    {p.custom ? `Trained on ${p.n_train} images` : auroc != null ? `Image AUROC ${fmt.num(auroc)}` : ''}
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </section>
    </div>
  )
}
