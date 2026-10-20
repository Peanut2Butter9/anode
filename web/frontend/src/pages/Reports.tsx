import { Fragment, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, type Batch, type Capability } from '../api'
import { PChart, ScoreHistogram } from '../components/charts'
import { ErrorBox, Kpi, Spinner } from '../components/ui'
import { fmt, useAsync, useProducts } from '../hooks'
import { store } from '../store'

function cpkTone(c?: Capability | null) {
  if (!c) return undefined
  return c.rating === 'capable' ? 'good' : c.rating === 'marginal' ? 'warn' : 'bad'
}

export default function Reports() {
  const { id } = useParams()
  const [params] = useSearchParams()
  const nav = useNavigate()
  const highlight = params.get('batch')
  const { data: products, loading: pl } = useProducts()
  const [includeDemo, setIncludeDemo] = useState(true)
  const [open, setOpen] = useState<string | null>(highlight)
  const [version, setVersion] = useState(0)

  const product = products?.find((p) => p.id === id) ?? products?.[0]
  const pid = product?.id ?? ''
  const demo = useAsync(() => (product && !product.custom ? api.demoHistory(pid) : Promise.resolve([] as Batch[])), [pid])
  const local = useMemo(() => (pid ? store.batches(pid) : []), [pid, version])
  const batches = useMemo(
    () => [...(includeDemo ? demo.data ?? [] : []), ...local].sort((a, b) => a.date.localeCompare(b.date) || a.id.localeCompare(b.id)),
    [includeDemo, demo.data, local])
  const key = batches.map((b) => b.id).join('|')
  const report = useAsync(() => (batches.length ? api.qc(batches) : Promise.resolve(null)), [key])

  if (pl || !product) return <Spinner label="Loading" />

  const units = batches.reduce((a, b) => a + b.n, 0)
  const ooc = report.data?.p_chart.points.filter((p) => p.out_of_control).length ?? 0
  const allScores = batches.flatMap((b) => b.scores)
  const cap = report.data?.capability

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Quality report</h1>
          <p className="text-sm muted">Defect rate per batch (P-chart) and process capability of the anomaly score.</p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="label" htmlFor="p">Product</label>
            <select id="p" className="input" value={pid} onChange={(e) => nav(`/reports/${e.target.value}`)}>
              {products!.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </div>
          {!product.custom && (
            <label className="flex items-center gap-2 pb-2 text-sm">
              <input type="checkbox" checked={includeDemo} onChange={(e) => setIncludeDemo(e.target.checked)} />
              Include simulated history
            </label>
          )}
          <Link to={`/inspect/${pid}`} className="btn-primary">New batch</Link>
        </div>
      </div>

      {includeDemo && !product.custom && (
        <p className="rounded-lg bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          Simulated batches draw real {product.name.toLowerCase()} test-set scores from the deployed model at a 2–5% defect
          rate. One batch has a raised rate (an assignable cause) to show an out-of-control point. Batches you inspect
          are saved in this browser and appear here too.
        </p>
      )}

      {demo.loading || report.loading ? <Spinner label="Computing control limits" /> : null}
      <ErrorBox error={demo.error || report.error} />

      {!batches.length && !demo.loading && (
        <div className="card p-8 text-center text-sm muted">No batches yet. <Link className="underline" to={`/inspect/${pid}`}>Inspect one</Link>.</div>
      )}

      {report.data && batches.length > 0 && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <Kpi label="Batches" value={batches.length} hint={`${local.length} inspected here`} />
            <Kpi label="Units inspected" value={units.toLocaleString()} />
            <Kpi label="Mean defect rate p̄" value={fmt.pct(report.data.p_chart.p_bar, 2)} />
            <Kpi label="Out-of-control batches" value={ooc} tone={ooc ? 'bad' : 'good'} hint="outside the 3σ limits" />
            <Kpi label="Cpk of the score" value={cap ? cap.cpk.toFixed(2) : '–'} tone={cpkTone(cap)}
              hint={cap ? `${cap.rating} · ${cap.cpk_from_yield != null ? `${cap.cpk_from_yield.toFixed(2)} from yield` : 'no units above USL'}` : 'needs scores'} />
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
            <section className="card p-5">
              <h2 className="font-semibold">P-chart</h2>
              <p className="mb-2 text-xs muted">p̄ ± 3·√(p̄(1−p̄)/n). The limits widen for smaller batches.</p>
              <PChart batches={batches} report={report.data} />
            </section>
            <section className="card p-5">
              <h2 className="font-semibold">Score distribution</h2>
              <p className="mb-2 text-xs muted">All units, scores as a multiple of the threshold (USL = 1). Cpk = (USL − μ) / 3σ.</p>
              <ScoreHistogram scores={allScores} />
              {cap && (
                <p className="mt-2 text-xs muted">
                  μ = {cap.mean.toFixed(3)}, σ = {cap.std.toFixed(3)}, {fmt.pct(cap.fraction_above_usl)} above USL.
                  Good and defective parts make the distribution bimodal, which inflates σ. The yield-based index
                  Φ⁻¹(1 − p)/3 = {cap.cpk_from_yield != null ? cap.cpk_from_yield.toFixed(2) : '∞'} is the more robust reading.
                </p>
              )}
            </section>
          </div>

          <section className="card overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide muted dark:border-slate-800">
                <tr>{['Date', 'Batch', 'Units', 'NOK', 'Pass rate', 'Cpk', 'Control', 'Source', ''].map((h) => <th key={h} className="px-4 py-3 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody>
                {batches.map((b, i) => {
                  const pt = report.data!.p_chart.points[i]
                  const c = report.data!.capability_per_batch[i]
                  return (
                    <Fragment key={b.id}>
                      <tr className={`border-b border-slate-100 dark:border-slate-800/60 ${b.id === highlight ? 'bg-teal-50 dark:bg-teal-950/40' : ''}`}>
                        <td className="px-4 py-2 tabular-nums">{b.date}</td>
                        <td className="px-4 py-2 font-mono text-xs">{b.batch_no}</td>
                        <td className="px-4 py-2 tabular-nums">{b.n}</td>
                        <td className="px-4 py-2 tabular-nums">{b.defective}</td>
                        <td className="px-4 py-2 tabular-nums">{fmt.pct(1 - b.defective / b.n)}</td>
                        <td className="px-4 py-2 tabular-nums">{c ? c.cpk.toFixed(2) : '–'}</td>
                        <td className="px-4 py-2">{pt.out_of_control
                          ? <span className="font-semibold text-rose-600 dark:text-rose-400">Out of control</span>
                          : <span className="muted">In control</span>}</td>
                        <td className="px-4 py-2 text-xs muted">{b.demo ? 'simulated' : 'inspected'}</td>
                        <td className="px-4 py-2 text-right">
                          {!b.demo && (
                            <span className="flex justify-end gap-3 text-xs">
                              {b.defects?.length ? <button className="underline" onClick={() => setOpen(open === b.id ? null : b.id)}>Defect detail</button> : null}
                              <button className="muted underline" onClick={() => { store.deleteBatch(b.id); setVersion((v) => v + 1) }}>Delete</button>
                            </span>
                          )}
                        </td>
                      </tr>
                      {open === b.id && b.defects?.length ? (
                        <tr><td colSpan={9} className="bg-slate-50 px-4 py-3 dark:bg-slate-900/60">
                          <div className="flex flex-wrap gap-3">
                            {b.defects.map((d) => (
                              <figure key={d.name} className="w-28">
                                <img src={d.thumb} alt={d.name} className="rounded-md" />
                                <figcaption className="mt-1 truncate text-xs muted">{d.name} · {d.normalized.toFixed(2)}×</figcaption>
                              </figure>
                            ))}
                          </div>
                        </td></tr>
                      ) : null}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  )
}
