import { useState, type ReactNode } from 'react'
import { MODEL_INFO, type InspectResult, type ModelKey, type Product } from '../api'

export function ModelBadge({ model }: { model: ModelKey }) {
  const color = {
    glass: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-500/15 dark:text-indigo-300',
    csad: 'bg-amber-100 text-amber-800 dark:bg-amber-500/15 dark:text-amber-300',
    patchcore: 'bg-sky-100 text-sky-800 dark:bg-sky-500/15 dark:text-sky-300',
  }[model]
  return <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${color}`}>{MODEL_INFO[model].label}</span>
}

export function KindBadge({ kind }: { kind: Product['kind'] }) {
  const label = { object: 'Object', texture: 'Texture', logical: 'Logical constraints', custom: 'Custom' }[kind]
  return <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">{label}</span>
}

export function Verdict({ v, size = 'sm' }: { v: 'OK' | 'NOK'; size?: 'sm' | 'lg' }) {
  const base = size === 'lg' ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-xs'
  return v === 'OK'
    ? <span className={`rounded-md bg-emerald-100 font-bold text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-300 ${base}`}>OK</span>
    : <span className={`rounded-md bg-rose-100 font-bold text-rose-800 dark:bg-rose-500/15 dark:text-rose-300 ${base}`}>NOK</span>
}

/** Normalised score (score / threshold) on a 0-2 scale, with the threshold at the middle. */
export function ScoreBar({ value }: { value: number }) {
  const pct = Math.min(100, (value / 2) * 100)
  return (
    <div className="relative h-2 w-full rounded-full bg-slate-200 dark:bg-slate-800" title={`${value.toFixed(2)} × threshold`}>
      <div className={`h-2 rounded-full ${value >= 1 ? 'bg-rose-500' : 'bg-emerald-500'}`} style={{ width: `${pct}%` }} />
      <div className="absolute top-[-3px] h-3.5 w-0.5 bg-slate-700 dark:bg-slate-300" style={{ left: '50%' }} />
    </div>
  )
}

export function Kpi({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: 'good' | 'bad' | 'warn' }) {
  const color = tone === 'good' ? 'text-emerald-600 dark:text-emerald-400' : tone === 'bad' ? 'text-rose-600 dark:text-rose-400'
    : tone === 'warn' ? 'text-amber-600 dark:text-amber-400' : ''
  return (
    <div className="card p-4">
      <div className="label">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${color}`}>{value}</div>
      {hint && <div className="mt-1 text-xs muted">{hint}</div>}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-teal-600" />
      {label}
    </div>
  )
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null
  return <div className="rounded-lg border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800 dark:border-rose-800 dark:bg-rose-950 dark:text-rose-200">{String(error instanceof Error ? error.message : error)}</div>
}

export function ResultCard({ r }: { r: InspectResult }) {
  const tabs = [['view', 'Image'], ...(r.heatmap ? [['heatmap', 'Heatmap'], ['segmentation', 'Segmentation']] : [])] as const
  const [tab, setTab] = useState<string>(r.heatmap ? 'heatmap' : 'view')
  const src = tab === 'heatmap' ? r.heatmap : tab === 'segmentation' ? r.segmentation : r.view
  const wrong = r.truth && r.truth !== r.verdict
  return (
    <div className={`card overflow-hidden ${r.verdict === 'NOK' ? 'ring-1 ring-rose-400/60' : ''}`}>
      <div className="relative aspect-square bg-slate-100 dark:bg-slate-800">
        <img src={src} alt={r.name} className="h-full w-full object-contain" />
        <div className="absolute left-2 top-2"><Verdict v={r.verdict} /></div>
      </div>
      {tabs.length > 1 && (
        <div className="flex border-b border-slate-200 text-xs dark:border-slate-800">
          {tabs.map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`flex-1 py-1.5 ${tab === k ? 'bg-slate-100 font-semibold dark:bg-slate-800' : 'muted hover:bg-slate-50 dark:hover:bg-slate-800/50'}`}>
              {label}
            </button>
          ))}
        </div>
      )}
      <div className="space-y-2 p-3">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-xs muted" title={r.name}>{r.name}</span>
          <span className="font-mono text-xs tabular-nums">{r.normalized.toFixed(2)}×</span>
        </div>
        <ScoreBar value={r.normalized} />
        {r.truth && (
          <div className={`text-xs ${wrong ? 'text-rose-600 dark:text-rose-400' : 'muted'}`}>
            Ground truth: {r.truth}{r.defect_type && r.defect_type !== 'good' ? ` (${r.defect_type.replaceAll('_', ' ')})` : ''}
            {wrong ? ', misclassified' : ''}
          </div>
        )}
      </div>
    </div>
  )
}
