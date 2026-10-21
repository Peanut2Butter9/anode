import { useState } from 'react'
import { api, type Metrics } from '../api'
import { ErrorBox, Spinner } from '../components/ui'
import { fmt, useAsync } from '../hooks'

const METRICS: Record<string, { label: string; keys: Record<string, keyof Metrics>; lowerBetter?: boolean }> = {
  auroc: { label: 'Image AUROC', keys: { mvtec_ad: 'image_auroc', mvtec_loco: 'auroc_loco_mean' } },
  logical: { label: 'Logical AUROC', keys: { mvtec_loco: 'auroc_logical' } },
  structural: { label: 'Structural AUROC', keys: { mvtec_loco: 'auroc_structural' } },
  pixel: { label: 'Pixel AUROC', keys: { mvtec_ad: 'pixel_auroc' } },
  aupro: { label: 'AUPRO (FPR ≤ 30%)', keys: { mvtec_ad: 'aupro_30' } },
  f1: { label: 'F1 at validation threshold', keys: { mvtec_ad: 'val_f1', mvtec_loco: 'val_f1' } },
  latency: { label: 'ms / image (T4)', keys: { mvtec_ad: 'ms_per_image', mvtec_loco: 'ms_per_image' }, lowerBetter: true },
}
const NAMES: Record<string, string> = { mvtec_ad: 'MVTec AD', mvtec_loco: 'MVTec LOCO AD' }

function cellColor(v: number | null | undefined, lowerBetter?: boolean) {
  if (v == null || lowerBetter) return ''
  if (v >= 0.98) return 'bg-emerald-100 dark:bg-emerald-500/20'
  if (v >= 0.95) return 'bg-emerald-50 dark:bg-emerald-500/10'
  if (v >= 0.85) return ''
  if (v >= 0.7) return 'bg-amber-50 dark:bg-amber-500/10'
  return 'bg-rose-50 dark:bg-rose-500/10'
}

export default function BenchmarkPage() {
  const { data, error, loading } = useAsync(() => api.benchmark(), [])
  const [ds, setDs] = useState('mvtec_ad')
  const [metric, setMetric] = useState('auroc')
  if (loading) return <Spinner label="Loading benchmark" />
  if (error || !data) return <ErrorBox error={error} />

  const d = data.datasets[ds]
  const m = METRICS[metric]
  const key = m.keys[ds]
  const available = Object.entries(METRICS).filter(([, x]) => x.keys[ds])
  const models = d.models.filter((x) => d.categories.some((c) => d.per_category[x]?.[c]?.[key] != null))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Benchmark</h1>
        <p className="mt-1 max-w-3xl text-sm muted">
          Six methods (seven configurations), with the same split and preprocessing for all of them, run on Kaggle T4 GPUs. Seeded models show the
          mean over 3 seeds. Thresholds for F1 come from held-out normal images, never from test labels.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {Object.keys(data.datasets).map((k) => (
          <button key={k} onClick={() => { setDs(k); if (!METRICS[metric].keys[k]) setMetric('auroc') }}
            className={ds === k ? 'btn-primary' : 'btn-ghost'}>{NAMES[k] ?? k}</button>
        ))}
        <select className="input w-auto" value={metric} onChange={(e) => setMetric(e.target.value)}>
          {available.map(([k, x]) => <option key={k} value={k}>{x.label}</option>)}
        </select>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead className="border-b border-slate-200 text-xs dark:border-slate-800">
            <tr>
              <th className="px-3 py-3 text-left font-medium uppercase tracking-wide muted">Category</th>
              {models.map((x) => (
                <th key={x} className={`px-3 py-3 text-right font-medium ${x === d.deployed ? 'text-teal-700 dark:text-teal-400' : ''}`}>
                  {data.models[x] ?? x}{x === d.deployed && <div className="text-[10px] font-normal uppercase">deployed</div>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...d.categories, 'mean'].map((c) => (
              <tr key={c} className={`border-b border-slate-100 dark:border-slate-800/60 ${c === 'mean' ? 'font-semibold' : ''}`}>
                <td className="px-3 py-2">{c === 'mean' ? 'Mean' : c.replaceAll('_', ' ')}</td>
                {models.map((x) => {
                  const v = c === 'mean' ? d.mean[x]?.[key] : d.per_category[x]?.[c]?.[key]
                  return (
                    <td key={x} className={`px-3 py-2 text-right font-mono tabular-nums ${cellColor(v, m.lowerBetter)}`}>
                      {metric === 'latency' ? fmt.num(v, 1) : fmt.num(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="grid gap-4 text-sm lg:grid-cols-2">
        <div className="card space-y-2 p-5">
          <h2 className="font-semibold">Reading the table</h2>
          <ul className="list-disc space-y-1.5 pl-5 muted">
            <li><b>Released</b> GLASS and CSAD checkpoints come from their authors. Both official training scripts keep the epoch with the best <i>test</i> score (GLASS's metal nut checkpoint is from epoch 5 of 640), so these columns are optimistic upper bounds.</li>
            <li><b>GLASS and CSAD (our training)</b> use the official code with a shortened schedule (GLASS 50 of 640 epochs, CSAD 12,000 of 60,000 iterations) and report the last checkpoint. The "test-selected" columns pick the best checkpoint on the test set, only to show how much that choice inflates the score.</li>
            <li>A blank mean means the model was not run on every category.</li>
            <li><b>Deployed</b> marks the model the website uses. For MVTec AD that is PatchCore, not released GLASS: an operating threshold has to come from normal images the model never saw, and the released GLASS checkpoints leave none.</li>
          </ul>
        </div>
        <div className="card space-y-2 p-5">
          <h2 className="font-semibold">What changed from the original report</h2>
          <ul className="list-disc space-y-1.5 pl-5 muted">
            <li>The original "Res2Net" model was a feature autoencoder on a ResNet-50 backbone. We run it with both backbones, and ResNet-50 is better (0.980 vs 0.901).</li>
            <li>Its accuracy and F1 used thresholds tuned on test labels. Here they come from normal validation images only.</li>
            <li>Every model now runs on both datasets, so LOCO has a real comparison: CSAD's lead comes from logical anomalies.</li>
          </ul>
        </div>
      </section>
    </div>
  )
}
