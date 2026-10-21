import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, DEPLOYED_KEY, MODEL_INFO, type InspectResult } from '../api'
import { ErrorBox, KindBadge, ModelBadge, ResultCard, Spinner } from '../components/ui'
import { fmt, useAsync } from '../hooks'

const MODEL_NAMES: Record<string, string> = {
  autoencoder_l2: 'Autoencoder (L2)', resnet50_knn: 'ResNet-50 + kNN', feature_ae_resnet50: 'Feature AE (ResNet-50)',
  feature_ae_res2net50: 'Feature AE (Res2Net-50)', patchcore: 'PatchCore', glass: 'GLASS (our training, 50 ep.)',
  glass_testselected: 'GLASS (ours, test-selected)', glass_official: 'GLASS (released)',
  csad: 'CSAD (our training, 12k it.)', csad_testselected: 'CSAD (ours, test-selected)', csad_official: 'CSAD (released)',
}

export default function ProductPage() {
  const { id = '' } = useParams()
  const { data: p, error, loading } = useAsync(() => api.product(id), [id])
  const [results, setResults] = useState<InspectResult[] | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<unknown>(null)
  const [showMasks, setShowMasks] = useState(false)

  if (loading) return <Spinner label="Loading product" />
  if (error || !p) return <ErrorBox error={error ?? 'not found'} />

  const info = MODEL_INFO[p.model]
  const bars = Object.entries(p.benchmark).filter(([k, v]) => v != null && !k.endsWith('_testselected'))
    .sort((a, b) => (b[1] as number) - (a[1] as number))
  const deployed = DEPLOYED_KEY[p.model]
  const correct = results?.filter((r) => r.truth === r.verdict).length ?? 0

  async function runSamples() {
    setRunning(true); setRunError(null)
    try { setResults(await api.inspectSamples(p!.id)) } catch (e) { setRunError(e) } finally { setRunning(false) }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link to="/" className="text-sm muted hover:underline">← All products</Link>
          <h1 className="mt-1 text-3xl font-bold">{p.name}</h1>
          <div className="mt-2 flex flex-wrap gap-2"><ModelBadge model={p.model} /><KindBadge kind={p.kind} />
            <span className="text-xs muted">{p.dataset === 'mvtec_ad' ? 'MVTec AD' : p.dataset === 'mvtec_loco' ? 'MVTec LOCO AD' : 'Created in this browser'}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Link to={`/reports/${p.id}`} className="btn-ghost">Reports</Link>
          <Link to={`/inspect/${p.id}`} className="btn-primary">Inspect a batch</Link>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card space-y-4 p-5">
          <h2 className="text-lg font-semibold">Model: {info.label} <span className="text-sm font-normal muted">({info.venue})</span></h2>
          <p className="text-sm muted">{info.blurb}</p>
          <dl className="grid grid-cols-2 gap-3 text-sm">
            <div><dt className="label">Decision threshold</dt><dd className="font-mono">{p.threshold.toPrecision(4)}</dd></div>
            <div><dt className="label">Threshold rule</dt><dd>mean + 3σ of {p.val_scores.length} held-out normal images</dd></div>
          </dl>
          <p className="text-xs muted">
            Scores are shown as a multiple of the threshold, so 1.00× is the boundary between OK and NOK. The rule never
            looks at defective images.
            {p.model === 'patchcore' && !p.custom && ' The held-out images are the 10% of normal training images left out of the memory bank.'}
            {p.model === 'csad' && ' The held-out images are the official validation split, which CSAD never trains on.'}
          </p>
        </section>

        <section className="card p-5">
          <h2 className="text-lg font-semibold">Why this model?</h2>
          {bars.length ? (
            <>
              <p className="mt-1 text-sm muted">Image AUROC on the {p.name.toLowerCase()} test set, all models in the benchmark{p.dataset === 'mvtec_loco' ? ' (mean of logical and structural)' : ''}.</p>
              <div className="mt-4 space-y-2">
                {bars.map(([k, v]) => (
                  <div key={k} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3 text-sm">
                    <span className={`truncate ${k === deployed ? 'font-semibold' : 'muted'}`}>{MODEL_NAMES[k] ?? k}</span>
                    <div className="h-2.5 rounded-full bg-slate-100 dark:bg-slate-800">
                      <div className={`h-2.5 rounded-full ${k === deployed ? 'bg-teal-600' : 'bg-slate-400 dark:bg-slate-600'}`}
                        style={{ width: `${Math.max(0, ((v as number) - 0.5) / 0.5) * 100}%` }} />
                    </div>
                    <span className="text-right font-mono tabular-nums">{fmt.num(v as number)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs muted">
                Bars start at 0.5 (chance). "Released" checkpoints were chosen by their authors on the test set, so treat
                them as upper bounds.
                {p.dataset === 'mvtec_ad' && ' Released GLASS scores higher, but it was trained on every normal image and leaves none to set a threshold. On new normal images it flags about one in three good parts, so it stays in the benchmark only.'}
                {' '}See <Link to="/benchmark" className="underline">Benchmark</Link>.
              </p>
            </>
          ) : (
            <p className="mt-2 text-sm muted">
              Custom products use PatchCore: it needs only normal images and no training loop, so it can be built on the
              server in minutes. It reaches 0.982 image AUROC on MVTec AD in our benchmark. Trained on {p.n_train} images,
              calibrated on {p.n_val}.
            </p>
          )}
        </section>
      </div>

      {!p.custom && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">Try it on sample images</h2>
              <p className="text-sm muted">{p.samples.length} test images from the dataset, normal and defective. The model never saw them during training.</p>
            </div>
            <div className="flex items-center gap-3">
              {!results && p.samples.some((s) => s.mask_url) && (
                <label className="flex items-center gap-2 text-sm muted">
                  <input type="checkbox" checked={showMasks} onChange={(e) => setShowMasks(e.target.checked)} /> Show ground-truth masks
                </label>
              )}
              <button className="btn-primary" onClick={runSamples} disabled={running}>{running ? 'Running…' : results ? 'Run again' : 'Run the model'}</button>
            </div>
          </div>
          <ErrorBox error={runError} />
          {running && <Spinner label={`${info.label} is inspecting ${p.samples.length} images on the CPU`} />}
          {results ? (
            <>
              <div className="text-sm"><b>{correct}/{results.length}</b> classified correctly at the deployed threshold.</div>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">{results.map((r) => <ResultCard key={r.name} r={r} />)}</div>
            </>
          ) : (
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-6">
              {p.samples.map((s) => (
                <figure key={s.file} className="card overflow-hidden">
                  <div className="relative aspect-square bg-slate-100 dark:bg-slate-800">
                    <img src={s.url} alt={s.defect_type} loading="lazy" className="h-full w-full object-contain" />
                    {showMasks && s.mask_url && <img src={s.mask_url} alt="" className="absolute inset-0 h-full w-full object-contain opacity-50 mix-blend-screen" />}
                  </div>
                  <figcaption className="px-2 py-1.5 text-xs muted">{s.defect_type.replaceAll('_', ' ')}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
