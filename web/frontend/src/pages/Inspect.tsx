import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, LIVE_MESSAGE, MODEL_INFO, REPO_URL, STATIC, type Batch, type InspectResult, type Product } from '../api'
import { ErrorBox, Kpi, ModelBadge, ResultCard, Spinner } from '../components/ui'
import { fmt, useProducts } from '../hooks'
import { store } from '../store'

type Source = 'upload' | 'camera' | 'line'

function defaultBatchNo() {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `B${String(d.getFullYear()).slice(2)}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`
}

async function thumbnail(src: string, size = 160): Promise<string> {
  const img = new Image()
  img.src = src
  await img.decode()
  const c = document.createElement('canvas')
  const s = size / Math.max(img.width, img.height)
  c.width = Math.round(img.width * s)
  c.height = Math.round(img.height * s)
  c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.7)
}

function ProductPicker({ products }: { products: Product[] }) {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Start an inspection</h1>
      <p className="muted">Choose the product on the line. Each one has its own model and threshold.</p>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {products.map((p) => (
          <Link key={p.id} to={`/inspect/${p.id}`} className="card flex items-center justify-between gap-2 px-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-800">
            <span className="font-medium">{p.name}</span><ModelBadge model={p.model} />
          </Link>
        ))}
      </div>
    </div>
  )
}

function Camera({ busy, onCapture }: { busy: boolean; onCapture: (b: Blob) => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [auto, setAuto] = useState(false)

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream])

  async function start() {
    setError(null)
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } } })
      setStream(s)
      if (video.current) video.current.srcObject = s
    } catch (e) { setError(e) }
  }

  function capture() {
    const v = video.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    // PNG, not JPEG: compression artefacts move CSAD's score by more than the threshold
    c.toBlob((b) => b && onCapture(b), 'image/png')
  }

  useEffect(() => {
    if (!auto || busy || !stream) return
    const t = setTimeout(capture, 2500)
    return () => clearTimeout(t)
  })

  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-lg bg-slate-900">
        <video ref={video} autoPlay playsInline muted className="mx-auto max-h-80" />
      </div>
      <ErrorBox error={error} />
      <div className="flex flex-wrap items-center gap-3">
        {!stream ? <button className="btn-primary" onClick={start}>Connect camera</button> : <>
          <button className="btn-primary" onClick={capture} disabled={busy}>Capture & inspect</button>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />Capture automatically every 2.5 s</label>
        </>}
      </div>
      <p className="text-xs muted">The models were trained on MVTec images, so a webcam photo of a different object will usually be flagged. Use a custom product for your own parts.</p>
    </div>
  )
}

export default function Inspect() {
  const { id } = useParams()
  const nav = useNavigate()
  const { data: products, error, loading } = useProducts()
  const [date, setDate] = useState(fmt.today())
  const [batchNo, setBatchNo] = useState(defaultBatchNo())
  const [source, setSource] = useState<Source>('line')
  const [results, setResults] = useState<InspectResult[]>([])
  const [busy, setBusy] = useState(false)
  const [runError, setRunError] = useState<unknown>(null)
  const [lineOn, setLineOn] = useState(false)
  const lineRef = useRef(false)
  const product = products?.find((p) => p.id === id)

  useEffect(() => {
    if (product?.custom) setSource('upload')
  }, [product?.custom])

  // Simulated line: inspect the bundled samples one at a time, in random order
  useEffect(() => {
    lineRef.current = lineOn
    if (!lineOn || !product) return
    let cancelled = false
    ;(async () => {
      while (lineRef.current && !cancelled) {
        const i = Math.floor(Math.random() * product.samples.length)
        try {
          setBusy(true)
          const r = await api.inspectSamples(product.id, [i])
          if (!cancelled) setResults((prev) => [{ ...r[0], name: `unit-${prev.length + 1}` }, ...prev])
        } catch (e) { setRunError(e); setLineOn(false) } finally { setBusy(false) }
        await new Promise((res) => setTimeout(res, 700))
      }
    })()
    return () => { cancelled = true }
  }, [lineOn, product])

  if (loading) return <Spinner label="Loading products" />
  if (error) return <ErrorBox error={error} />
  if (!id || !product) return <ProductPicker products={products ?? []} />

  async function inspectBlobs(blobs: Blob[], names: string[]) {
    setBusy(true); setRunError(null)
    try {
      for (let i = 0; i < blobs.length; i += 6) {
        const r = await api.inspectFiles(product!.id, blobs.slice(i, i + 6), names.slice(i, i + 6))
        setResults((prev) => [...r.reverse(), ...prev])
      }
    } catch (e) { setRunError(e) } finally { setBusy(false) }
  }

  async function finish() {
    setLineOn(false)
    const nok = results.filter((r) => r.verdict === 'NOK')
    const defects = await Promise.all(nok.slice(0, 8).map(async (r) => ({
      name: r.name, normalized: r.normalized, thumb: await thumbnail(r.heatmap ?? r.view),
    })))
    const b: Batch = {
      id: `live-${Date.now()}`, product_id: product!.id, date, batch_no: batchNo.trim() || defaultBatchNo(),
      n: results.length, defective: nok.length, scores: results.map((r) => Math.round(r.normalized * 1e4) / 1e4), defects,
    }
    store.saveBatch(b)
    nav(`/reports/${product!.id}?batch=${b.id}`)
  }

  const nok = results.filter((r) => r.verdict === 'NOK').length
  const latency = results.length ? results.reduce((a, r) => a + r.latency_ms, 0) / results.length : 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link to={`/products/${product.id}`} className="text-sm muted hover:underline">← {product.name}</Link>
          <h1 className="mt-1 text-2xl font-bold">Inspection: {product.name}</h1>
          <p className="text-sm muted">{MODEL_INFO[product.model].label} · threshold {product.threshold.toPrecision(4)}{!product.has_map && ' · image-level decision (no heatmap)'}</p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div><label className="label" htmlFor="d">Production date</label><input id="d" type="date" className="input" value={date} onChange={(e) => setDate(e.target.value)} /></div>
          <div><label className="label" htmlFor="b">Batch number</label><input id="b" className="input font-mono" value={batchNo} onChange={(e) => setBatchNo(e.target.value)} /></div>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-4">
        <Kpi label="Inspected" value={results.length} />
        <Kpi label="OK" value={results.length - nok} tone="good" />
        <Kpi label="NOK" value={nok} tone={nok ? 'bad' : undefined} />
        <Kpi label="Pass rate" value={results.length ? fmt.pct((results.length - nok) / results.length) : '–'} hint={latency ? `${latency.toFixed(0)} ms per image on CPU` : undefined} />
      </div>

      <section className="card p-5">
        <div className="mb-4 flex gap-1 border-b border-slate-200 dark:border-slate-800">
          {([['line', 'Simulated line'], ['upload', 'Upload images'], ['camera', 'Camera']] as const).map(([k, label]) => (
            <button key={k} onClick={() => setSource(k)} disabled={k === 'line' && product.custom}
              className={`-mb-px border-b-2 px-3 py-2 text-sm disabled:opacity-40 ${source === k ? 'border-teal-600 font-medium' : 'border-transparent muted'}`}>
              {label}
            </button>
          ))}
        </div>
        {source === 'line' && (
          <div className="flex flex-wrap items-center gap-4">
            <button className={lineOn ? 'btn-ghost' : 'btn-primary'} onClick={() => setLineOn(!lineOn)}>{lineOn ? 'Pause the line' : 'Start the line'}</button>
            <p className="text-sm muted">Feeds this product's sample images (normal and defective) to the model one at a time, like parts passing a line camera.</p>
          </div>
        )}
        {STATIC && source !== 'line' && (
          <div className="rounded-lg bg-amber-50 p-4 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-200">
            {LIVE_MESSAGE} <a href={REPO_URL} className="underline" target="_blank" rel="noreferrer">How to run it locally</a>.
          </div>
        )}
        {!STATIC && source === 'upload' && (
          <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-slate-300 p-8 text-center hover:border-teal-600 dark:border-slate-700"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = [...e.dataTransfer.files].filter((x) => x.type.startsWith('image/')); inspectBlobs(f, f.map((x) => x.name)) }}>
            <span className="font-medium">Drop images here or click to choose</span>
            <span className="text-xs muted">PNG or JPEG, up to 64 per upload{product.model === 'csad' && '. Use PNG: CSAD is very sensitive to JPEG compression'}</span>
            <input type="file" accept="image/*" multiple className="hidden" disabled={busy}
              onChange={(e) => { const f = [...(e.target.files ?? [])]; inspectBlobs(f, f.map((x) => x.name)); e.target.value = '' }} />
          </label>
        )}
        {!STATIC && source === 'camera' && <Camera busy={busy} onCapture={(b) => inspectBlobs([b], [`camera-${results.length + 1}.png`])} />}
        <div className="mt-4"><ErrorBox error={runError} /></div>
        {busy && <div className="mt-3"><Spinner label="Inspecting" /></div>}
      </section>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Results {results.length > 0 && <span className="text-sm font-normal muted">(newest first)</span>}</h2>
        <div className="flex gap-2">
          {results.length > 0 && <button className="btn-ghost" onClick={() => setResults([])}>Clear</button>}
          <button className="btn-primary" onClick={finish} disabled={!results.length || busy}>Finish batch</button>
        </div>
      </div>
      {results.length ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">{results.map((r, i) => <ResultCard key={`${r.name}-${results.length - i}`} r={r} />)}</div>
      ) : (
        <div className="card p-8 text-center text-sm muted">No units inspected yet in this batch.</div>
      )}
    </div>
  )
}
