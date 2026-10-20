import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, LIVE_MESSAGE, REPO_URL, STATIC, type Job } from '../api'
import { ErrorBox, Spinner } from '../components/ui'
import { invalidateProducts } from '../hooks'
import { store } from '../store'

export default function NewProduct() {
  const nav = useNavigate()
  const [name, setName] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [job, setJob] = useState<{ id: string; state: Job } | null>(null)
  const [error, setError] = useState<unknown>(null)

  useEffect(() => {
    const urls = files.slice(0, 24).map((f) => URL.createObjectURL(f))
    setPreviews(urls)
    return () => urls.forEach((u) => URL.revokeObjectURL(u))
  }, [files])

  useEffect(() => {
    if (!job || job.state.status === 'done' || job.state.status === 'error') return
    const t = setTimeout(async () => {
      try {
        const s = await api.job(job.id)
        setJob({ id: job.id, state: s })
        if (s.status === 'done' && s.product_id) {
          store.addCustomId(s.product_id)
          invalidateProducts()
          nav(`/products/${s.product_id}`)
        }
      } catch (e) { setError(e) }
    }, 1500)
    return () => clearTimeout(t)
  }, [job, nav])

  function add(list: FileList | File[]) {
    setFiles((prev) => [...prev, ...[...list].filter((f) => f.type.startsWith('image/'))].slice(0, 100))
  }

  async function submit() {
    setError(null)
    try {
      const { job_id } = await api.createCustom(name, files)
      setJob({ id: job_id, state: { status: 'queued', progress: 'uploading' } })
    } catch (e) { setError(e) }
  }

  const running = job && job.state.status !== 'error'
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">New product</h1>
        <p className="mt-1 text-sm muted">
          Upload 10–100 photos of <b>good</b> parts, taken from the same viewpoint and lighting as the line camera.
          AnoDe builds a PatchCore model from them on the server, holds out 20% to set the decision threshold, and
          adds the product to this browser.
        </p>
      </div>
      {STATIC && (
        <div className="rounded-lg bg-amber-50 p-4 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          {LIVE_MESSAGE} <a href={REPO_URL} className="underline" target="_blank" rel="noreferrer">How to run it locally</a>.
        </div>
      )}
      <div className="card space-y-5 p-5">
        <div>
          <label className="label" htmlFor="n">Product name</label>
          <input id="n" className="input" value={name} maxLength={60} placeholder="e.g. M6 hex bolt" onChange={(e) => setName(e.target.value)} />
        </div>
        <label className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed border-slate-300 p-8 text-center hover:border-teal-600 dark:border-slate-700"
          onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); add(e.dataTransfer.files) }}>
          <span className="font-medium">Drop normal images here or click to choose</span>
          <span className="text-xs muted">{files.length ? `${files.length} selected` : 'PNG or JPEG'}</span>
          <input type="file" accept="image/*" multiple className="hidden" onChange={(e) => e.target.files && add(e.target.files)} />
        </label>
        {previews.length > 0 && (
          <div className="grid grid-cols-6 gap-2 sm:grid-cols-8">
            {previews.map((u) => <img key={u} src={u} alt="" className="aspect-square rounded-md object-cover" />)}
            {files.length > previews.length && <div className="flex aspect-square items-center justify-center rounded-md bg-slate-100 text-xs muted dark:bg-slate-800">+{files.length - previews.length}</div>}
          </div>
        )}
        <ErrorBox error={error || (job?.state.status === 'error' ? job.state.progress : null)} />
        {running ? <Spinner label={`${job!.state.progress}…`} /> : (
          <div className="flex items-center justify-between gap-3">
            <button className="btn-ghost" onClick={() => setFiles([])} disabled={!files.length}>Clear</button>
            <button className="btn-primary" onClick={submit} disabled={STATIC || files.length < 10 || !name.trim()}>Build the model</button>
          </div>
        )}
      </div>
      <p className="text-xs muted">
        Building takes a few minutes on the free CPU. The model stays on the server only until it restarts, and only
        this browser can see it. Uploaded photos are not kept, only the extracted features.
      </p>
    </div>
  )
}
