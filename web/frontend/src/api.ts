// Types and calls for the AnoDe API (web/server/anode_web/app.py)

export type ModelKey = 'glass' | 'csad' | 'patchcore'
export type Dataset = 'mvtec_ad' | 'mvtec_loco' | 'custom'

export interface Sample {
  file: string
  url: string
  mask_url: string | null
  label: number
  defect_type: string
}

export interface Product {
  id: string
  name: string
  dataset: Dataset
  category: string
  kind: 'object' | 'texture' | 'logical' | 'custom'
  model: ModelKey
  threshold: number
  custom: boolean
  has_map: boolean
  samples: Sample[]
  benchmark: Record<string, number | null>
  val_scores: number[]
  n_train?: number
  n_val?: number
  created?: number
}

export interface InspectResult {
  name: string
  score: number
  normalized: number
  verdict: 'OK' | 'NOK'
  latency_ms: number
  view: string
  heatmap?: string
  segmentation?: string
  truth?: 'OK' | 'NOK'
  defect_type?: string
}

export interface Batch {
  id: string
  product_id: string
  date: string // YYYY-MM-DD
  batch_no: string
  n: number
  defective: number
  scores: number[] // normalised: score / threshold
  demo?: boolean
  defects?: { name: string; thumb: string; normalized: number }[]
}

export interface Capability {
  mean: number
  std: number
  usl: number
  cpu: number
  cpk: number
  cp: number | null
  fraction_above_usl: number
  cpk_from_yield: number | null
  rating: 'capable' | 'marginal' | 'not capable'
}

export interface QCReport {
  p_chart: { p_bar: number; points: { p: number; ucl: number; lcl: number; out_of_control: boolean }[] }
  capability: Capability | null
  capability_per_batch: (Capability | null)[]
}

export type Metrics = Partial<Record<
  'image_auroc' | 'auroc_logical' | 'auroc_structural' | 'auroc_loco_mean' | 'pixel_auroc' | 'aupro_30' | 'val_f1' |
  'f1_max' | 'ms_per_image', number | null>>

export interface Benchmark {
  models: Record<string, string>
  datasets: Record<string, {
    categories: string[]
    models: string[]
    per_category: Record<string, Record<string, Metrics>>
    mean: Record<string, Metrics>
    deployed: string
  }>
}

export interface Job {
  status: 'queued' | 'running' | 'done' | 'error'
  progress: string
  product_id?: string
  seconds?: number
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      if (body.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

const server = {
  products: (customIds: string[]) => call<Product[]>(`/api/products?custom=${customIds.join(',')}`),
  product: (id: string) => call<Product>(`/api/products/${encodeURIComponent(id)}`),
  benchmark: () => call<Benchmark>('/api/benchmark'),
  demoHistory: (id: string, days = 30) => call<Batch[]>(`/api/demo-history/${encodeURIComponent(id)}?days=${days}`),
  inspectFiles: (id: string, files: Blob[], names: string[]) => {
    const fd = new FormData()
    files.forEach((f, i) => fd.append('files', f, names[i]))
    return call<InspectResult[]>(`/api/inspect/${encodeURIComponent(id)}`, { method: 'POST', body: fd })
  },
  inspectSamples: (id: string, indices?: number[]) =>
    call<InspectResult[]>(`/api/inspect/${encodeURIComponent(id)}/samples`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(indices ?? null),
    }),
  createCustom: (name: string, files: File[]) => {
    const fd = new FormData()
    fd.append('name', name)
    files.forEach((f) => fd.append('files', f, f.name))
    return call<{ job_id: string }>('/api/products/custom', { method: 'POST', body: fd })
  },
  job: (id: string) => call<Job>(`/api/jobs/${id}`),
  qc: (batches: Pick<Batch, 'n' | 'defective' | 'scores'>[]) =>
    call<QCReport>('/api/qc', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batches: batches.map(({ n, defective, scores }) => ({ n, defective, scores })) }),
    }),
}

/** True in the static build (free hosting): no server, precomputed sample results. */
export const STATIC = import.meta.env.VITE_STATIC === '1'
export const REPO_URL = 'https://github.com/Peanut2Butter9/anode'
export const LIVE_MESSAGE = 'Live inference needs the AnoDe server, which this free static demo does not run. ' +
  'Sample images below use precomputed results; to inspect your own images, run AnoDe locally (see the README).'

const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`
const json = <T,>(path: string) => call<T>(asset(path))
let productsCache: Promise<Product[]> | null = null
const staticProducts = () => (productsCache ??= json<Product[]>('data/products.json'))

const staticApi: typeof server = {
  products: () => staticProducts(),
  product: async (id) => {
    const p = (await staticProducts()).find((x) => x.id === id)
    if (!p) throw new Error(`unknown product ${id}`)
    return p
  },
  benchmark: () => json<Benchmark>('data/benchmark.json'),
  demoHistory: async (id, days = 30) => {
    const [p, pools] = await Promise.all([staticApi.product(id),
      json<Record<string, { good: number[]; defective: number[] }>>('data/demo_scores.json')])
    const { demoHistory } = await import('./static')
    return pools[p.category] ? demoHistory(id, pools[p.category], p.threshold, days) : []
  },
  inspectFiles: async () => { throw new Error(LIVE_MESSAGE) },
  inspectSamples: async (id, indices) => {
    const all = await json<InspectResult[]>(`data/inspect/${id}.json`)
    const picked = indices ? indices.map((i) => all[i]).filter(Boolean) : all
    await new Promise((r) => setTimeout(r, 120 * picked.length))  // a little of the real CPU latency
    return picked.map((r) => ({
      ...r,
      view: asset(r.view), heatmap: r.heatmap && asset(r.heatmap), segmentation: r.segmentation && asset(r.segmentation),
    }))
  },
  createCustom: async () => { throw new Error(LIVE_MESSAGE) },
  job: async () => { throw new Error(LIVE_MESSAGE) },
  qc: async (batches) => (await import('./static')).qcReport(batches),
}

export const api = STATIC ? staticApi : server

// The benchmark key of the model each product family uses on the website
export const DEPLOYED_KEY: Record<ModelKey, string> = {
  glass: 'glass_official', csad: 'csad_official', patchcore: 'patchcore',
}

export const MODEL_INFO: Record<ModelKey, { label: string; venue: string; blurb: string }> = {
  glass: {
    label: 'GLASS', venue: 'ECCV 2024',
    blurb: 'Synthesises near-normal anomalies with gradient ascent and trains a patch discriminator. Best on structural defects.',
  },
  csad: {
    label: 'CSAD', venue: 'BMVC 2024',
    blurb: 'Segments each image into components and checks their counts and layout, which catches logical anomalies (missing or misplaced parts). Gives a score but no heatmap.',
  },
  patchcore: {
    label: 'PatchCore', venue: 'CVPR 2022',
    blurb: 'Compares every image patch with a memory bank of normal patches and marks the most unusual ones. It needs no training loop, so new products can be added on the server in minutes.',
  },
}
