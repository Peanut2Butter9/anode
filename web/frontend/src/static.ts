// Browser versions of the server's quality-control math and simulated history,
// used by the static deployment (no Python server). They mirror
// src/anode/qc.py and web/server/anode_web/demo.py; tests in tests/test_qc.py
// pin the Python versions.

import type { Batch, Capability, QCReport } from './api'

/** Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9). */
export function normInv(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239]
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
  const lo = 0.02425
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  }
  if (p > 1 - lo) return -normInv(1 - p)
  const q = p - 0.5
  const r = q * q
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
}

export function pChart(defectives: number[], sizes: number[], k = 3): QCReport['p_chart'] {
  const pBar = defectives.reduce((a, x) => a + x, 0) / sizes.reduce((a, x) => a + x, 0)
  return {
    p_bar: pBar,
    points: sizes.map((n, i) => {
      const s = Math.sqrt((pBar * (1 - pBar)) / n)
      const p = defectives[i] / n
      const ucl = Math.min(1, pBar + k * s)
      const lcl = Math.max(0, pBar - k * s)
      return { p, ucl, lcl, out_of_control: p > ucl || p < lcl }
    }),
  }
}

const rate = (cpk: number): Capability['rating'] => (cpk >= 1.33 ? 'capable' : cpk >= 1 ? 'marginal' : 'not capable')

export function capability(scores: number[], usl = 1): Capability {
  const n = scores.length
  const mean = scores.reduce((a, x) => a + x, 0) / n
  const std = Math.sqrt(scores.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1)) || Number.EPSILON
  const cpu = (usl - mean) / (3 * std)
  const above = scores.filter((x) => x > usl).length / n
  return {
    mean, std, usl, cpu, cpk: cpu, cp: null, fraction_above_usl: above,
    cpk_from_yield: above > 0 && above < 1 ? normInv(1 - above) / 3 : null, rating: rate(cpu),
  }
}

export function qcReport(batches: Pick<Batch, 'n' | 'defective' | 'scores'>[]): QCReport {
  const all = batches.flatMap((b) => b.scores)
  return {
    p_chart: pChart(batches.map((b) => b.defective), batches.map((b) => b.n)),
    capability: all.length >= 2 ? capability(all) : null,
    capability_per_batch: batches.map((b) => (b.scores.length >= 2 ? capability(b.scores) : null)),
  }
}

// Seeded generator, so a product's simulated history is the same on every visit
function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hash(s: string) {
  let h = 2166136261
  for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  return h >>> 0
}

/** Same scheme as demo.py: real test-set scores at a 2-5% defect rate, one batch at 4x. */
export function demoHistory(productId: string, pool: { good: number[]; defective: number[] }, threshold: number, days = 30): Batch[] {
  const rnd = mulberry32(hash(productId))
  const pick = (xs: number[]) => xs[Math.floor(rnd() * xs.length)]
  const today = new Date()
  const dates: Date[] = []
  for (let d = days; d >= 1; d--) {
    const day = new Date(today.getFullYear(), today.getMonth(), today.getDate() - d)
    if (day.getDay() !== 0) dates.push(day) // Monday to Saturday
  }
  const spike = Math.floor(dates.length / 3 + rnd() * (dates.length - Math.floor(dates.length / 3)))
  const baseRate = 0.02 + rnd() * 0.03
  return dates.map((day, i) => {
    const n = 80 + Math.floor(rnd() * 81)
    const r = baseRate * (i === spike ? 4 : 1)
    const scores = Array.from({ length: n }, () => Math.round((rnd() < r ? pick(pool.defective) : pick(pool.good)) / threshold * 1e4) / 1e4)
    const iso = `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, '0')}-${String(day.getDate()).padStart(2, '0')}`
    return {
      id: `demo-${productId}-${iso}`, product_id: productId, date: iso,
      batch_no: `B${iso.slice(2).replaceAll('-', '')}-${String(i + 1).padStart(2, '0')}`,
      n, defective: scores.filter((s) => s >= 1).length, scores, demo: true,
    }
  })
}
