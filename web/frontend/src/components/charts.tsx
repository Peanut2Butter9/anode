import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { Batch, QCReport } from '../api'

const AXIS = { fontSize: 11, fill: 'currentColor', opacity: 0.6 }
const RED = '#e11d48'
const TEAL = '#0d9488'
const SLATE = '#64748b'

export function PChart({ batches, report }: { batches: Batch[]; report: QCReport }) {
  const data = batches.map((b, i) => {
    const pt = report.p_chart.points[i]
    return { label: b.date.slice(5), batch: b.batch_no, n: b.n, p: pt.p, ucl: pt.ucl, lcl: pt.lcl, ooc: pt.out_of_control }
  })
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" strokeOpacity={0.1} />
        <XAxis dataKey="label" tick={AXIS} interval="preserveStartEnd" minTickGap={16} />
        <YAxis tick={AXIS} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} width={44} />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 8 }}
          formatter={(v, name) => [`${(Number(v) * 100).toFixed(2)}%`, String(name)]}
          labelFormatter={(_, p) => (p?.[0] ? `${p[0].payload.batch} · n = ${p[0].payload.n}` : '')} />
        <Line type="stepAfter" dataKey="ucl" name="UCL" stroke={RED} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
        <Line type="stepAfter" dataKey="lcl" name="LCL" stroke={RED} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
        <ReferenceLine y={report.p_chart.p_bar} stroke={TEAL} strokeWidth={1.5}
          label={{ value: `p̄ = ${(report.p_chart.p_bar * 100).toFixed(2)}%`, position: 'insideTopRight', fontSize: 11, fill: TEAL }} />
        <Line type="linear" dataKey="p" name="Defect rate" stroke={SLATE} strokeWidth={2} isAnimationActive={false}
          dot={(props) => {
            const { cx, cy, payload, index } = props as { cx: number; cy: number; payload: { ooc: boolean }; index: number }
            return <circle key={index} cx={cx} cy={cy} r={payload.ooc ? 5.5 : 3.5} fill={payload.ooc ? RED : SLATE} stroke="white" strokeWidth={1} />
          }} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

/** Distribution of normalised scores (score / threshold); USL = 1. */
export function ScoreHistogram({ scores }: { scores: number[] }) {
  const step = 0.1
  const top = 2.5
  const bins = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => ({ x: i * step, label: (i * step).toFixed(1), count: 0 }))
  for (const s of scores) bins[Math.min(bins.length - 1, Math.max(0, Math.floor(s / step)))].count++
  bins[bins.length - 1].label = `≥${top}`
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={bins} margin={{ top: 10, right: 16, bottom: 0, left: 0 }} barCategoryGap={1}>
        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" strokeOpacity={0.1} vertical={false} />
        <XAxis dataKey="label" tick={AXIS} interval={4} />
        <YAxis tick={AXIS} width={44} allowDecimals={false} />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v) => [String(v), 'units']}
          labelFormatter={(l) => `score ${l}× threshold`} />
        <ReferenceLine x="1.0" stroke={RED} strokeWidth={1.5} label={{ value: 'USL', position: 'top', fontSize: 11, fill: RED }} />
        <Bar dataKey="count" isAnimationActive={false}>
          {bins.map((b) => <Cell key={b.label} fill={b.x >= 1 ? RED : TEAL} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
