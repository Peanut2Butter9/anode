// Per-browser state. There are no accounts: custom products and inspection
// batches belong to the browser that created them.

import type { Batch } from './api'

const KEYS = { custom: 'anode.customProducts', batches: 'anode.batches' }

function read<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? (JSON.parse(v) as T) : fallback
  } catch {
    return fallback
  }
}

function write(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value))
    return true
  } catch {
    return false // private mode or quota exceeded
  }
}

export const store = {
  customIds: (): string[] => read<string[]>(KEYS.custom, []),
  addCustomId(id: string) {
    write(KEYS.custom, [...new Set([...store.customIds(), id])])
  },
  batches: (productId?: string): Batch[] =>
    read<Batch[]>(KEYS.batches, []).filter((b) => !productId || b.product_id === productId),
  saveBatch(b: Batch): boolean {
    const all = read<Batch[]>(KEYS.batches, []).filter((x) => x.id !== b.id)
    all.push(b)
    // Keep the newest batches if storage runs out
    while (all.length && !write(KEYS.batches, all)) all.shift()
    return all.includes(b)
  },
  deleteBatch(id: string) {
    write(KEYS.batches, read<Batch[]>(KEYS.batches, []).filter((b) => b.id !== id))
  },
}
