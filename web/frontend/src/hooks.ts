import { useEffect, useState } from 'react'
import { api, type Product } from './api'
import { store } from './store'

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<{ data?: T; error?: unknown; loading: boolean }>({ loading: true })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true, error: undefined }))
    fn().then((data) => alive && setState({ data, loading: false }))
      .catch((error) => alive && setState({ error, loading: false }))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}

let cache: Promise<Product[]> | null = null

/** All products: the bundled ones plus this browser's custom products. */
export function useProducts(refresh = 0) {
  return useAsync(() => {
    if (!cache || refresh) cache = api.products(store.customIds())
    return cache
  }, [refresh])
}

export function invalidateProducts() {
  cache = null
}

export const fmt = {
  pct: (x: number, d = 1) => `${(x * 100).toFixed(d)}%`,
  num: (x: number | null | undefined, d = 3) => (x == null || Number.isNaN(x) ? '–' : x.toFixed(d)),
  today: () => new Date().toISOString().slice(0, 10),
}
