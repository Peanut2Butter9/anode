import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, createHashRouter, RouterProvider } from 'react-router-dom'
import { STATIC } from './api'
import Layout from './components/Layout'
import './index.css'
import About from './pages/About'
import BenchmarkPage from './pages/Benchmark'
import Inspect from './pages/Inspect'
import NewProduct from './pages/NewProduct'
import ProductPage from './pages/Product'
import Products from './pages/Products'
import Reports from './pages/Reports'

// Static hosting cannot rewrite deep links to index.html, so the static build uses hash URLs
const router = (STATIC ? createHashRouter : createBrowserRouter)([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <Products /> },
      { path: '/products/:id', element: <ProductPage /> },
      { path: '/inspect', element: <Inspect /> },
      { path: '/inspect/:id', element: <Inspect /> },
      { path: '/reports', element: <Reports /> },
      { path: '/reports/:id', element: <Reports /> },
      { path: '/benchmark', element: <BenchmarkPage /> },
      { path: '/new', element: <NewProduct /> },
      { path: '/about', element: <About /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
