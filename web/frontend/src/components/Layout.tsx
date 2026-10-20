import { NavLink, Link, Outlet } from 'react-router-dom'
import { REPO_URL, STATIC } from '../api'

const NAV = [
  ['/', 'Products'],
  ['/inspect', 'Inspect'],
  ['/reports', 'Reports'],
  ['/benchmark', 'Benchmark'],
  ['/about', 'About'],
] as const

export default function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="h-7 w-7" />
            <span>AnoDe</span>
          </Link>
          <nav className="-mx-1 flex flex-1 gap-1 overflow-x-auto text-sm">
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === '/'}
                className={({ isActive }) => `whitespace-nowrap rounded-lg px-3 py-1.5 ${isActive
                  ? 'bg-slate-100 font-medium text-slate-900 dark:bg-slate-800 dark:text-white'
                  : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'}`}>
                {label}
              </NavLink>
            ))}
          </nav>
          <Link to="/new" className="btn-primary hidden sm:inline-flex">+ New product</Link>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:py-8">
        <Outlet />
      </main>
      <footer className="border-t border-slate-200 py-6 text-center text-xs muted dark:border-slate-800">
        AnoDe, a reproduction of a UGM Machine Learning capstone. Sample images: MVTec AD and MVTec LOCO AD
        (© MVTec Software GmbH, CC BY-NC-SA 4.0). For demonstration only, not for commercial use.
        {STATIC && <> Static demo: model results for the sample images are precomputed. </>}
        {' '}<a href={REPO_URL} className="underline" target="_blank" rel="noreferrer">Code and paper on GitHub</a>.
      </footer>
    </div>
  )
}
