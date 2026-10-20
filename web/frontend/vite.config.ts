import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI server (web/server) serves /api and /bundle on port 7860
const api = 'http://127.0.0.1:7860'

export default defineConfig({
  // Relative asset paths in the static build, so it works under any hosting path
  base: process.env.VITE_STATIC === '1' ? './' : '/',
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': api, '/bundle': api } },
})
