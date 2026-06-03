import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8002',
        changeOrigin: true,
        // Long SSE streams (/chat): avoid proxy closing early
        timeout: 0,
        proxyTimeout: 0,
      },
      '/uploads': {
        target: 'http://127.0.0.1:8002',
        changeOrigin: true,
      },
    },
  },
})
