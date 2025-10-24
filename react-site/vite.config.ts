import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
    server: {
    port: 3000, // Specify the port here
    proxy: {
      '/api': {
        // target: 'http://localhost:5000', // Replace with your API server URL
        target: 'http://host.docker.internal:5000/',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
