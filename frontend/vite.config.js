import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: '../static/dashboard', emptyOutDir: true,
    rollupOptions: { output: { manualChunks(id) {
      if (id.includes('/node_modules/zrender/')) return 'renderer';
      if (id.includes('/node_modules/echarts/')) return 'chart';
    } } },
  },
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': process.env.PARSELABS_API_URL || 'http://127.0.0.1:7862',
      '/brand': process.env.PARSELABS_API_URL || 'http://127.0.0.1:7862',
    },
  },
});
