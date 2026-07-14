import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Static build is served by Hugo from static/basin-explorer/ (no site deps).
// Build with `npm run build`; commit the built assets.
export default defineConfig({
  plugins: [react()],
  base: '/basin-explorer/',
  build: {
    outDir: '../static/basin-explorer',
    emptyOutDir: true,
  },
});
