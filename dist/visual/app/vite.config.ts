import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// Production output lands in the distribution root (dist/visual/): index.html + assets/.
// The Vite project root is this app/ directory, so the build never overwrites its own
// source entry; stale hashed assets are cleaned by tools/clean-build.mjs before builds.
export default defineConfig({
  plugins: [react()],
  base: './',
  publicDir: false,
  build: { outDir: '..', emptyOutDir: false },
  test: { environment: 'node' }
});
