import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_TARGET || 'http://localhost:5006';

  const proxyConfig = {
    target,
    changeOrigin: true,
    secure: false,
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': proxyConfig,
        '/odata': proxyConfig,
        '/swagger': proxyConfig,
        '/health': proxyConfig,
        '/openapi': proxyConfig,
      },
    },
  };
});
