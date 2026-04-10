import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_TARGET || 'http://localhost:5006';
  const assistantTarget = env.VITE_ASSISTANT_TARGET || 'http://127.0.0.1:8000';

  const proxyConfig = {
    target,
    changeOrigin: true,
    secure: false,
  };

  const assistantProxyConfig = {
    target: assistantTarget,
    changeOrigin: true,
    secure: false,
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/assistant': assistantProxyConfig,
        '/api': proxyConfig,
        '/odata': proxyConfig,
        '/swagger': proxyConfig,
        '/health': proxyConfig,
        '/openapi': proxyConfig,
      },
    },
  };
});
