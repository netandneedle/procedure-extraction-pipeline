import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Where the API lives. The dev server proxies /api and /ws there so the
// browser only ever talks to one origin. Override with VITE_API_URL when the
// backend is not on localhost:8000.
const apiUrl = process.env.VITE_API_URL || "http://localhost:8000";
const wsUrl = apiUrl.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Loopback only, like every other port in the stack. To reach the UI from
    // another machine set `host: true` here and read SECURITY.md first: the
    // API behind it has no authentication.
    port: 5173,
    proxy: {
      "/api": {
        target: apiUrl,
        changeOrigin: true,
      },
      "/ws": {
        target: wsUrl,
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
