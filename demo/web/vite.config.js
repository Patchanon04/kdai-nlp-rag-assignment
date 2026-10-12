import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" ให้ build เดียวกันใช้ได้ทั้งหลัง FastAPI และบน GitHub Pages (sub-path)
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { proxy: { "/api": "http://localhost:8080" } },
});
