/**
 * Axios instance with base URL configured.
 * In dev, Vite proxies /api to localhost:8000 (FastAPI).
 */
import axios from "axios";

const client = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

export default client;
