/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",                         // small Docker image (web/Dockerfile)
  serverExternalPackages: ["better-sqlite3"],   // native module: load from node_modules, don't bundle
};
export default nextConfig;
