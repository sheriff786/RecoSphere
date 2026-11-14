# RecoSphere Play (Next.js) - multi-stage, standalone output (~150 MB image).
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npx next build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=3000 DATABASE_PATH=/app/data/app.db
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
# better-sqlite3 (native, serverExternalPackages) is traced into .next/standalone/node_modules automatically
RUN mkdir -p /app/data
EXPOSE 3000
CMD ["node", "server.js"]
