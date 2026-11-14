// Wait until the engine and the app answer before any test runs. A freshly started engine loads its
// models for 1-2 minutes; testing during that window only measures the startup, not the product.
const ENGINE = process.env.E2E_ENGINE_URL ?? "http://127.0.0.1:8000";
const APP = process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000";
const DEADLINE_S = Number(process.env.E2E_WAIT_SECONDS ?? 240);

async function ok(url: string): Promise<boolean> {
  try {
    return (await fetch(url, { redirect: "manual", signal: AbortSignal.timeout(5_000) })).status === 200;
  } catch {
    return false;
  }
}

export default async function globalSetup() {
  const t0 = Date.now();
  for (const url of [`${ENGINE}/health`, `${APP}/login`]) {
    while (!(await ok(url))) {
      if ((Date.now() - t0) / 1000 > DEADLINE_S) throw new Error(`not ready after ${DEADLINE_S}s: ${url}`);
      await new Promise((r) => setTimeout(r, 2_000));
    }
  }
  console.log(`engine + app ready (${((Date.now() - t0) / 1000).toFixed(0)}s): ${ENGINE} ${APP}`);
}
