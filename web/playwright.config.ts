// End-to-end tests (P0). The app and the engine must already be running:
//   make up && make e2e            real stack in Docker (http://127.0.0.1:3000), no rollback
//   make e2e-synthetic             synthetic engine + standalone app, rollback on (what CI runs)
// E2E_BASE_URL points the tests elsewhere. 127.0.0.1, not localhost: IPv6 delay on Windows.
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",   // waits for engine /health and app /login (E2E_ENGINE_URL)
  fullyParallel: false,
  workers: 1,                       // one shared engine; the admin test changes production
  retries: 0,                       // a flaky e2e is a bug to fix, not to retry away
  timeout: 60_000,
  expect: { timeout: 15_000 },      // first engine calls after a start can be slow
  forbidOnly: !!process.env.CI,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
