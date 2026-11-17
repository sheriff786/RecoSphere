// P1 phase F: an engine restart must not make app users strangers. Postgres (app_api library) is the source of
// truth; the web app replays it into the engine on login (user B) and on the home page when the engine has no
// history for a signed-in user (user A, who keeps the cookie). Run by scripts/e2e_stack.sh in two phases:
//   E2E_RESTART_PHASE=before  -> two users onboard; state saved to E2E_RESTART_STATE
//   (the script restarts the engine and checks it really forgot both users)
//   E2E_RESTART_PHASE=after   -> both users get a personalized feed again
// Skipped in a normal `npx playwright test` run (needs the script to restart the engine in between).
import { readFileSync, writeFileSync } from "node:fs";
import { expect, test, type Browser, type Page } from "@playwright/test";

const PHASE = process.env.E2E_RESTART_PHASE;
const STATE = process.env.E2E_RESTART_STATE ?? "";
const PASSWORD = "e2e-password-123";
const ENGINE = process.env.E2E_ENGINE_URL ?? "http://127.0.0.1:8000";

test.describe.configure({ mode: "serial" });
test.skip(!PHASE || !STATE, "run by scripts/e2e_stack.sh (E2E_RESTART_PHASE + E2E_RESTART_STATE)");

// engine_user_id is added by the script between the phases (it reads users.id from the throwaway database)
type Saved = { a: { email: string; engine_user_id: string; storage: Awaited<ReturnType<import("@playwright/test").BrowserContext["storageState"]>> };
               b: { email: string; engine_user_id: string } };

const atPath = (page: Page, path: string) =>
  page.waitForURL((u) => u.pathname === path && u.origin === new URL(test.info().project.use.baseURL!).origin);

async function signupAndOnboard(page: Page, email: string) {
  await page.goto("/signup");
  await page.getByPlaceholder("Display name").fill("E2E Restart");
  await page.getByPlaceholder("Email").fill(email);
  await page.getByPlaceholder(/^Password/).fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await atPath(page, "/onboarding");
  const picks = page.getByTestId("onboard-game");
  for (let i = 0; i < 3; i++) await picks.nth(i).click();
  await page.getByTestId("onboard-done").click();
  await atPath(page, "/");
  await expect(page.getByTestId("engine-strip")).toContainText("source personalized");
  await expect(page.getByRole("heading", { name: /^Because you played / })).toBeVisible();
}

const historyIds = async (page: Page, engineUser: string): Promise<string[]> => {
  const r = await page.request.get(`${ENGINE}/users/${encodeURIComponent(engineUser)}/history?k=50`);
  expect(r.ok()).toBeTruthy();
  return ((await r.json()).items as { item_id: string }[]).map((g) => g.item_id);
};

async function expectPersonalized(page: Page) {
  await expect(page.getByTestId("engine-strip")).toContainText("source personalized");
  await expect(page.getByRole("heading", { name: /^Because you played / })).toBeVisible();
  await expect(page.getByText("We don't know your taste yet")).toHaveCount(0);
}

test("before the engine restart: two users onboard", async ({ browser }: { browser: Browser }) => {
  test.skip(PHASE !== "before");
  const stamp = Date.now();
  const saved = { a: { email: `e2e-restart-a-${stamp}@example.test` }, b: { email: `e2e-restart-b-${stamp}@example.test` } };
  const ctxA = await browser.newContext();
  await signupAndOnboard(await ctxA.newPage(), saved.a.email);
  const storage = await ctxA.storageState();                     // A keeps this cookie across the restart
  const ctxB = await browser.newContext();
  await signupAndOnboard(await ctxB.newPage(), saved.b.email);
  writeFileSync(STATE, JSON.stringify({ a: { ...saved.a, storage }, b: saved.b }));
});

test("after the engine restart: the home page and login put the library back", async ({ browser }: { browser: Browser }) => {
  test.skip(PHASE !== "after");
  const saved: Saved = JSON.parse(readFileSync(STATE, "utf8"));

  // A: still signed in (same cookie), just opens the home page -> replay from the home page
  const pageA = await (await browser.newContext({ storageState: saved.a.storage })).newPage();
  expect(await historyIds(pageA, saved.a.engine_user_id), "the restart wiped A").toEqual([]);
  await pageA.goto("/");
  await atPath(pageA, "/");
  await expectPersonalized(pageA);
  expect(await historyIds(pageA, saved.a.engine_user_id)).toHaveLength(3);

  // B: signs in -> replay from the login route. Checked on the engine BEFORE any page loads, or the home page's
  // own replay would hide a login route that forgot to replay.
  const pageB = await (await browser.newContext()).newPage();
  expect(await historyIds(pageB, saved.b.engine_user_id), "the restart wiped B").toEqual([]);
  const res = await pageB.request.post("/api/auth/login", { data: { email: saved.b.email, password: PASSWORD } });
  expect(res.ok()).toBeTruthy();
  expect(await historyIds(pageB, saved.b.engine_user_id), "login replayed the library").toHaveLength(3);
  await pageB.goto("/");
  await atPath(pageB, "/");
  await expectPersonalized(pageB);
});
