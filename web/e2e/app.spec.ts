// RecoSphere Play end to end: browser -> Next.js app -> engine. Data-agnostic: the same assertions hold on
// the synthetic engine (CI) and on the real Video_Games model (make up). See playwright.config.ts.
import { expect, test, type Locator, type Page } from "@playwright/test";

const ADMIN_EMAIL = process.env.E2E_ADMIN_EMAIL ?? "admin@recosphere.dev";
const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD ?? "admin12345";
// Roll back rewrites production.json. Never on a real stack unless asked: CI's synthetic run sets it.
const ALLOW_ROLLBACK = process.env.E2E_ALLOW_ROLLBACK === "1";
const PASSWORD = "e2e-password-123";
const ENGINE = process.env.E2E_ENGINE_URL ?? "http://127.0.0.1:8000";
// A game title never contains these words: data/item_types.py classifies such items as accessories.
const ACCESSORY_WORDS = /\b(controllers?|gamepads?|headsets?|chargers?)\b/i;

test.describe.configure({ mode: "serial" });

// Every page must load without browser console errors (hydration errors, failed fetches ...).
let consoleErrors: string[] = [];
test.beforeEach(({ page }) => {
  consoleErrors = [];
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(`[${new URL(page.url()).pathname}] ${m.text()}`); });
  page.on("pageerror", (e) => consoleErrors.push(`[${new URL(page.url()).pathname}] ${e}`));});
test.afterEach(() => expect(consoleErrors, "browser console errors").toEqual([]));

// Same ORIGIN too: a redirect to another host (localhost vs 127.0.0.1) loses the session cookie.
const atPath = (page: Page, path: string) =>
  page.waitForURL((u) => u.pathname === path && u.origin === new URL(test.info().project.use.baseURL!).origin);

async function login(page: Page, email: string, password: string) {
  await page.goto("/login");
  await page.getByPlaceholder("Email").fill(email);
  await page.getByPlaceholder(/^Password/).fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await atPath(page, "/");
}

const row = (page: Page, title: string | RegExp) =>
  page.getByTestId("row").filter({ has: page.getByRole("heading", { name: title }) });

// titles drawn on generated covers (svg aria-label) inside a locator
const coverTitles = (l: Locator) =>
  l.locator("svg[role=img]").evaluateAll((els) => els.map((e) => e.getAttribute("aria-label") ?? ""));

// item ids of the cards in a row, in order (from the card links: /game/<id>?rid=...)
async function cardIds(r: Locator): Promise<string[]> {
  const hrefs = await r.getByTestId("game-card").evaluateAll((els) => els.map((e) => e.getAttribute("href") ?? ""));
  return hrefs.map((h) => decodeURIComponent(h.split("/game/")[1].split("?")[0]));
}

test("logged-out visitors go to login and the APIs refuse them", async ({ page }) => {
  await page.goto("/");
  await atPath(page, "/login");
  expect((await page.request.get("/api/admin/summary")).status()).toBe(401);
  expect((await page.request.post("/api/play", { data: { item_id: "x" } })).status()).toBe(401);
});

test("new user: signup -> onboarding -> personalized feed -> play changes it -> sign out/in", async ({ page }) => {
  const email = `e2e-${Date.now()}@example.test`;

  // signup -> onboarding: Continue stays disabled until 3 picks
  await page.goto("/signup");
  await page.getByPlaceholder("Display name").fill("E2E Player");
  await page.getByPlaceholder("Email").fill(email);
  await page.getByPlaceholder(/^Password/).fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await atPath(page, "/onboarding");
  const picks = page.getByTestId("onboard-game");
  const done = page.getByTestId("onboard-done");
  await expect(picks.first()).toBeVisible();
  expect((await coverTitles(page.getByTestId("onboard-game"))).filter((t) => ACCESSORY_WORDS.test(t)),
         "onboarding offers games only").toEqual([]);
  await expect(done).toBeDisabled();
  for (let i = 0; i < 3; i++) await picks.nth(i).click();
  await expect(done).toBeEnabled();
  await done.click();

  // home: the 3 picks are history, so the engine personalizes (not the popular fallback)
  await atPath(page, "/");
  await expect(page.getByTestId("engine-strip")).toContainText("source personalized");
  await expect(page.getByTestId("hero-title")).not.toBeEmpty();
  const top = row(page, "Top picks for you");
  const before = await cardIds(top);
  expect(before.length).toBeGreaterThanOrEqual(5);
  const homeTitles = await page.getByTestId("card-title").evaluateAll((els) => els.map((e) => e.getAttribute("title") ?? ""));
  expect(homeTitles.filter((t) => ACCESSORY_WORDS.test(t)), "home rows show games only").toEqual([]);

  // an ordinary user is not an admin
  expect((await page.request.get("/api/admin/summary")).status()).toBe(403);

  // open a top pick: details + "Why this?"
  const card = top.getByTestId("game-card").first();
  const title = (await card.getByTestId("card-title").getAttribute("title"))!;
  const played = before[0];
  await card.click();
  await expect(page.getByTestId("game-title")).toHaveText(title);
  await expect(page.getByTestId("why-this")).toBeVisible();
  await expect(page.getByTestId("why-title")).toHaveText("Why this?");             // it IS in the feed top 10
  const whyText = (await page.getByTestId("why-this").textContent()) ?? "";
  expect(whyText, "users see strength bars, not SHAP numbers or ranks").not.toMatch(/\d\.\d{2}|candidates/);

  // play it: the engine records the event
  await page.getByTestId("play-button").click();
  await expect(page.getByTestId("player-title")).toHaveText(title);
  await expect(page.getByTestId("play-toast")).toContainText("Added to your history");

  // back home: the feed reacts at once
  await page.getByTestId("end-session").click();
  await atPath(page, "/");
  await expect(page.getByRole("heading", { name: `Because you played ${title}` })).toBeVisible();
  const after = await cardIds(top);
  expect(after, "a played game is never recommended again").not.toContain(played);
  expect(after).not.toEqual(before);

  // the played game is no longer in the feed: its panel explains the score instead of claiming "why this"
  await page.goto(`/game/${encodeURIComponent(played)}`);
  await expect(page.getByTestId("why-title")).toHaveText("How this game scores for you");

  // sign out, sign in again: straight to the feed (already onboarded), still personalized
  await page.getByRole("button", { name: "Sign out" }).click();
  await atPath(page, "/login");
  await login(page, email, PASSWORD);
  await expect(page.getByTestId("engine-strip")).toContainText("source personalized");
});

test("accessories are never offered as games: search, game page, play", async ({ page, request }) => {
  await page.goto("/signup");
  await page.getByPlaceholder("Email").fill(`e2e-gear-${Date.now()}@example.test`);
  await page.getByPlaceholder(/^Password/).fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await atPath(page, "/onboarding");

  await page.goto("/search?q=controller");
  await expect(page.getByText("No games found.")).toBeVisible();

  const gear = (await (await request.get(`${ENGINE}/catalog/popular?k=1&item_type=accessory`)).json()).items[0];
  expect(gear?.item_type, "the catalog has at least one accessory").toBe("accessory");
  const id = encodeURIComponent(gear.item_id);
  await page.goto(`/game/${id}`);
  await expect(page.getByTestId("not-playable")).toBeVisible();
  await expect(page.getByTestId("play-button")).toHaveCount(0);
  await page.goto(`/play/${id}`);
  await page.waitForURL((u) => u.pathname === `/game/${id}` || u.pathname === `/game/${gear.item_id}`);
  expect((await page.request.post("/api/play", { data: { item_id: gear.item_id } })).status()).toBe(400);
});

test("game page hides the trending rank of a game with fewer than 20 players in 90 days", async ({ page, request }) => {
  let rare: { item_id: string; plays_90d: number } | undefined;
  for (let offset = 0; offset < 3000 && !rare; offset += 100) {
    const items = (await (await request.get(`${ENGINE}/catalog/popular?k=100&offset=${offset}&item_type=game`)).json()).items;
    if (!items.length) break;
    rare = items.find((x: { plays_90d: number }) => x.plays_90d > 0 && x.plays_90d < 20);
  }
  test.skip(!rare, "no game with 1-19 players in 90 days in this catalog");
  await page.goto("/signup");
  await page.getByPlaceholder("Email").fill(`e2e-rare-${Date.now()}@example.test`);
  await page.getByPlaceholder(/^Password/).fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await atPath(page, "/onboarding");
  await page.goto(`/game/${encodeURIComponent(rare!.item_id)}`);
  await expect(page.getByText(/in the last 90 days/)).toBeVisible();
  await expect(page.getByText(/trending #/)).toHaveCount(0);
});

test("admin: engine dashboard shows the live model and the A/B table", async ({ page }) => {
  await login(page, ADMIN_EMAIL, ADMIN_PASSWORD);
  await page.goto("/admin");
  await expect(page.getByTestId("admin-model")).not.toBeEmpty();
  await expect(page.getByRole("heading", { name: "A/B experiment" })).toBeVisible();

  // Why-this inspector: the raw numbers users never see
  const who = (await (await page.request.get(`${ENGINE}/users/demo?k=1`)).json()).users[0].user_id;
  const what = (await (await page.request.get(`${ENGINE}/recommend/${encodeURIComponent(who)}?k=1&item_type=game`)).json()).items[0].item_id;
  await page.getByLabel("user id").fill(who);
  await page.getByLabel("item id").fill(what);
  await page.getByRole("button", { name: "Explain" }).click();
  await expect(page.getByTestId("explain-raw")).toContainText("in top 10 (game feed): true");
  await expect(page.getByTestId("explain-raw")).toContainText(/\d\.\d{3}/);            // SHAP weights
  expect((await page.request.get("/api/admin/summary")).status()).toBe(200);
});

test("admin: roll back hot-swaps the engine to the previous model", async ({ page }) => {
  test.skip(!ALLOW_ROLLBACK, "rewrites production.json: only on a disposable stack (E2E_ALLOW_ROLLBACK=1)");
  await login(page, ADMIN_EMAIL, ADMIN_PASSWORD);
  await page.goto("/admin");
  const model = page.getByTestId("admin-model");
  await expect(model).not.toBeEmpty();
  const before = (await model.textContent())!.trim();
  page.once("dialog", (d) => d.accept());
  await page.getByTestId("rollback").click();
  const msg = page.getByText(/^Rolled back to /);
  await expect(msg).toBeVisible();
  const target = (await msg.textContent())!.replace("Rolled back to ", "").trim();
  expect(target).not.toBe(before);
  await expect(model).toHaveText(target);           // the engine hot-swapped to it
});
