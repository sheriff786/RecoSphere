import "server-only";
// Typed client for app_api (user data on Postgres; docs/API.md section 3). Server-side only: the browser never
// talks to app_api, and the service token never leaves the server.

const BASE = process.env.APP_API_URL || "http://127.0.0.1:8100";
const TOKEN = process.env.APP_API_TOKEN || "";

export type ApiUser = {
  id: string; kind: "registered" | "demo"; email: string | null; display_name: string; role: "player" | "admin";
  dataset_user_id: string | null; onboarded_at: string | null; created_at: string;
};
export type ApiSession = { session_id: string; expires_at: string; needs_onboarding: boolean; user: ApiUser };
export type PlayOut = {
  play_id: string; item_id: string; started_at: string; ended_at: string | null; seconds_played: number | null; first_time: boolean;
};
export type Audit = {
  id: number; actor_id: string | null; action: string; target: string | null; details: Record<string, unknown>; created_at: string;
};
export type LibraryItem = {
  item_id: string; title: string; source: "onboarding" | "play"; added_at: string; last_played_at: string | null;
  play_count: number; seconds_played: number;
};

/** app_api answered with an error status; `code` is app_api's service error code when it sent one. */
export class ApiError extends Error {
  constructor(public status: number, public detail: string, public code?: string) { super(`app_api ${status}: ${detail}`); }
}

async function call<T>(method: string, path: string, opts: { sid?: string; body?: unknown; ua?: string } = {}): Promise<T> {
  const headers: Record<string, string> = { "X-Service-Token": TOKEN };
  if (opts.sid) headers["X-Session-Id"] = opts.sid;
  if (opts.ua) headers["X-Client-User-Agent"] = opts.ua.slice(0, 300);
  if (opts.body !== undefined) headers["content-type"] = "application/json";
  const r = await fetch(`${BASE}${path}`, {
    method, headers, cache: "no-store", body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    const detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    throw new ApiError(r.status, detail, j.code);
  }
  return r.json() as Promise<T>;
}

export const api = {
  signup: (email: string, password: string, display_name: string | undefined, ua?: string) =>
    call<ApiSession>("POST", "/auth/signup", { body: { email, password, display_name }, ua }),
  login: (email: string, password: string, ua?: string) =>
    call<ApiSession>("POST", "/auth/login", { body: { email, password }, ua }),
  demo: (dataset_user_id: string, ua?: string) => call<ApiSession>("POST", "/auth/demo", { body: { dataset_user_id }, ua }),
  logout: (sid: string) => call<{ revoked: boolean }>("POST", "/auth/logout", { sid }),
  me: (sid: string) => call<ApiSession>("GET", "/auth/me", { sid }),
  picks: (sid: string, item_ids: string[]) => call<{ added: string[] }>("POST", "/library/picks", { sid, body: { item_ids } }),
  library: (sid: string, k = 20) => call<LibraryItem[]>("GET", `/library?k=${k}`, { sid }),
  stats: (sid: string) => call<{ games: number; plays: number; seconds_played: number }>("GET", "/library/stats", { sid }),
  playStart: (sid: string, body: { item_id: string; client_event_id: string; request_id?: string | null; surface?: string | null }) =>
    call<PlayOut>("POST", "/plays/start", { sid, body }),
  playEnd: (sid: string, body: { client_event_id: string; seconds: number }) => call<PlayOut>("POST", "/plays/end", { sid, body }),
  // P1 phase G: "Not interested" (item_feedback) and the admin audit log
  setFeedback: (sid: string, item_id: string, kind: "like" | "dislike" | "not_interested") =>
    call<{ item_id: string; kind: string }>("PUT", `/feedback/${encodeURIComponent(item_id)}`, { sid, body: { kind } }),
  clearFeedback: (sid: string, item_id: string) =>
    call<{ cleared: boolean }>("DELETE", `/feedback/${encodeURIComponent(item_id)}`, { sid }),
  hidden: (sid: string) => call<{ item_ids: string[] }>("GET", "/feedback/hidden", { sid }).then((r) => r.item_ids),
  audit: (sid: string, body: { action: string; target?: string | null; details?: Record<string, unknown> }) =>
    call<Audit>("POST", "/admin/audit", { sid, body }),
  auditLog: (sid: string, k = 50) => call<Audit[]>("GET", `/admin/audit?k=${k}`, { sid }),
};
