import "server-only";
// Typed client for the RecoSphere engine (FastAPI). Server-side only: the browser never talks to the
// engine directly, so the engine needs no CORS and its admin token never reaches the client.

const BASE = process.env.ENGINE_URL || "http://127.0.0.1:8000";

// game | accessory | other (engine data/item_types.py). Lists default to games: the app streams games, while the
// model (trained on Amazon purchases) also ranks controllers, cases and gift cards.
export type ItemType = "game" | "accessory" | "other";
export type Game = {
  item_id: string; title: string; platforms: string[]; categories: string[]; year: number | null;
  description: string; is_new: boolean; item_type: ItemType; plays_90d: number; trending_rank: number;
  score?: number | null; similarity?: number;
};
export type Reason = {
  group: string; text: string; weight?: number; feature?: string; share?: number; strength?: "strong" | "medium" | "weak";
};
export type Explanation = {
  in_candidates?: boolean; in_top10?: boolean; rank?: number; n_candidates?: number; score?: number; sources?: string[];
  because_you_played?: string | null; reasons: Reason[];
};
export type RecResponse = {
  user_id: string; source: "personalized" | "fallback" | "fallback_error"; items: Game[];
  request_id: string; experiment: string | null; arm: string; model_run: string;
};

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { cache: "no-store", ...init });
  if (!r.ok) throw new Error(`engine ${path}: HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown, headers: Record<string, string> = {}): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST", cache: "no-store", headers: { "content-type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`engine ${path}: HTTP ${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

const enc = encodeURIComponent;
// the user's "Not interested" games (app_api /feedback/hidden): the engine drops them after ranking (P1 phase G)
const exq = (ex: string[]) => (ex.length ? `&exclude=${ex.slice(0, 500).map(enc).join(",")}` : "");
const adminHeaders = (): Record<string, string> =>
  process.env.ENGINE_ADMIN_TOKEN ? { "x-admin-token": process.env.ENGINE_ADMIN_TOKEN } : {};

export const engine = {
  recommend: (user: string, k = 20, type: ItemType = "game", ex: string[] = []) =>
    get<RecResponse>(`/recommend/${enc(user)}?k=${k}&meta=true&item_type=${type}${exq(ex)}`),
  explain: (user: string, items?: string[], type: ItemType = "game", ex: string[] = []) =>
    get<{ source: string; arm: string; model_run: string; items: Record<string, Explanation> }>(
      `/explain/${enc(user)}?item_type=${type}${items?.length ? `&items=${items.map(enc).join(",")}` : ""}${exq(ex)}`),
  popular: (k = 24, offset = 0, newOnly = false, type: ItemType = "game", ex: string[] = []) =>
    get<{ items: Game[] }>(`/catalog/popular?k=${k}&offset=${offset}&new_only=${newOnly}&item_type=${type}${exq(ex)}`).then((r) => r.items),
  items: (ids: string[]) => get<{ items: Game[] }>(`/catalog/items?ids=${ids.map(enc).join(",")}`).then((r) => r.items),
  search: (q: string, k = 30, type: ItemType = "game") =>
    get<{ items: Game[] }>(`/catalog/search?q=${enc(q)}&k=${k}&item_type=${type}`).then((r) => r.items),
  similar: (item: string, k = 12, user?: string, type: ItemType = "game", ex: string[] = []) =>
    get<{ items: Game[] }>(`/catalog/similar/${enc(item)}?k=${k}&item_type=${type}${user ? `&user_id=${enc(user)}` : ""}${exq(ex)}`).then((r) => r.items),
  history: (user: string, k = 20, type: ItemType = "game") =>
    get<{ items: Game[] }>(`/users/${enc(user)}/history?k=${k}&item_type=${type}`).then((r) => r.items),
  // bulk /event, oldest first (P1 phase F: put the saved library back after an engine restart)
  replay: (user: string, itemIds: string[]) =>
    post<{ recorded: number; already_seen: number; unknown: string[] }>(
      `/users/${enc(user)}/history`, { items: itemIds.map((item_id) => ({ item_id })) }),
  // new to the catalog (not a release date): by the user's taste when there is history, else newest first
  fresh: (user: string, k = 20, type: ItemType = "game", ex: string[] = []) =>
    get<{ ranked_by: "taste" | "newest"; items: Game[] }>(`/users/${enc(user)}/fresh?k=${k}&item_type=${type}${exq(ex)}`),
  demoUsers: (k = 9) =>
    get<{ users: { user_id: string; n_games: number; recent: string[] }[] }>(`/users/demo?k=${k}`).then((r) => r.users),
  event: (user: string, item: string) => post(`/event`, { user_id: user, item_id: item }),
  feedback: (user: string, item: string, requestId: string, action: "click" | "play") =>
    post(`/feedback`, { user_id: user, item_id: item, request_id: requestId, action }),
  health: () => get<Record<string, unknown>>(`/health`),
  adminSummary: () => get<AdminSummary>(`/admin/summary`, { headers: adminHeaders() }),
  rollback: () => post<{ rolled_back_to: string; log: string[]; swap: Record<string, unknown> }>(`/admin/rollback`, undefined, adminHeaders()),
};

export type ArmStats = {
  requests: number; personalized: number; fallback: number; clicks: number; plays: number;
  ctr: number | null; play_rate: number | null; p50_ms_le: number | null; p95_ms_le: number | null;
};
export type AdminSummary = {
  health: Record<string, unknown> & {
    model_run: string; emb_version: string; model_loaded_at: string; users: number; items: number; store: string;
    reloads: number; last_reload_error: string | null; new_item_slots: number;
    experiment: null | { name: string; salt: string; arms: { name: string; model: string; traffic: number }[] };
    experiment_error: string | null; experiment_warning: string | null;
  };
  live: { arms: Record<string, ArmStats>; errors: Record<string, number>; reloads: Record<string, number> };
  production_history: { run_id: string; reason: string; promoted_at: string; metrics: Record<string, number | null> }[];
};
