import "server-only";
import { api } from "@/lib/api";
import type { User } from "@/lib/auth";
import { engine } from "@/lib/engine";

// P1 phase F. Postgres (app_api `library`) is the source of truth for what an app user played; the engine's
// history (Redis or in-memory) is a cache of it that an engine restart can lose. This puts the library back.
// The engine dedups exactly like POST /event, so replaying twice records nothing new.
const REPLAY_MAX = 200; // app_api's /library limit; the engine keeps only the last hist_len of them anyway

export async function replayLibrary(user: Pick<User, "sid" | "engine_user_id">) {
  const lib = await api.library(user.sid, REPLAY_MAX);          // newest first
  if (lib.length === 0) return null;
  return engine.replay(user.engine_user_id, lib.map((g) => g.item_id).reverse()); // oldest first
}

/** Replay, but never fail the caller: a login or a page must still work when the engine is down. */
export async function replayLibrarySafe(user: Pick<User, "sid" | "engine_user_id">) {
  try {
    return await replayLibrary(user);
  } catch (e) {
    console.error("library replay failed", e);
    return null;
  }
}
