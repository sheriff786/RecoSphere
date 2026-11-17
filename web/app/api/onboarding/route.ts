import { NextResponse } from "next/server";
import { api, ApiError } from "@/lib/api";
import { currentUser } from "@/lib/auth";
import { engine } from "@/lib/engine";

// Onboarding picks: saved in the user's library (Postgres, the source of truth) and sent to the engine as the
// first history (cold start). app_api rejects anything that is not an active game.
export async function POST(req: Request) {
  const u = await currentUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { item_ids } = await req.json().catch(() => ({ item_ids: [] }));
  if (!Array.isArray(item_ids) || item_ids.length < 3) return NextResponse.json({ error: "pick at least 3" }, { status: 400 });
  const ids = [...new Set(item_ids.map(String))].slice(0, 20);
  try {
    await api.picks(u.sid, ids);
  } catch (e) {
    if (e instanceof ApiError && e.status === 422) return NextResponse.json({ error: "pick games only" }, { status: 400 });
    throw e;
  }
  for (const id of ids) await engine.event(u.engine_user_id, id);
  return NextResponse.json({ ok: true });
}
