import { NextResponse } from "next/server";
import { api, ApiError } from "@/lib/api";
import { currentUser } from "@/lib/auth";
import { engine } from "@/lib/engine";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// start: the play is stored in Postgres (app_api: play_sessions + library, one statement, idempotent by
//        client_event_id) and is the interaction the engine learns from -> /event, and, when it came from a
//        recommendation, /feedback action=play (A/B outcome).
// end:   the stream length (sent as a beacon when the page closes).
export async function POST(req: Request) {
  const u = await currentUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  const cid = String(body.client_event_id || "");
  if (!UUID.test(cid)) return NextResponse.json({ error: "client_event_id (uuid) required" }, { status: 400 });
  try {
    if (body.end) {
      const seconds = Math.max(0, Math.min(86_400, Math.round(Number(body.seconds) || 0)));
      await api.playEnd(u.sid, { client_event_id: cid, seconds });
      return NextResponse.json({ ok: true });
    }
    const item = String(body.item_id || "");
    const rid = body.rid ? String(body.rid) : null;
    const play = await api.playStart(u.sid, { item_id: item, client_event_id: cid, request_id: rid,
                                              surface: body.surface ? String(body.surface).slice(0, 64) : null });
    await engine.event(u.engine_user_id, item);
    if (rid) await engine.feedback(u.engine_user_id, item, rid, "play").catch(() => null);
    return NextResponse.json({ ok: true, play_id: play.play_id, first_time: play.first_time });
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return NextResponse.json({ error: "unknown item" }, { status: 404 });
    if (e instanceof ApiError && e.status === 422) return NextResponse.json({ error: "not a game" }, { status: 400 });
    if (e instanceof ApiError && e.status === 409) return NextResponse.json({ error: "conflict" }, { status: 409 });
    throw e;
  }
}
