import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { endPlay, startPlay } from "@/lib/db";
import { engine } from "@/lib/engine";

// start: the play IS the interaction the engine learns from -> /event (history + seen), and if it came from a
//        recommendation, /feedback action=play (A/B outcome). end: store session length.
export async function POST(req: Request) {
  const u = await currentUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  if (body.end) {
    endPlay(u.id, Number(body.play_id), Number(body.seconds) || 0);
    return NextResponse.json({ ok: true });
  }
  const item = String(body.item_id || "");
  const [g] = await engine.items([item]);
  if (!g) return NextResponse.json({ error: "unknown item" }, { status: 404 });
  if (g.item_type !== "game") return NextResponse.json({ error: "not a game" }, { status: 400 });
  const ev = (await engine.event(u.engine_user_id, item)) as { recorded: boolean };
  if (body.rid) await engine.feedback(u.engine_user_id, item, String(body.rid), "play").catch(() => null);
  const playId = startPlay(u.id, item, body.rid || null);
  return NextResponse.json({ ok: true, play_id: playId, first_time: ev.recorded });
}
