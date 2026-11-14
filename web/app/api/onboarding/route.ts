import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { setOnboarded } from "@/lib/db";
import { engine } from "@/lib/engine";

export async function POST(req: Request) {
  const u = await currentUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { item_ids } = await req.json().catch(() => ({ item_ids: [] }));
  if (!Array.isArray(item_ids) || item_ids.length < 3) return NextResponse.json({ error: "pick at least 3" }, { status: 400 });
  for (const id of item_ids.slice(0, 20)) await engine.event(u.engine_user_id, String(id));   // cold start -> history
  setOnboarded(u.id);
  return NextResponse.json({ ok: true });
}
