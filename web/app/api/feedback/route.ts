import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { engine } from "@/lib/engine";

// Click beacon from a recommendation tile -> engine /feedback (joins the exposure log on request_id).
export async function POST(req: Request) {
  const u = await currentUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { request_id, item_id, action } = await req.json().catch(() => ({}));
  if (!request_id || !item_id) return NextResponse.json({ ok: false }, { status: 400 });
  await engine.feedback(u.engine_user_id, item_id, request_id, action === "play" ? "play" : "click").catch(() => null);
  return NextResponse.json({ ok: true });
}
