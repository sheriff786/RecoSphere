import { NextResponse } from "next/server";
import { api, ApiError } from "@/lib/api";
import { currentUser } from "@/lib/auth";

// "Not interested" (P1 phase G): POST hides a game from every recommendation row, DELETE undoes it.
// Stored in app_api item_feedback (kind not_interested); the pages send the list to the engine as `exclude`.
async function handle(req: Request, hide: boolean) {
  const u = await currentUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { item_id } = await req.json().catch(() => ({}));
  if (!item_id || typeof item_id !== "string") return NextResponse.json({ error: "item_id required" }, { status: 400 });
  try {
    if (hide) await api.setFeedback(u.sid, item_id, "not_interested");
    else await api.clearFeedback(u.sid, item_id);
    return NextResponse.json({ ok: true, hidden: hide });
  } catch (e) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 422)) {
      return NextResponse.json({ error: "unknown game" }, { status: 400 });
    }
    throw e;
  }
}

export async function POST(req: Request) { return handle(req, true); }
export async function DELETE(req: Request) { return handle(req, false); }
