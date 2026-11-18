import { NextResponse } from "next/server";
import { api } from "@/lib/api";
import { currentUser } from "@/lib/auth";
import { engine } from "@/lib/engine";

// Admin "Roll back": engine POST /admin/rollback, then one row in app_api admin_audit_log (P1 phase G) with who,
// from which run to which, and whether it worked. A failed attempt is audited too.
export async function POST() {
  const u = await currentUser();
  if (!u?.admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const from = await engine.health().then((h) => String(h.model_run ?? "")).catch(() => null);
  let result: Awaited<ReturnType<typeof engine.rollback>> | null = null;
  let error: string | null = null;
  try { result = await engine.rollback(); } catch (e) { error = String(e); }
  const audited = await api.audit(u.sid, {
    action: "model.rollback", target: result?.rolled_back_to ?? null,
    details: { from_run: from, to_run: result?.rolled_back_to ?? null, ok: !error, ...(error ? { error: error.slice(0, 500) } : {}) },
  }).then(() => true).catch((e) => { console.error("audit write failed", e); return false; });
  if (error) return NextResponse.json({ error, audited }, { status: 500 });
  return NextResponse.json({ ...result, audited });
}
