import { NextResponse } from "next/server";
import { api } from "@/lib/api";
import { currentUser } from "@/lib/auth";

// Admin audit log (app_api admin_audit_log, newest first) for the admin dashboard (P1 phase G).
export async function GET() {
  const u = await currentUser();
  if (!u?.admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  return NextResponse.json({ items: await api.auditLog(u.sid, 20) });
}
