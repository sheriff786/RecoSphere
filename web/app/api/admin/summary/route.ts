import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { isAdmin } from "@/lib/db";
import { engine } from "@/lib/engine";

export async function GET() {
  const u = await currentUser();
  if (!isAdmin(u)) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  return NextResponse.json(await engine.adminSummary());
}
