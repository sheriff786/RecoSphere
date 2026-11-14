import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { isAdmin } from "@/lib/db";
import { engine } from "@/lib/engine";

export async function POST() {
  const u = await currentUser();
  if (!isAdmin(u)) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  try { return NextResponse.json(await engine.rollback()); }
  catch (e) { return NextResponse.json({ error: String(e) }, { status: 500 }); }
}
