import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { engine } from "@/lib/engine";

export async function GET() {
  const u = await currentUser();
  if (!u?.admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  return NextResponse.json(await engine.adminSummary());
}
