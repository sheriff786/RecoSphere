import { NextResponse } from "next/server";
import { datasetUser } from "@/lib/db";
import { setSessionCookie } from "@/lib/auth";
import { engine } from "@/lib/engine";

// "Explore as a real player": sign in as one of the engine's dataset users (only ids the engine offers).
export async function POST(req: Request) {
  const { user_id } = await req.json().catch(() => ({}));
  const offered = await engine.demoUsers(50);
  if (!offered.some((u) => u.user_id === user_id)) return NextResponse.json({ error: "unknown player" }, { status: 400 });
  const u = datasetUser(user_id);
  await setSessionCookie(u.id);
  return NextResponse.json({ ok: true, next: "/" });
}
