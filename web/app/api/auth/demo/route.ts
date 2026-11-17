import { NextResponse } from "next/server";
import { api } from "@/lib/api";
import { clientUserAgent, setSessionCookie, toUser } from "@/lib/auth";
import { replayLibrarySafe } from "@/lib/history";
import { engine } from "@/lib/engine";

// "Explore as a real player": sign in as one of the engine's dataset users (only ids the engine offers; app_api
// trusts this check, see docs/API.md POST /auth/demo).
export async function POST(req: Request) {
  const { user_id } = await req.json().catch(() => ({}));
  const offered = await engine.demoUsers(50);
  if (!offered.some((u) => u.user_id === user_id)) return NextResponse.json({ error: "unknown player" }, { status: 400 });
  const s = await api.demo(String(user_id), await clientUserAgent());
  await setSessionCookie(s.session_id);
  await replayLibrarySafe(toUser(s)); // in-app plays on top of the dataset history
  return NextResponse.json({ ok: true, next: "/" });
}
