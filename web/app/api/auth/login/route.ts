import { NextResponse } from "next/server";
import { api, ApiError } from "@/lib/api";
import { clientUserAgent, setSessionCookie, toUser } from "@/lib/auth";
import { replayLibrarySafe } from "@/lib/history";

export async function POST(req: Request) {
  const { email, password } = await req.json().catch(() => ({}));
  if (!email || !password) return NextResponse.json({ error: "Wrong email or password" }, { status: 401 });
  try {
    const s = await api.login(String(email), String(password), await clientUserAgent());
    await setSessionCookie(s.session_id);
    await replayLibrarySafe(toUser(s)); // the engine may have restarted since the last visit
    return NextResponse.json({ ok: true, next: s.needs_onboarding ? "/onboarding" : "/" });
  } catch (e) {
    if (e instanceof ApiError && (e.status === 401 || e.status === 422)) {
      return NextResponse.json({ error: "Wrong email or password" }, { status: 401 });
    }
    throw e;
  }
}
