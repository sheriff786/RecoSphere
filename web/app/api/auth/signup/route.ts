import { NextResponse } from "next/server";
import { api, ApiError } from "@/lib/api";
import { clientUserAgent, setSessionCookie } from "@/lib/auth";

export async function POST(req: Request) {
  const { email, name, password } = await req.json().catch(() => ({}));
  if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return NextResponse.json({ error: "Enter a valid email" }, { status: 400 });
  if (!password || password.length < 8) return NextResponse.json({ error: "Password needs at least 8 characters" }, { status: 400 });
  try {
    const s = await api.signup(email, password, name ? String(name).slice(0, 40) : undefined, await clientUserAgent());
    await setSessionCookie(s.session_id);
    return NextResponse.json({ ok: true, next: "/onboarding" });
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) return NextResponse.json({ error: "An account with this email exists" }, { status: 409 });
    if (e instanceof ApiError && e.status === 422) return NextResponse.json({ error: "Check the email and password (8-72 bytes)" }, { status: 400 });
    throw e;
  }
}
