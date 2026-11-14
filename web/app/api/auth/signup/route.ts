import { NextResponse } from "next/server";
import { createAppUser } from "@/lib/db";
import { setSessionCookie } from "@/lib/auth";

export async function POST(req: Request) {
  const { email, name, password } = await req.json().catch(() => ({}));
  if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return NextResponse.json({ error: "Enter a valid email" }, { status: 400 });
  if (!password || password.length < 8) return NextResponse.json({ error: "Password needs at least 8 characters" }, { status: 400 });
  try {
    const u = createAppUser(email, (name || email.split("@")[0]).slice(0, 40), password);
    await setSessionCookie(u.id);
    return NextResponse.json({ ok: true, next: "/onboarding" });
  } catch (e) {
    if (String(e).includes("UNIQUE")) return NextResponse.json({ error: "An account with this email exists" }, { status: 409 });
    throw e;
  }
}
