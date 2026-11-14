import { NextResponse } from "next/server";
import { checkPassword } from "@/lib/db";
import { setSessionCookie } from "@/lib/auth";

export async function POST(req: Request) {
  const { email, password } = await req.json().catch(() => ({}));
  const u = email && password ? checkPassword(email, password) : undefined;
  if (!u) return NextResponse.json({ error: "Wrong email or password" }, { status: 401 });
  await setSessionCookie(u.id);
  return NextResponse.json({ ok: true, next: u.onboarded ? "/" : "/onboarding" });
}
