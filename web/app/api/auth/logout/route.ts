import { NextResponse } from "next/server";
import { api } from "@/lib/api";
import { clearSession, cookieSid } from "@/lib/auth";

// Revokes the login session in app_api (the old cookie is dead from now on, even if someone kept a copy) and
// clears the cookie. Returns JSON; the BROWSER navigates to /login (components/LogoutButton.tsx): a server
// redirect once sent users to another origin (localhost) where the cookie does not exist.
export async function POST() {
  const sid = await cookieSid();
  if (sid) await api.logout(sid).catch(() => null);      // best effort: the cookie goes either way
  await clearSession();
  return NextResponse.json({ ok: true });
}
