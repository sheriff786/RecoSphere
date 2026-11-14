import { NextResponse } from "next/server";
import { clearSession } from "@/lib/auth";

// Returns JSON; the BROWSER navigates to /login (components/LogoutButton.tsx). The first version redirected
// with new URL("/login", req.url): inside Next, req.url carries the server's own host (localhost), so a user
// on 127.0.0.1 / a real domain was sent to another origin where the session cookie does not exist (e2e test).
export async function POST() {
  await clearSession();
  return NextResponse.json({ ok: true });
}
