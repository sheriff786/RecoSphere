import "server-only";
import { cache } from "react";
import { cookies, headers } from "next/headers";
import { redirect } from "next/navigation";
import { COOKIE, signSession, verifySession } from "./session";
import { api, ApiError, type ApiSession } from "./api";

// The shape pages use. kind: "app" = an account made here, "dataset" = a real player from the dataset (demo).
export type User = {
  id: string; sid: string; email: string | null; name: string; kind: "app" | "dataset";
  engine_user_id: string; onboarded: boolean; admin: boolean;
};

export function toUser(s: ApiSession): User {
  const u = s.user;
  const kind = u.kind === "demo" ? "dataset" : "app";
  return {
    id: u.id, sid: s.session_id, email: u.email, name: u.display_name, kind,
    engine_user_id: kind === "dataset" ? (u.dataset_user_id as string) : `app:${u.id}`,
    onboarded: !s.needs_onboarding, admin: u.role === "admin",
  };
}

/** The signed-in user, or null. One app_api /auth/me per request (cache), however many components ask. */
export const currentUser = cache(async (): Promise<User | null> => {
  const claims = await verifySession((await cookies()).get(COOKIE)?.value);
  if (!claims) return null;
  try {
    return toUser(await api.me(claims.sid));
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) return null;     // logged out, expired or revoked
    throw e;
  }
});

export async function requireUser(): Promise<User> {
  const u = await currentUser();
  if (!u) redirect("/login");
  return u;
}

export async function clientUserAgent(): Promise<string | undefined> {
  return (await headers()).get("user-agent") ?? undefined;
}

export async function setSessionCookie(sid: string) {
  const jar = await cookies();
  jar.set(COOKIE, await signSession(sid), {
    httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production" && process.env.COOKIE_SECURE !== "0",
    path: "/", maxAge: 60 * 60 * 24 * 30,
  });
}

/** The current cookie's session id (even if the session is already dead), for logout. */
export async function cookieSid(): Promise<string | null> {
  return (await verifySession((await cookies()).get(COOKIE)?.value))?.sid ?? null;
}

export async function clearSession() {
  (await cookies()).delete(COOKIE);
}
