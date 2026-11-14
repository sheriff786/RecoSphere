import "server-only";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { COOKIE, signSession, verifySession } from "./session";
import { getUser, type User } from "./db";

export async function currentUser(): Promise<User | null> {
  const jar = await cookies();
  const claims = await verifySession(jar.get(COOKIE)?.value);
  return claims ? getUser(claims.uid) ?? null : null;
}

export async function requireUser(): Promise<User> {
  const u = await currentUser();
  if (!u) redirect("/login");
  return u;
}

export async function setSessionCookie(uid: number) {
  const jar = await cookies();
  jar.set(COOKIE, await signSession(uid), {
    httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production" && process.env.COOKIE_SECURE !== "0",
    path: "/", maxAge: 60 * 60 * 24 * 30,
  });
}

export async function clearSession() {
  (await cookies()).delete(COOKIE);
}
