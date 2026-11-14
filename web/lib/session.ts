// Stateless session: a signed JWT in an httpOnly cookie. Edge-safe (used by middleware.ts too).
import { SignJWT, jwtVerify } from "jose";

export const COOKIE = "rs_session";
const key = () => new TextEncoder().encode(process.env.AUTH_SECRET || "dev-only-secret-change-me");

export type SessionClaims = { uid: number };

export async function signSession(uid: number): Promise<string> {
  return new SignJWT({ uid }).setProtectedHeader({ alg: "HS256" }).setIssuedAt().setExpirationTime("30d").sign(key());
}

export async function verifySession(token: string | undefined): Promise<SessionClaims | null> {
  if (!token) return null;
  try {
    const { payload } = await jwtVerify(token, key());
    return typeof payload.uid === "number" ? { uid: payload.uid } : null;
  } catch {
    return null;
  }
}
