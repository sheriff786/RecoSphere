// The cookie: a signed JWT carrying `sid`, the id of the login session row in app_api (Postgres). The signature
// lets middleware.ts (edge runtime, no database) reject forged cookies cheaply; whether the session is still
// alive (not logged out, not expired, user not banned) is checked server-side via app_api /auth/me.
import { SignJWT, jwtVerify } from "jose";

export const COOKIE = "rs_session";
const key = () => new TextEncoder().encode(process.env.AUTH_SECRET || "dev-only-secret-change-me");
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export type SessionClaims = { sid: string };

export async function signSession(sid: string): Promise<string> {
  return new SignJWT({ sid }).setProtectedHeader({ alg: "HS256" }).setIssuedAt().setExpirationTime("30d").sign(key());
}

export async function verifySession(token: string | undefined): Promise<SessionClaims | null> {
  if (!token) return null;
  try {
    const { payload } = await jwtVerify(token, key());
    // cookies from before P1 carried { uid: number }: no sid -> logged out, the user signs in again
    return typeof payload.sid === "string" && UUID.test(payload.sid) ? { sid: payload.sid } : null;
  } catch {
    return null;
  }
}
