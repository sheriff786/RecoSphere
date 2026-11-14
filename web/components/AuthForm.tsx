"use client";
import { useState } from "react";
import Link from "next/link";

export default function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true); setErr(null);
    const body = Object.fromEntries(new FormData(e.currentTarget));
    const r = await fetch(`/api/auth/${mode}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) { setErr(d.error || "Something went wrong"); setBusy(false); return; }
    window.location.href = d.next;
  }
  const field = "w-full rounded-lg bg-white/5 px-4 py-3 text-white placeholder-zinc-500 ring-1 ring-white/10 outline-none focus:ring-violet-400/70";
  return (
    <form onSubmit={submit} className="space-y-3">
      {mode === "signup" && <input name="name" placeholder="Display name" className={field} autoComplete="nickname" />}
      <input name="email" type="email" placeholder="Email" required className={field} autoComplete="email" />
      <input name="password" type="password" placeholder="Password (8+ characters)" required minLength={mode === "signup" ? 8 : 1}
             className={field} autoComplete={mode === "signup" ? "new-password" : "current-password"} />
      {err && <p className="text-sm text-rose-400" role="alert">{err}</p>}
      <button disabled={busy} className="w-full rounded-lg bg-gradient-to-r from-violet-500 to-cyan-400 py-3 font-semibold text-black disabled:opacity-60">
        {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
      </button>
      <p className="text-center text-sm text-zinc-400">
        {mode === "login" ? <>New here? <Link href="/signup" className="text-violet-300 hover:underline">Create an account</Link></>
                          : <>Have an account? <Link href="/login" className="text-violet-300 hover:underline">Sign in</Link></>}
      </p>
    </form>
  );
}
