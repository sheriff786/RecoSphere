"use client";
import { useState } from "react";

export default function DemoPlayers({ users }: { users: { user_id: string; n_games: number; recent: string[] }[] }) {
  const [busy, setBusy] = useState<string | null>(null);
  async function pick(id: string) {
    setBusy(id);
    const r = await fetch("/api/auth/demo", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ user_id: id }) });
    if (r.ok) window.location.href = (await r.json()).next; else setBusy(null);
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {users.map((u) => (
        <button key={u.user_id} onClick={() => pick(u.user_id)} disabled={!!busy} data-testid="demo-player"
                className="rounded-xl bg-white/[0.03] p-4 text-left ring-1 ring-white/10 transition hover:bg-white/[0.06] hover:ring-cyan-400/50 disabled:opacity-50">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs text-zinc-500">{u.user_id.slice(0, 14)}…</span>
            <span className="text-xs text-cyan-300">{u.n_games} games</span>
          </div>
          <div className="mt-2 text-xs text-zinc-400">Recently:</div>
          <ul className="mt-1 space-y-0.5 text-sm text-zinc-200">
            {u.recent.map((t) => <li key={t} className="line-clamp-1">• {t}</li>)}
          </ul>
          {busy === u.user_id && <div className="mt-2 text-xs text-violet-300">Signing in…</div>}
        </button>
      ))}
    </div>
  );
}
