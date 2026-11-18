"use client";
import { useState } from "react";

// "Not interested" on a game page (P1 phase G): saved in app_api item_feedback; every recommendation row then
// leaves the game out (the engine drops it after ranking). Undo clears it. Search still finds the game.
export default function NotInterested({ itemId, initiallyHidden }: { itemId: string; initiallyHidden: boolean }) {
  const [hidden, setHidden] = useState(initiallyHidden);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function set(hide: boolean) {
    setBusy(true); setErr(null);
    const r = await fetch("/api/not-interested", {
      method: hide ? "POST" : "DELETE", headers: { "content-type": "application/json" }, body: JSON.stringify({ item_id: itemId }),
    }).catch(() => null);
    if (r?.ok) setHidden(hide); else setErr("Could not save, try again");
    setBusy(false);
  }
  if (hidden) {
    return (
      <span className="text-sm text-zinc-400" data-testid="not-interested-done">
        Hidden from your recommendations ·{" "}
        <button onClick={() => set(false)} disabled={busy} data-testid="not-interested-undo" className="text-violet-300 underline hover:text-violet-200">Undo</button>
        {err && <span className="ml-2 text-rose-300">{err}</span>}
      </span>
    );
  }
  return (
    <span>
      <button onClick={() => set(true)} disabled={busy} data-testid="not-interested"
              className="rounded-full bg-white/5 px-4 py-2 text-sm text-zinc-300 ring-1 ring-white/15 hover:bg-white/10">
        Not interested
      </button>
      {err && <span className="ml-2 text-sm text-rose-300">{err}</span>}
    </span>
  );
}
