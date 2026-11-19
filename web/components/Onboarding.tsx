"use client";
import { useState } from "react";
import Cover from "./Cover";
import type { Game } from "@/lib/engine";

export default function Onboarding({ games }: { games: Game[] }) {
  const [sel, setSel] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const toggle = (id: string) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  async function done() {
    setBusy(true);
    const r = await fetch("/api/onboarding", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ item_ids: sel }) });
    if (r.ok) window.location.href = "/"; else setBusy(false);
  }
  return (
    <>
      <div className="grid grid-cols-3 gap-4 sm:grid-cols-4 md:grid-cols-6">
        {games.map((g) => {
          const on = sel.includes(g.item_id);
          return (
            <button key={g.item_id} onClick={() => toggle(g.item_id)} data-testid="onboard-game" aria-pressed={on}
                    className={`relative overflow-hidden rounded-xl text-left ring-2 transition ${on ? "ring-cyan-400 scale-[0.97]" : "ring-transparent hover:ring-white/30"}`}>
              <Cover title={g.title} platform={g.platforms?.[0]} isNew={g.is_new} image={g.image} className="block aspect-[3/4] w-full" />
              {on && <span className="absolute right-2 top-2 grid h-7 w-7 place-items-center rounded-full bg-cyan-400 font-bold text-black">✓</span>}
            </button>
          );
        })}
      </div>
      <div className="fixed inset-x-0 bottom-0 border-t border-white/10 bg-[#07070c]/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <span className="text-sm text-zinc-300">{sel.length < 3 ? `Pick ${3 - sel.length} more` : `${sel.length} selected. Nice taste.`}</span>
          <button onClick={done} disabled={sel.length < 3 || busy} data-testid="onboard-done"
                  className="rounded-full bg-gradient-to-r from-violet-500 to-cyan-400 px-6 py-2.5 font-semibold text-black disabled:opacity-40">
            {busy ? "Building your feed…" : "Continue"}
          </button>
        </div>
      </div>
    </>
  );
}
