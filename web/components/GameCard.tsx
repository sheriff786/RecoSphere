"use client";
import Link from "next/link";
import Cover from "./Cover";
import type { Game, Explanation } from "@/lib/engine";

// One tile. Clicking it sends a click beacon tied to the engine's request_id (the A/B outcome), then
// navigates; the request_id travels in the URL so a later Play is attributed to the same exposure.
export default function GameCard({ game, rid, why, size = "md" }: {
  game: Game; rid?: string; why?: Explanation; size?: "md" | "lg";
}) {
  const href = `/game/${encodeURIComponent(game.item_id)}${rid ? `?rid=${rid}` : ""}`;
  const onClick = () => {
    if (!rid) return;
    navigator.sendBeacon("/api/feedback", new Blob([JSON.stringify({ request_id: rid, item_id: game.item_id, action: "click" })],
      { type: "application/json" }));
  };
  const reason = why?.reasons?.[0]?.text;
  return (
    <Link href={href} onClick={onClick} data-testid="game-card"
          className={`group relative shrink-0 ${size === "lg" ? "w-52" : "w-40"} snap-start`}>
      <div className="overflow-hidden rounded-xl ring-1 ring-white/10 transition duration-200 group-hover:-translate-y-1 group-hover:ring-2 group-hover:ring-violet-400/80 group-hover:shadow-[0_10px_40px_-10px_rgba(139,92,246,0.7)]">
        <Cover title={game.title} platform={game.platforms?.[0]} isNew={game.is_new} className="block aspect-[3/4] w-full" />
      </div>
      <div className="mt-2 line-clamp-2 text-sm font-medium text-zinc-100" title={game.title}>{game.title}</div>
      <div className="mt-0.5 text-xs text-zinc-400">
        {[game.year, ...(game.platforms ?? [])].filter(Boolean).join(" · ") || " "}
      </div>
      {reason && <div className="mt-1 line-clamp-1 text-[11px] text-violet-300/90" title={why?.reasons.map((r) => r.text).join(" · ")}>✦ {reason}</div>}
    </Link>
  );
}
