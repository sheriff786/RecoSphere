import GameCard from "./GameCard";
import type { Game, Explanation } from "@/lib/engine";

export default function Row({ title, subtitle, games, rid, why, badge }: {
  title: string; subtitle?: string; games: Game[]; rid?: string; why?: Record<string, Explanation>; badge?: string;
}) {
  if (!games.length) return null;
  return (
    <section className="mt-10" data-testid="row">
      <div className="mb-3 flex items-baseline gap-3 px-6 md:px-10">
        <h2 className="text-lg font-semibold tracking-tight text-white md:text-xl">{title}</h2>
        {badge && <span className="rounded-full bg-violet-500/15 px-2 py-0.5 text-[11px] font-medium text-violet-300 ring-1 ring-violet-400/30">{badge}</span>}
        {subtitle && <span className="hidden text-sm text-zinc-400 md:inline">{subtitle}</span>}
      </div>
      <div className="no-scrollbar flex snap-x gap-4 overflow-x-auto px-6 pb-2 md:px-10">
        {games.map((g) => <GameCard key={g.item_id} game={g} rid={rid} why={why?.[g.item_id]} />)}
      </div>
    </section>
  );
}
