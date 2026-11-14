import GameCard from "./GameCard";
import type { Game } from "@/lib/engine";

export default function Grid({ games }: { games: Game[] }) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-6 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-6 [&>a]:w-auto">
      {games.map((g) => <GameCard key={g.item_id} game={g} />)}
    </div>
  );
}
