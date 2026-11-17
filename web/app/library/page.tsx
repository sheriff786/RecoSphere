import { requireUser } from "@/lib/auth";
import { api } from "@/lib/api";
import { engine, type Game } from "@/lib/engine";
import Grid from "@/components/Grid";

export const dynamic = "force-dynamic";

// The library lives in Postgres (app_api): onboarding picks + every game played here, most recent first.
export default async function Library() {
  const user = await requireUser();
  const [lib, stats] = await Promise.all([api.library(user.sid, 60), api.stats(user.sid)]);
  const cards = lib.length ? await engine.items(lib.map((x) => x.item_id)) : [];
  const byId = new Map(cards.map((g) => [g.item_id, g] as const));
  const games = lib.map((x) => byId.get(x.item_id)).filter((g): g is Game => !!g);
  return (
    <div className="px-6 py-10 md:px-10">
      <h1 className="text-2xl font-bold text-white">Your library</h1>
      <p className="mt-1 text-sm text-zinc-400" data-testid="library-stats">
        {stats.games} games · {stats.plays} plays · {Math.round(stats.seconds_played / 60)} min streamed
      </p>
      <div className="mt-8"><Grid games={games} /></div>
    </div>
  );
}
