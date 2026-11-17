import Link from "next/link";
import { redirect } from "next/navigation";
import { requireUser } from "@/lib/auth";
import { engine, type Game } from "@/lib/engine";
import Cover from "@/components/Cover";
import Row from "@/components/Row";
import WhyThis from "@/components/WhyThis";
import { replayLibrarySafe } from "@/lib/history";

export const dynamic = "force-dynamic";

export default async function Home() {
  const user = await requireUser();
  if (user.kind === "app" && !user.onboarded) redirect("/onboarding");
  const uid = user.engine_user_id;
  const t0 = Date.now();
  const [recs, history, trending, fresh] = await Promise.all([
    engine.recommend(uid, 20), engine.history(uid, 12), engine.popular(20), engine.fresh(uid, 20),
  ]);
  // An onboarded app user with no engine history: the engine restarted (its history is a cache). Put the saved
  // library back from Postgres and render again (P1 phase F); costs nothing on the normal path. A redirect, not a
  // second fetch here: Next memoizes identical GET fetches within one render, so they would return the old feed.
  // No loop: on the next request the history is there, or the replay records nothing new (dedup).
  if (user.kind === "app" && history.length === 0) {
    const r = await replayLibrarySafe(user);
    if (r && r.recorded > 0) redirect("/");
  }
  const last = history[0];
  const [why, because] = await Promise.all([
    engine.explain(uid, recs.items.map((g) => g.item_id)).catch(() => null),
    last ? engine.similar(last.item_id, 16, uid) : Promise.resolve([] as Game[]),
  ]);
  const ms = Date.now() - t0;
  const hero = recs.items[0];
  const seen = new Set(history.map((g) => g.item_id));
  const personalized = recs.source === "personalized";

  return (
    <div>
      {hero && (
        <section className="relative overflow-hidden border-b border-white/5">
          <div className="absolute inset-0 opacity-30 blur-3xl"><Cover title={hero.title} className="h-full w-full" /></div>
          <div className="relative mx-auto grid max-w-7xl items-center gap-10 px-6 py-12 md:grid-cols-[260px_1fr_340px] md:px-10">
            <Link href={`/game/${encodeURIComponent(hero.item_id)}?rid=${recs.request_id}`}
                  className="hidden overflow-hidden rounded-2xl shadow-2xl ring-1 ring-white/20 md:block">
              <Cover title={hero.title} platform={hero.platforms[0]} isNew={hero.is_new} className="block aspect-[3/4] w-full" />
            </Link>
            <div>
              <div className="text-xs font-semibold tracking-widest text-violet-300">{personalized ? "TOP PICK FOR YOU" : "POPULAR RIGHT NOW"}</div>
              <h1 className="mt-2 text-3xl font-extrabold tracking-tight text-white md:text-5xl" data-testid="hero-title">{hero.title}</h1>
              <div className="mt-3 text-sm text-zinc-300">{[hero.year, ...hero.platforms, ...hero.categories.slice(0, 2)].filter(Boolean).join(" · ")}</div>
              {hero.description && <p className="mt-4 line-clamp-3 max-w-2xl text-zinc-300">{hero.description}</p>}
              <div className="mt-6 flex gap-3">
                <Link href={`/play/${encodeURIComponent(hero.item_id)}?rid=${recs.request_id}`} data-testid="hero-play"
                      className="rounded-full bg-white px-6 py-3 font-semibold text-black hover:bg-zinc-200">▶ Play now</Link>
                <Link href={`/game/${encodeURIComponent(hero.item_id)}?rid=${recs.request_id}`}
                      className="rounded-full bg-white/10 px-6 py-3 font-semibold text-white ring-1 ring-white/20 hover:bg-white/15">Details</Link>
              </div>
            </div>
            <div className="hidden md:block"><WhyThis why={why?.items[hero.item_id]} /></div>
          </div>
        </section>
      )}

      {!personalized && (
        <div className="mx-6 mt-8 rounded-xl bg-amber-400/10 p-4 text-sm text-amber-200 ring-1 ring-amber-400/30 md:mx-10">
          We don&apos;t know your taste yet, so this is what&apos;s popular. Play anything and your feed personalizes instantly.
        </div>
      )}

      <Row title={personalized ? "Top picks for you" : "Popular picks"} games={recs.items.slice(1)} rid={recs.request_id} why={why?.items}
           badge={personalized ? "personalized" : undefined} subtitle="ranked by the LightGBM ranker over two-tower + trending candidates" />
      {last && <Row title={`Because you played ${last.title}`} games={because.filter((g) => !seen.has(g.item_id))}
                    subtitle="nearest games in the two-tower embedding space" />}
      <Row title="Fresh on RecoSphere" games={fresh.items}
           subtitle={`new to the catalog (not by release date) · ${fresh.ranked_by === "taste" ? "sorted by your taste (two-tower)" : "newest first"}`} />
      <Row title="Trending now" games={trending.filter((g) => !seen.has(g.item_id))} />
      {history.length > 0 && <Row title="Continue playing" games={history} />}

      <footer className="mx-6 mt-14 flex flex-wrap gap-x-6 gap-y-1 rounded-xl bg-white/[0.03] px-4 py-3 font-mono text-[11px] text-zinc-500 ring-1 ring-white/5 md:mx-10" data-testid="engine-strip">
        <span>model {recs.model_run}</span><span>source {recs.source}</span>
        <span>experiment {recs.experiment ?? "none"} / arm {recs.arm}</span><span>page data {ms} ms</span>
        <span>request {recs.request_id.slice(0, 8)}</span>
      </footer>
    </div>
  );
}
