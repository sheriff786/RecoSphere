import Link from "next/link";
import { notFound } from "next/navigation";
import { requireUser } from "@/lib/auth";
import { engine } from "@/lib/engine";
import Cover from "@/components/Cover";
import Row from "@/components/Row";
import WhyThis from "@/components/WhyThis";
import NotInterested from "@/components/NotInterested";
import { hiddenIds } from "@/lib/hidden";

export const dynamic = "force-dynamic";

export default async function GamePage({ params, searchParams }: {
  params: Promise<{ id: string }>; searchParams: Promise<{ rid?: string }>;
}) {
  const user = await requireUser();
  const { id } = await params;
  const { rid } = await searchParams;
  const itemId = decodeURIComponent(id);
  const hidden = await hiddenIds(user.sid);   // "Not interested" games: out of "More like this" and the feed top 10
  const [items, similar, why, history] = await Promise.all([
    engine.items([itemId]),
    engine.similar(itemId, 16, user.engine_user_id, "game", hidden).catch(() => [] as Awaited<ReturnType<typeof engine.similar>>),
    engine.explain(user.engine_user_id, [itemId], "game", hidden).catch(() => null), engine.history(user.engine_user_id, 200),
  ]);
  const g = items[0];
  if (!g) notFound();
  const played = history.some((h) => h.item_id === g.item_id);
  const playable = g.item_type === "game";   // accessories / gift cards can be viewed, not streamed
  return (
    <div>
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 opacity-25 blur-3xl"><Cover title={g.title} className="h-full w-full" /></div>
        <div className="relative mx-auto grid max-w-7xl gap-10 px-6 py-12 md:grid-cols-[300px_1fr_340px] md:px-10">
          <div className="overflow-hidden rounded-2xl shadow-2xl ring-1 ring-white/20">
            <Cover title={g.title} platform={g.platforms[0]} isNew={g.is_new} className="block aspect-[3/4] w-full" />
          </div>
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-white md:text-4xl" data-testid="game-title">{g.title}</h1>
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              {/* a platform can also be a category ("PC"): dedupe, React keys must be unique */}
              {[...new Set([g.year && String(g.year), ...g.platforms, ...g.categories].filter(Boolean))].map((t) => (
                <span key={t as string} className="rounded-full bg-white/10 px-2.5 py-1 text-zinc-200">{t}</span>))}
              {g.is_new && <span className="rounded-full bg-cyan-400/20 px-2.5 py-1 text-cyan-200">New on RecoSphere</span>}
            </div>
            <div className="mt-4 text-sm text-zinc-400">
              {/* a trending rank means little for a game almost nobody plays: rank only from 20 players / 90 days */}
              {g.plays_90d >= 20 ? `${g.plays_90d.toLocaleString()} players in the last 90 days · trending #${g.trending_rank.toLocaleString()}`
                : g.plays_90d > 0 ? `${g.plays_90d.toLocaleString()} player${g.plays_90d === 1 ? "" : "s"} in the last 90 days`
                : "Brand new: no players yet"}
            </div>
            {g.description && <p className="mt-5 max-w-2xl leading-relaxed text-zinc-300">{g.description}</p>}
            <div className="mt-7 flex items-center gap-3">
              {playable ? (
                <>
                  <Link href={`/play/${encodeURIComponent(g.item_id)}${rid ? `?rid=${rid}` : ""}`} data-testid="play-button"
                        className="rounded-full bg-gradient-to-r from-violet-500 to-cyan-400 px-7 py-3 font-semibold text-black">
                    ▶ {played ? "Play again" : "Play now"}
                  </Link>
                  <span className="text-xs text-zinc-500">Streams instantly. No download.</span>
                </>
              ) : (
                <span data-testid="not-playable" className="rounded-full bg-white/10 px-4 py-2 text-sm text-zinc-300 ring-1 ring-white/15">
                  {g.item_type === "accessory" ? "Gear" : "Not a game"} · not streamable
                </span>
              )}
              {playable && !played && <NotInterested itemId={g.item_id} initiallyHidden={hidden.includes(g.item_id)} />}
            </div>
          </div>
          <WhyThis why={why?.items[g.item_id]} />
        </div>
      </section>
      <Row title="More like this" games={similar} subtitle="cosine similarity of two-tower item embeddings" />
    </div>
  );
}
