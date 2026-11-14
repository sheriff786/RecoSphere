import { requireUser } from "@/lib/auth";
import { engine } from "@/lib/engine";
import Grid from "@/components/Grid";

export const dynamic = "force-dynamic";

export default async function Search({ searchParams }: { searchParams: Promise<{ q?: string }> }) {
  await requireUser();
  const q = ((await searchParams).q || "").trim();
  const results = q ? await engine.search(q, 36) : [];
  return (
    <div className="px-6 py-10 md:px-10">
      <h1 className="text-2xl font-bold text-white">{q ? <>Results for “{q}”</> : "Search"}</h1>
      <p className="mt-1 text-sm text-zinc-400">{q ? `${results.length} games · title match, then trending` : "Type in the search box above."}</p>
      <div className="mt-8">{results.length ? <Grid games={results} /> : q && <p className="text-zinc-400">No games found.</p>}</div>
    </div>
  );
}
