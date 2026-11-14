import { requireUser } from "@/lib/auth";
import { engine } from "@/lib/engine";
import { playStats } from "@/lib/db";
import Grid from "@/components/Grid";

export const dynamic = "force-dynamic";

export default async function Library() {
  const user = await requireUser();
  const [history, stats] = [await engine.history(user.engine_user_id, 60), playStats(user.id)];
  return (
    <div className="px-6 py-10 md:px-10">
      <h1 className="text-2xl font-bold text-white">Your library</h1>
      <p className="mt-1 text-sm text-zinc-400">
        {history.length} games in your history (what the engine&apos;s user tower sees: the last 20 drive your feed)
        {stats.sessions > 0 && <> · {stats.sessions} sessions here · {Math.round(stats.seconds / 60)} min streamed</>}
      </p>
      <div className="mt-8"><Grid games={history} /></div>
    </div>
  );
}
