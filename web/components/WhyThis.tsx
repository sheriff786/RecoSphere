import type { Explanation } from "@/lib/engine";

const COLORS: Record<string, string> = {
  taste: "bg-violet-400", similar: "bg-fuchsia-400", trending: "bg-amber-400", popular: "bg-orange-400",
  fresh: "bg-cyan-400", publisher: "bg-emerald-400", format: "bg-sky-400",
};

export default function WhyThis({ why, compact = false }: { why?: Explanation; compact?: boolean }) {
  if (!why) return null;
  const max = Math.max(...why.reasons.map((r) => r.weight ?? 0), 1e-9);
  return (
    <div className={`rounded-xl bg-white/[0.04] ring-1 ring-white/10 ${compact ? "p-3" : "p-5"}`} data-testid="why-this">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Why this?</h3>
        {why.rank && <span className="text-xs text-zinc-400">#{why.rank} of {why.n_candidates} candidates</span>}
      </div>
      <ul className="mt-3 space-y-2">
        {why.reasons.map((r) => (
          <li key={r.text} className="text-sm text-zinc-200">
            <div className="flex items-center justify-between gap-3"><span>{r.text}</span>
              {r.weight !== undefined && !compact && <span className="font-mono text-[11px] text-zinc-500">+{r.weight.toFixed(2)}</span>}</div>
            {r.weight !== undefined && (
              <div className="mt-1 h-1.5 rounded-full bg-white/5">
                <div className={`h-1.5 rounded-full ${COLORS[r.group] ?? "bg-zinc-400"}`} style={{ width: `${Math.max(8, (100 * r.weight) / max)}%` }} />
              </div>
            )}
          </li>
        ))}
      </ul>
      {!compact && (
        <div className="mt-4 space-y-1 text-xs text-zinc-400">
          {why.because_you_played && <p>Closest game in your history: <span className="text-zinc-200">{why.because_you_played}</span></p>}
          {why.sources?.length ? <p>Found by: {why.sources.join(" + ")}</p> : null}
          {why.in_candidates === false && <p>Not in your current candidate set (shown because it is similar to your games).</p>}
          <p className="pt-1 text-zinc-500">Weights = the ranker&apos;s feature contributions (SHAP) for this game.</p>
        </div>
      )}
    </div>
  );
}
