import type { Explanation } from "@/lib/engine";

const COLORS: Record<string, string> = {
  taste: "bg-violet-400", similar: "bg-fuchsia-400", trending: "bg-amber-400", popular: "bg-orange-400",
  fresh: "bg-cyan-400", publisher: "bg-emerald-400", format: "bg-sky-400",
};
// Users see how much a reason mattered, not SHAP numbers (raw weights: /admin "Why-this inspector").
const STRENGTH = { strong: { width: "100%", label: "Strong" }, medium: { width: "60%", label: "Medium" }, weak: { width: "28%", label: "Weak" } };

export default function WhyThis({ why, compact = false }: { why?: Explanation; compact?: boolean }) {
  if (!why) return null;
  // "Why this?" only for games really in the user's feed top 10; any other game page explains its score.
  const title = why.in_top10 === false ? "How this game scores for you" : "Why this?";
  return (
    <div className={`rounded-xl bg-white/[0.04] ring-1 ring-white/10 ${compact ? "p-3" : "p-5"}`} data-testid="why-this">
      <h3 className="text-sm font-semibold text-white" data-testid="why-title">{title}</h3>
      <ul className="mt-3 space-y-2">
        {why.reasons.map((r) => {
          const s = r.strength ? STRENGTH[r.strength] : undefined;
          return (
            <li key={r.text} className="text-sm text-zinc-200">
              <div className="flex items-center justify-between gap-3"><span>{r.text}</span>
                {s && !compact && <span className="text-[11px] text-zinc-400">{s.label}</span>}</div>
              {s && (
                <div className="mt-1 h-1.5 rounded-full bg-white/5" aria-label={`${s.label} reason`}>
                  <div className={`h-1.5 rounded-full ${COLORS[r.group] ?? "bg-zinc-400"}`} style={{ width: s.width }} />
                </div>
              )}
            </li>
          );
        })}
      </ul>
      {!compact && (
        <div className="mt-4 space-y-1 text-xs text-zinc-400">
          {why.because_you_played && <p>Closest game in your history: <span className="text-zinc-200">{why.because_you_played}</span></p>}
          {why.sources?.length ? <p>Found by: {why.sources.join(" + ")}</p> : null}
          {why.in_candidates === false && <p>Not among the games the engine picked for you right now (shown because it is similar to your games).</p>}
          <p className="pt-1 text-zinc-500">Strength = how much each reason pushed this game up for you.</p>
        </div>
      )}
    </div>
  );
}
