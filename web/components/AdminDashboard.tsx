"use client";
import { useEffect, useState } from "react";
import type { AdminSummary } from "@/lib/engine";

const pct = (x: number | null | undefined) => (x == null ? "–" : `${(100 * x).toFixed(2)}%`);

function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="rounded-xl bg-white/[0.04] p-4 ring-1 ring-white/10">
      <div className="text-xs uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="mt-1 truncate text-xl font-semibold text-white">{value}</div>
      {sub && <div className="mt-0.5 truncate text-xs text-zinc-400">{sub}</div>}
    </div>
  );
}

export default function AdminDashboard() {
  const [d, setD] = useState<AdminSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const load = () => fetch("/api/admin/summary").then(async (r) => (r.ok ? setD(await r.json()) : setErr(`HTTP ${r.status}`))).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, []);
  async function rollback() {
    if (!confirm("Roll production back to the previous model? Every API worker hot-swaps within seconds.")) return;
    const r = await fetch("/api/admin/rollback", { method: "POST" });
    const j = await r.json();
    setMsg(r.ok ? `Rolled back to ${j.rolled_back_to}` : `Rollback failed: ${j.error}`);
    load();
  }
  if (err) return <p className="text-rose-300">Engine admin API error: {err}</p>;
  if (!d) return <div className="skeleton h-40 rounded-xl" />;
  const h = d.health, arms = Object.entries(d.live.arms);
  const exp = h.experiment;
  return (
    <div className="space-y-8" data-testid="admin">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Production model" value={<span data-testid="admin-model">{h.model_run}</span>} sub={`loaded ${h.model_loaded_at} · emb ${h.emb_version}`} />
        <Stat label="Feature store" value={`${h.users.toLocaleString()} users`} sub={`${h.store} · ${h.items.toLocaleString()} games`} />
        <Stat label="Hot swaps" value={h.reloads} sub={h.last_reload_error ? `last error: ${h.last_reload_error}` : "no reload errors"} />
        <Stat label="Errors" value={Object.values(d.live.errors).reduce((a, b) => a + b, 0)} sub={Object.entries(d.live.errors).map(([k, v]) => `${k} ${v}`).join(" · ")} />
      </div>

      <section className="rounded-xl bg-white/[0.03] p-5 ring-1 ring-white/10">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-semibold text-white">A/B experiment</h2>
          <span className="text-sm text-zinc-400">{exp ? `${exp.name} · salt ${exp.salt}` : "no experiment running (all traffic = control)"}</span>
        </div>
        {h.experiment_warning && <p className="mt-2 text-sm text-amber-300">⚠ {h.experiment_warning}</p>}
        {h.experiment_error && <p className="mt-2 text-sm text-rose-300">Experiment rejected: {h.experiment_error}</p>}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wider text-zinc-500">
              <tr><th className="py-2">Arm</th><th>Model / policy</th><th>Traffic</th><th>Requests</th><th>Personalized</th><th>Clicks</th><th>CTR</th><th>Plays</th><th>Play rate</th><th>p95 latency</th></tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {arms.map(([name, a]) => {
                const cfg = exp?.arms.find((x) => x.name === name);
                return (
                  <tr key={name} className="text-zinc-200">
                    <td className="py-2 font-medium text-white">{name}</td><td className="text-zinc-400">{cfg?.model ?? "production"}</td>
                    <td>{cfg ? `${Math.round(cfg.traffic * 100)}%` : "100%"}</td><td>{a.requests}</td><td>{a.personalized}</td>
                    <td>{a.clicks}</td><td>{pct(a.ctr)}</td><td>{a.plays}</td><td>{pct(a.play_rate)}</td>
                    <td>{a.p95_ms_le == null ? "–" : `≤ ${a.p95_ms_le} ms`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-zinc-500">Live counters for monitoring only. Ship/no-ship decisions come from <code>python -m ab.analyze</code> (SRM, power, CUPED, guardrails), never from this table.</p>
      </section>

      <section className="rounded-xl bg-white/[0.03] p-5 ring-1 ring-white/10">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Production history</h2>
          <button onClick={rollback} data-testid="rollback" className="rounded-full bg-rose-500/15 px-4 py-1.5 text-sm font-medium text-rose-200 ring-1 ring-rose-400/40 hover:bg-rose-500/25">Roll back</button>
        </div>
        {msg && <p className="mt-2 text-sm text-violet-200">{msg}</p>}
        <table className="mt-4 w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wider text-zinc-500"><tr><th className="py-2">Run</th><th>Reason</th><th>Promoted</th><th>NDCG@10</th><th>Recall@100</th></tr></thead>
          <tbody className="divide-y divide-white/5">
            {d.production_history.map((p, i) => (
              <tr key={i} className={p.run_id === h.model_run ? "text-white" : "text-zinc-400"}>
                <td className="py-2 font-mono">{p.run_id}{p.run_id === h.model_run && i === 0 && <span className="ml-2 rounded bg-emerald-400/20 px-1.5 text-[10px] text-emerald-300">live</span>}</td>
                <td>{p.reason}</td><td>{p.promoted_at?.slice(0, 16).replace("T", " ")}</td>
                <td>{p.metrics?.["ranker/all/ndcg@10"]?.toFixed(4) ?? "–"}</td><td>{p.metrics?.["ranker/all/recall@100"]?.toFixed(4) ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <p className="text-xs text-zinc-500">Deeper views: Grafana (localhost:3001 in docker compose), Prometheus alerts (localhost:9090/alerts), MLflow UI.</p>
    </div>
  );
}
