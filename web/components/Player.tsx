"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Game } from "@/lib/engine";

function hue(s: string) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0; return h % 360; }

// Simulated cloud-game stream: a procedurally animated "game" on a canvas + a streaming HUD.
// The real work: POST /api/play on start (-> engine /event, so the feed changes) and on end (session length).
export default function Player({ game, rid }: { game: Game; rid?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const playId = useRef<number | null>(null);
  const started = useRef(Date.now());
  const [secs, setSecs] = useState(0);
  const [hud, setHud] = useState({ ping: 18, fps: 60, mbps: 24.0 });
  const [status, setStatus] = useState<"connecting" | "live" | "ended">("connecting");
  const [firstTime, setFirstTime] = useState<boolean | null>(null);

  // ONE start request per game, even when React runs this effect twice (StrictMode in dev: mount, cleanup,
  // mount). A second POST would log a second play and make the engine answer "played before".
  const startReq = useRef<{ item: string; done: Promise<{ play_id: number; first_time: boolean }> } | null>(null);

  useEffect(() => {
    let alive = true;
    if (startReq.current?.item !== game.item_id) {
      startReq.current = { item: game.item_id, done: fetch("/api/play", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ item_id: game.item_id, rid }) }).then((r) => r.json()) };
    }
    startReq.current.done.then((d) => { if (!alive) return; playId.current = d.play_id; setFirstTime(d.first_time); setStatus("live"); started.current = Date.now(); });
    const end = () => {
      if (playId.current == null) return;
      navigator.sendBeacon("/api/play", new Blob([JSON.stringify({ end: true, play_id: playId.current, seconds: (Date.now() - started.current) / 1000 })], { type: "application/json" }));
      playId.current = null;
    };
    window.addEventListener("pagehide", end);
    return () => { alive = false; end(); window.removeEventListener("pagehide", end); };
  }, [game.item_id, rid]);

  useEffect(() => {
    const t = setInterval(() => {
      setSecs(Math.floor((Date.now() - started.current) / 1000));
      setHud((h) => ({ ping: Math.max(9, Math.min(41, h.ping + Math.round((Math.random() - 0.5) * 6))),
                      fps: Math.random() < 0.92 ? 60 : 58 + Math.round(Math.random()), mbps: +(20 + Math.random() * 9).toFixed(1) }));
    }, 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const c = canvas.current!; const ctx = c.getContext("2d")!;
    const base = hue(game.title); let raf = 0; let t = 0;
    const stars = Array.from({ length: 140 }, () => ({ x: Math.random(), y: Math.random(), z: Math.random() * 0.8 + 0.2 }));
    const ship = { x: 0.5, y: 0.75 }; const keys = new Set<string>();
    const down = (e: KeyboardEvent) => keys.add(e.key); const up = (e: KeyboardEvent) => keys.delete(e.key);
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    const draw = () => {
      const w = (c.width = c.clientWidth * devicePixelRatio), h = (c.height = c.clientHeight * devicePixelRatio);
      t += 1;
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, `hsl(${base} 60% 12%)`); g.addColorStop(1, `hsl(${(base + 60) % 360} 70% 6%)`);
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
      for (const s of stars) {
        s.y += 0.002 * s.z * 2.5; if (s.y > 1) { s.y = 0; s.x = Math.random(); }
        ctx.fillStyle = `hsla(${(base + 180) % 360} 90% 80% / ${s.z})`;
        ctx.fillRect(s.x * w, s.y * h, 2.5 * s.z * devicePixelRatio, 7 * s.z * devicePixelRatio);
      }
      ctx.strokeStyle = `hsla(${(base + 30) % 360} 90% 60% / 0.35)`; ctx.lineWidth = 2 * devicePixelRatio;
      for (let i = 0; i < 6; i++) { // horizon grid
        const y = h * 0.6 + ((t * 2 + i * 40) % 240) * devicePixelRatio * 0.6;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      if (keys.has("ArrowLeft") || keys.has("a")) ship.x -= 0.012; if (keys.has("ArrowRight") || keys.has("d")) ship.x += 0.012;
      ship.x = Math.min(0.95, Math.max(0.05, ship.x + Math.sin(t / 40) * 0.002));
      const sx = ship.x * w, sy = ship.y * h, s = 26 * devicePixelRatio;
      ctx.fillStyle = `hsl(${(base + 180) % 360} 95% 65%)`;
      ctx.beginPath(); ctx.moveTo(sx, sy - s); ctx.lineTo(sx - s * 0.8, sy + s); ctx.lineTo(sx + s * 0.8, sy + s); ctx.closePath(); ctx.fill();
      ctx.fillStyle = `hsla(${(base + 40) % 360} 100% 60% / ${0.6 + 0.4 * Math.sin(t / 3)})`;
      ctx.fillRect(sx - 4 * devicePixelRatio, sy + s, 8 * devicePixelRatio, (12 + 8 * Math.random()) * devicePixelRatio);
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [game.title]);

  const mm = String(Math.floor(secs / 60)).padStart(2, "0"), ss = String(secs % 60).padStart(2, "0");
  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="relative overflow-hidden rounded-2xl ring-1 ring-white/10">
        <canvas ref={canvas} className="block aspect-video w-full bg-black" />
        <div className="absolute left-4 top-4 flex items-center gap-2 rounded-lg bg-black/60 px-3 py-1.5 font-mono text-xs text-zinc-200 backdrop-blur">
          <span className={`h-2 w-2 rounded-full ${status === "live" ? "animate-pulse bg-rose-500" : "bg-zinc-500"}`} />
          {status === "connecting" ? "CONNECTING…" : "LIVE"} · {mm}:{ss}
        </div>
        <div className="absolute right-4 top-4 rounded-lg bg-black/60 px-3 py-1.5 font-mono text-[11px] leading-5 text-emerald-300 backdrop-blur">
          ping {hud.ping} ms · {hud.fps} fps · {hud.mbps} Mbps · 1080p
        </div>
        <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
          <div>
            <div className="text-xs text-zinc-300">Now streaming</div>
            <div className="text-xl font-bold text-white" data-testid="player-title">{game.title}</div>
            <div className="text-xs text-zinc-400">Use ← → to steer</div>
          </div>
          <Link href="/?from=play" data-testid="end-session" className="rounded-full bg-white/90 px-5 py-2 text-sm font-semibold text-black hover:bg-white">End session</Link>
        </div>
      </div>
      {firstTime !== null && (
        <div className="mt-4 rounded-xl bg-violet-500/10 p-4 text-sm text-violet-200 ring-1 ring-violet-400/30" data-testid="play-toast">
          {firstTime ? <>Added to your history. The engine re-ranks your feed with this game <b>right now</b>. Go back home to see it change.</>
                     : <>You&apos;ve played this before, so your history is unchanged. Session time is still recorded.</>}
        </div>
      )}
    </div>
  );
}
