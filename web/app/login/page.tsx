import AuthForm from "@/components/AuthForm";
import DemoPlayers from "@/components/DemoPlayers";
import { engine } from "@/lib/engine";

export const dynamic = "force-dynamic";

export default async function Login() {
  const players = await engine.demoUsers(9).catch(() => []);
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-14 lg:grid-cols-[380px_1fr]">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Welcome back</h1>
        <p className="mt-2 text-zinc-400">Stream thousands of games. Your feed learns from every session.</p>
        <div className="mt-8"><AuthForm mode="login" /></div>
      </div>
      <div>
        <h2 className="text-lg font-semibold text-white">…or explore as a real player</h2>
        <p className="mt-1 text-sm text-zinc-400">
          These are real (anonymised) players from the Amazon Reviews 2023 data the engine was trained on. Their feed is
          personalized from their actual history.
        </p>
        <div className="mt-5">
          {players.length ? <DemoPlayers users={players} /> :
            <p className="text-sm text-rose-300">The recommendation engine is not reachable ({process.env.ENGINE_URL || "http://127.0.0.1:8000"}).</p>}
        </div>
      </div>
    </div>
  );
}
