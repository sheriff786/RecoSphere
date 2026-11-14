import { requireUser } from "@/lib/auth";
import { engine } from "@/lib/engine";
import Onboarding from "@/components/Onboarding";
import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

export default async function Page() {
  const user = await requireUser();
  if (user.kind === "dataset") redirect("/");
  const games = await engine.popular(36);
  return (
    <div className="mx-auto max-w-6xl px-6 py-10 pb-32">
      <h1 className="text-3xl font-bold tracking-tight text-white">What do you like to play?</h1>
      <p className="mt-2 text-zinc-400">Pick at least 3. The engine has no history for you yet; these picks are your first signal.</p>
      <div className="mt-8"><Onboarding games={games} /></div>
    </div>
  );
}
