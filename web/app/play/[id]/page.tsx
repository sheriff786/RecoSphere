import { notFound, redirect } from "next/navigation";
import { requireUser } from "@/lib/auth";
import { engine } from "@/lib/engine";
import Player from "@/components/Player";

export const dynamic = "force-dynamic";

export default async function PlayPage({ params, searchParams }: {
  params: Promise<{ id: string }>; searchParams: Promise<{ rid?: string }>;
}) {
  await requireUser();
  const { id } = await params;
  const { rid } = await searchParams;
  const [g] = await engine.items([decodeURIComponent(id)]);
  if (!g) notFound();
  if (g.item_type !== "game") redirect(`/game/${encodeURIComponent(g.item_id)}`);   // only games stream
  return <Player game={g} rid={rid} />;
}
