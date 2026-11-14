import { notFound } from "next/navigation";
import { requireUser } from "@/lib/auth";
import { isAdmin } from "@/lib/db";
import AdminDashboard from "@/components/AdminDashboard";

export const dynamic = "force-dynamic";

export default async function Admin() {
  const u = await requireUser();
  if (!isAdmin(u)) notFound();
  return (
    <div className="mx-auto max-w-7xl px-6 py-10 md:px-10">
      <h1 className="text-2xl font-bold text-white">Engine admin</h1>
      <p className="mt-1 text-sm text-zinc-400">Live view of the recommendation engine behind RecoSphere Play. Refreshes every 5 s.</p>
      <div className="mt-8"><AdminDashboard /></div>
    </div>
  );
}
