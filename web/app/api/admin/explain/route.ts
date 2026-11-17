import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { engine } from "@/lib/engine";

// Admin "Why-this inspector": the raw explanation (SHAP weights, ranks) the user-facing panel hides.
export async function GET(req: Request) {
  const u = await currentUser();
  if (!u?.admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const q = new URL(req.url).searchParams;
  const user = q.get("user_id")?.trim(), item = q.get("item_id")?.trim();
  if (!user || !item) return NextResponse.json({ error: "user_id and item_id are required" }, { status: 400 });
  try { return NextResponse.json(await engine.explain(user, [item])); }
  catch (e) { return NextResponse.json({ error: String(e) }, { status: 502 }); }
}
