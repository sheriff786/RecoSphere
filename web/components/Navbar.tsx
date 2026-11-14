import Link from "next/link";
import LogoutButton from "./LogoutButton";

export default function Navbar({ user }: { user: { name: string; kind: string; admin: boolean } | null }) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/5 bg-[#07070c]/80 backdrop-blur-xl">
      <div className="flex h-16 items-center gap-6 px-6 md:px-10">
        <Link href="/" className="flex items-center gap-2 font-bold tracking-tight text-white">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-violet-500 to-cyan-400 text-sm text-black">▶</span>
          <span className="text-lg">RecoSphere<span className="text-violet-400"> Play</span></span>
        </Link>
        {user && (
          <nav className="hidden items-center gap-5 text-sm text-zinc-400 md:flex">
            <Link href="/" className="hover:text-white">Home</Link>
            <Link href="/library" className="hover:text-white">Library</Link>
            {user.admin && <Link href="/admin" className="hover:text-white">Engine admin</Link>}
          </nav>
        )}
        <div className="ml-auto flex items-center gap-3">
          {user && (
            <form action="/search" className="hidden sm:block">
              <input name="q" placeholder="Search games…" aria-label="Search games"
                     className="w-56 rounded-full bg-white/5 px-4 py-2 text-sm text-white placeholder-zinc-500 ring-1 ring-white/10 outline-none focus:ring-violet-400/60 md:w-72" />
            </form>
          )}
          {user ? (
            <div className="flex items-center gap-3">
              <span className="hidden text-sm text-zinc-300 sm:inline">
                {user.name}{user.kind === "dataset" && <span className="ml-1 rounded bg-cyan-400/15 px-1.5 py-0.5 text-[10px] text-cyan-300">real player</span>}
              </span>
              <LogoutButton />
            </div>
          ) : (
            <Link href="/login" className="rounded-full bg-white px-4 py-1.5 text-sm font-semibold text-black">Sign in</Link>
          )}
        </div>
      </div>
    </header>
  );
}
