"use client";
export default function LogoutButton() {
  return (
    <button onClick={async () => { await fetch("/api/auth/logout", { method: "POST" }); window.location.href = "/login"; }}
            className="rounded-full px-3 py-1.5 text-sm text-zinc-400 ring-1 ring-white/10 hover:text-white">Sign out</button>
  );
}
