import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import { currentUser } from "@/lib/auth";

export const metadata: Metadata = {
  title: "RecoSphere Play",
  description: "Cloud game streaming, powered by the RecoSphere recommendation engine",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const user = await currentUser();
  return (
    <html lang="en">
      <body className="text-zinc-200 antialiased">
        <Navbar user={user ? { name: user.name, kind: user.kind, admin: user.admin } : null} />
        <main className="pb-24">{children}</main>
      </body>
    </html>
  );
}
