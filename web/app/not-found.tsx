import Link from "next/link";
export default function NotFound() {
  return <div className="px-10 py-24 text-center"><h1 className="text-3xl font-bold text-white">Game not found</h1>
    <Link href="/" className="mt-4 inline-block text-violet-300 hover:underline">Back home</Link></div>;
}
