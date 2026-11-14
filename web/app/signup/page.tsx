import AuthForm from "@/components/AuthForm";

export default function Signup() {
  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <h1 className="text-3xl font-bold tracking-tight text-white">Create your account</h1>
      <p className="mt-2 text-zinc-400">Pick a few games you like next, and your feed starts personalizing right away.</p>
      <div className="mt-8"><AuthForm mode="signup" /></div>
    </div>
  );
}
