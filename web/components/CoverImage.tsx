"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";

// The product's real picture (Amazon CDN, data/item_media.py). If it cannot load, the generated cover (children)
// takes its place. An image can fail before React hydrates, when onError has no listener yet, so the mount effect
// checks the finished-but-empty case too.
export default function CoverImage({ src, alt, isNew, className = "", children }: {
  src: string; alt: string; isNew?: boolean; className?: string; children: ReactNode;
}) {
  const [failed, setFailed] = useState(false);
  const ref = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const img = ref.current;
    if (img && img.complete && img.naturalWidth === 0) setFailed(true);
  }, []);
  if (failed) return <>{children}</>;
  return (
    <span className={`relative block overflow-hidden bg-zinc-900 ${className}`}>
      {/* eslint-disable-next-line @next/next/no-img-element -- remote CDN image, sized by CSS; no Next optimizer */}
      <img ref={ref} src={src} alt={alt} data-cover="" loading="lazy" decoding="async" referrerPolicy="no-referrer"
           onError={() => setFailed(true)} className="absolute inset-0 h-full w-full object-cover" />
      {isNew && (
        <span className="absolute right-2 top-2 rounded bg-cyan-400 px-1.5 py-0.5 text-[10px] font-extrabold text-[#04121a]">NEW</span>
      )}
    </span>
  );
}
