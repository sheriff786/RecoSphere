import CoverImage from "./CoverImage";

// A game's cover: its real picture when the engine sends one (`image`, data/item_media.py), else deterministic
// generated box art: hash(title) -> two hues + one of 4 patterns (works offline, the same game always gets the
// same cover). The generated cover is also the fallback when the picture fails to load.
function hash(s: string) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function wrap(title: string, max = 13, lines = 3): string[] {
  const words = title.replace(/\s+/g, " ").trim().split(" ");
  const out: string[] = [];
  let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length <= max) cur = (cur + " " + w).trim();
    else { if (cur) out.push(cur); cur = w.length > max ? w.slice(0, max - 1) + "…" : w; }
    if (out.length === lines) break;
  }
  if (out.length < lines && cur) out.push(cur);
  if (out.length === lines && words.join(" ").length > out.join(" ").length + 1) out[lines - 1] = out[lines - 1].replace(/.?$/, "…");
  return out.slice(0, lines);
}

export default function Cover({ title, platform, isNew, image, className = "" }: {
  title: string; platform?: string; isNew?: boolean; image?: string | null; className?: string;
}) {
  const generated = <GeneratedCover title={title} platform={platform} isNew={isNew} className={className} />;
  if (!image) return generated;
  return <CoverImage src={image} alt={title} isNew={isNew} className={className}>{generated}</CoverImage>;
}

function GeneratedCover({ title, platform, isNew, className = "" }: {
  title: string; platform?: string; isNew?: boolean; className?: string;
}) {
  const h = hash(title || "?");
  const h1 = h % 360, h2 = (h1 + 40 + ((h >> 9) % 140)) % 360;
  const pattern = (h >> 3) % 4;
  const id = `c${h.toString(36)}`;
  const lines = wrap(title || "Untitled");
  return (
    <svg viewBox="0 0 300 400" className={className} role="img" aria-label={title} preserveAspectRatio="xMidYMid slice">
      <defs>
        <linearGradient id={`${id}g`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={`hsl(${h1} 75% 42%)`} />
          <stop offset="1" stopColor={`hsl(${h2} 80% 18%)`} />
        </linearGradient>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0.45" stopColor="#000" stopOpacity="0" />
          <stop offset="1" stopColor="#000" stopOpacity="0.75" />
        </linearGradient>
      </defs>
      <rect width="300" height="400" fill={`url(#${id}g)`} />
      <g opacity={pattern === 3 ? 0.85 : 0.22} stroke={`hsl(${h2} 90% 75%)`} fill="none" strokeWidth="2">
        {pattern === 0 && Array.from({ length: 14 }, (_, i) => <line key={i} x1={-100 + i * 40} y1="0" x2={100 + i * 40} y2="400" />)}
        {pattern === 1 && Array.from({ length: 9 }, (_, i) => <circle key={i} cx={230} cy={110} r={20 + i * 28} />)}
        {pattern === 2 && Array.from({ length: 8 }, (_, r) => Array.from({ length: 6 }, (_, c) =>
          <circle key={`${r}-${c}`} cx={25 + c * 50} cy={25 + r * 50} r={(h >> (r + c)) % 3 === 0 ? 9 : 3} fill={`hsl(${h2} 90% 75%)`} />))}
        {pattern === 3 && <>
          <polygon points="0,300 90,170 170,270 230,190 300,260 300,400 0,400" fill={`hsl(${h2} 70% 12%)`} stroke="none" />
          <polygon points="0,330 120,230 300,320 300,400 0,400" fill={`hsl(${h1} 60% 8%)`} stroke="none" />
          <circle cx="225" cy="95" r="38" fill={`hsl(${h1} 95% 70%)`} stroke="none" opacity="0.6" />
        </>}
      </g>
      <rect width="300" height="400" fill={`url(#${id}s)`} />
      {platform && <g>
        <rect x="14" y="14" rx="6" width={platform.length * 9 + 18} height="26" fill="#000" opacity="0.55" />
        <text x="23" y="32" fill="#fff" fontSize="14" fontWeight="700" fontFamily="system-ui, sans-serif">{platform}</text>
      </g>}
      {isNew && <g>
        <rect x="222" y="14" rx="6" width="64" height="26" fill="#22d3ee" />
        <text x="235" y="32" fill="#04121a" fontSize="14" fontWeight="800" fontFamily="system-ui, sans-serif">NEW</text>
      </g>}
      {lines.map((l, i) => (
        <text key={i} x="18" y={400 - 22 - (lines.length - 1 - i) * 34} fill="#fff" fontSize="30" fontWeight="800"
              fontFamily="system-ui, -apple-system, Segoe UI, sans-serif" letterSpacing="-0.5">{l}</text>
      ))}
    </svg>
  );
}
