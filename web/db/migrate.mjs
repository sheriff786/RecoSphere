// Forward-only SQL migration runner (DB_SCHEMA.md). Applies web/db/migrations/NNN_*.sql in name order, each
// in its own transaction, and records it in schema_migrations with a sha256 checksum. An applied file that
// changed afterwards is an error: add a new file instead of editing an old one.
//
//   node db/migrate.mjs            apply pending migrations   (DATABASE_URL_DIRECT, else DATABASE_URL)
//   node db/migrate.mjs --status   list applied / pending, change nothing
//
// Use Neon's DIRECT host for migrations: the -pooler host is PgBouncer in transaction mode, where session
// state such as advisory locks does not hold.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

const DIR = join(dirname(fileURLToPath(import.meta.url)), "migrations");
const LOCK = 727_011;   // advisory lock id: one runner at a time

export async function migrate(url, { statusOnly = false, log = console.log } = {}) {
  if (!url) throw new Error("set DATABASE_URL_DIRECT (or DATABASE_URL)");
  const client = new pg.Client({ connectionString: url });
  await client.connect();
  try {
    await client.query("SELECT pg_advisory_lock($1)", [LOCK]);
    await client.query(`CREATE TABLE IF NOT EXISTS schema_migrations (
      version text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())`);
    const applied = new Map((await client.query("SELECT version, checksum FROM schema_migrations")).rows
      .map((r) => [r.version, r.checksum]));
    const files = readdirSync(DIR).filter((f) => /^\d{3}_.+\.sql$/.test(f)).sort();
    const done = [];
    for (const f of files) {
      const sql = readFileSync(join(DIR, f), "utf8");
      // checksum over LF line endings: a Windows checkout (CRLF) of the same SQL must not count as "changed"
      const sum = createHash("sha256").update(sql.replace(/\r\n/g, "\n")).digest("hex");
      if (applied.has(f)) {
        if (applied.get(f) !== sum) throw new Error(`${f} was changed after it was applied: add a new migration instead`);
        continue;
      }
      if (statusOnly) { log(`pending  ${f}`); continue; }
      await client.query("BEGIN");
      try {
        await client.query(sql);
        await client.query("INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)", [f, sum]);
        await client.query("COMMIT");
      } catch (e) {
        await client.query("ROLLBACK");
        throw new Error(`${f} failed, rolled back: ${e.message}`);
      }
      done.push(f);
      log(`applied  ${f}`);
    }
    for (const v of applied.keys()) if (statusOnly) log(`applied  ${v}`);
    if (!statusOnly && !done.length) log("up to date");
    return done;
  } finally {
    await client.query("SELECT pg_advisory_unlock($1)", [LOCK]).catch(() => {});
    await client.end();
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  migrate(process.env.DATABASE_URL_DIRECT || process.env.DATABASE_URL, { statusOnly: process.argv.includes("--status") })
    .catch((e) => { console.error("migrate:", e.message); process.exit(1); });
}
