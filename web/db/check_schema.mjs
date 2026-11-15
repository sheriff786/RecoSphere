// Exercises every rule of web/db/migrations/001_init.sql (DB_SCHEMA.md) with the scenarios of
// web/db/test_schema.sql, inside ONE transaction that is rolled back at the end: safe on Neon, no rows left.
// Exit code 1 if any check fails.
//
//   node db/check_schema.mjs        (DATABASE_URL_DIRECT, else DATABASE_URL)   ·   make db-check
import { fileURLToPath } from "node:url";
import pg from "pg";

const U = "11111111-1111-1111-1111-111111111111";   // Sheriff, as in test_schema.sql
const EVT = "aaaaaaaa-0000-0000-0000-000000000001";

export async function checkSchema(url, log = console.log) {
  const c = new pg.Client({ connectionString: url });
  await c.connect();
  const results = [];
  const ok = (name, pass, detail = "") => { results.push(pass); log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`); };
  const q = (sql, params) => c.query(sql, params);
  async function rejects(name, code, sql, params) {   // must fail with this SQLSTATE; a savepoint keeps the tx usable
    await q("SAVEPOINT s");
    try { await q(sql, params); await q("ROLLBACK TO SAVEPOINT s"); ok(`rejects: ${name}`, false, "was accepted"); }
    catch (e) { await q("ROLLBACK TO SAVEPOINT s"); ok(`rejects: ${name}`, e.code === code, `${e.code} ${e.constraint ?? ""}`.trim()); }
  }
  try {
    await q("BEGIN");
    const tables = (await q(`SELECT array_agg(table_name::text ORDER BY table_name) AS t FROM information_schema.tables
                             WHERE table_schema = 'public' AND table_name <> 'schema_migrations'`)).rows[0].t ?? [];
    ok("7 tables exist", ["admin_audit_log", "games", "item_feedback", "library", "play_sessions", "sessions", "users"]
      .every((t) => tables.includes(t)), tables.join(", "));

    // --- happy path (test_schema.sql)
    await q(`INSERT INTO games (item_id, title, title_family, item_type, platforms, publisher) VALUES
      ('B07R4C1G6Z','Dragon Ball FighterZ - PlayStation 4','dragon ball fighterz','game','{PS4}','Bandai Namco'),
      ('B0FF7R0001','Final Fantasy VII Remake - PlayStation 4 Deluxe Edition','final fantasy vii remake','game','{PS4}','Square Enix'),
      ('B0FF7R0002','FINAL FANTASY VII REMAKE (PS4)','final fantasy vii remake','game','{PS4}','Square Enix'),
      ('B0MOUSE001','Logitech G502 HERO Gaming Mouse','logitech g502 hero','accessory','{PC}','Logitech')`);
    await q(`INSERT INTO users (id, email, password_hash, display_name) VALUES ($1, 'Sheriff@Example.com', '$2b$10$x', 'Sheriff')`, [U]);
    const kind = (await q(`SELECT kind FROM users WHERE id = $1`, [U])).rows[0].kind;
    ok("kind defaults to 'registered'", kind === "registered", kind);
    await q(`INSERT INTO users (kind, dataset_user_id, display_name) VALUES ('demo', 'AGEYTHU7HPQ7VA', 'Player AGEYTHU7')`);
    await q(`INSERT INTO library (user_id, item_id, source, added_at) VALUES ($1, 'B0FF7R0001', 'onboarding', now() - interval '1 day')`, [U]);

    // one play = one transaction: session row + library upsert; a retried request (same client_event_id) is a no-op
    async function play(item, eventId, secs) {
      const ins = await q(`INSERT INTO play_sessions (user_id, item_id, client_event_id, seconds_played, request_id, surface)
                           VALUES ($1, $2, $3, $4, '978e1bcb', 'home:top_picks') ON CONFLICT (client_event_id) DO NOTHING`, [U, item, eventId, secs]);
      if (ins.rowCount === 1) {
        await q(`INSERT INTO library (user_id, item_id, source, last_played_at, play_count, seconds_played)
                 VALUES ($1, $2, 'play', now(), 1, $3)
                 ON CONFLICT (user_id, item_id) DO UPDATE SET last_played_at = now(),
                   play_count = library.play_count + 1, seconds_played = library.seconds_played + EXCLUDED.seconds_played`, [U, item, secs]);
      }
      return ins.rowCount;
    }
    const first = await play("B07R4C1G6Z", EVT, 120);
    const retry = await play("B07R4C1G6Z", EVT, 120);
    const lib = (await q(`SELECT play_count, seconds_played FROM library WHERE user_id = $1 AND item_id = 'B07R4C1G6Z'`, [U])).rows[0];
    ok("retried play is a no-op (ON CONFLICT client_event_id)", first === 1 && retry === 0 && lib.play_count === 1 && Number(lib.seconds_played) === 120,
       `inserted ${first} then ${retry}; library ${lib.play_count} play, ${lib.seconds_played} s`);
    await q(`INSERT INTO item_feedback (user_id, item_id, kind) VALUES ($1, 'B0MOUSE001', 'not_interested')`, [U]);
    await q(`INSERT INTO sessions (user_id, expires_at) VALUES ($1, now() + interval '7 days')`, [U]);
    await q(`INSERT INTO admin_audit_log (actor_id, action, target, details)
             VALUES ($1, 'model.rollback', '20261111-171751', '{"from":"20261114-090000"}')`, [U]);

    // --- constraints that must reject bad data
    await rejects("duplicate email differing only in case (citext)", "23505",
      `INSERT INTO users (email, password_hash, display_name) VALUES ('sheriff@example.com', '$2b$10$y', 'Dup')`);
    await rejects("registered user without password", "23514", `INSERT INTO users (email, display_name) VALUES ('a@b.com', 'NoPass')`);
    await rejects("demo user without dataset id", "23514", `INSERT INTO users (kind, display_name) VALUES ('demo', 'Ghost')`);
    await rejects("duplicate dataset id", "23505",
      `INSERT INTO users (kind, dataset_user_id, display_name) VALUES ('demo', 'AGEYTHU7HPQ7VA', 'Twin')`);
    await rejects("play for a game not in the catalog", "23503",
      `INSERT INTO play_sessions (user_id, item_id, client_event_id) VALUES ($1, 'NOPE', 'aaaaaaaa-0000-0000-0000-000000000009')`, [U]);
    await rejects("retried play request without ON CONFLICT", "23505",
      `INSERT INTO play_sessions (user_id, item_id, client_event_id) VALUES ($1, 'B07R4C1G6Z', $2)`, [U, EVT]);
    await rejects("negative seconds", "23514",
      `INSERT INTO play_sessions (user_id, item_id, client_event_id, seconds_played) VALUES ($1, 'B07R4C1G6Z', 'aaaaaaaa-0000-0000-0000-000000000003', -5)`, [U]);
    await rejects("session ending before it starts (ended_at)", "23514",
      `INSERT INTO play_sessions (user_id, item_id, client_event_id, started_at, ended_at)
       VALUES ($1, 'B07R4C1G6Z', 'aaaaaaaa-0000-0000-0000-000000000004', now(), now() - interval '1 minute')`, [U]);
    await rejects("login session expiring before it starts", "23514",
      `INSERT INTO sessions (user_id, expires_at) VALUES ($1, now() - interval '1 hour')`, [U]);
    await rejects("unknown item_type value", "22P02",
      `INSERT INTO games (item_id, title, title_family, item_type) VALUES ('X1', 'x', 'x', 'toaster')`);

    // --- queries the app and engine will run
    const hist = (await q(`SELECT item_id FROM library WHERE user_id = $1 ORDER BY COALESCE(last_played_at, added_at) DESC`, [U])).rows.map((r) => r.item_id);
    ok("history sent to the engine: most recent first", hist.join(",") === "B07R4C1G6Z,B0FF7R0001", hist.join(", "));
    const shown = (await q(`SELECT g.item_id FROM games g
       WHERE g.item_id IN ('B0FF7R0002','B07R4C1G6Z','B0MOUSE001') AND g.item_type IN ('game','dlc') AND g.is_active
         AND g.title_family NOT IN (SELECT g2.title_family FROM library l JOIN games g2 USING (item_id) WHERE l.user_id = $1)
         AND g.item_id NOT IN (SELECT item_id FROM item_feedback WHERE user_id = $1 AND kind IN ('dislike','not_interested'))`, [U])).rows;
    ok("dedup by title family + not_interested + accessory filter -> nothing left", shown.length === 0,
       "B0FF7R0002 = family of the owned Deluxe edition, B07R4C1G6Z played, B0MOUSE001 accessory + not_interested");
    const fuzzy = (await q(`SELECT item_id, round(similarity(title, 'fantsy remake')::numeric, 2) AS sim FROM games
                            WHERE title % 'fantsy remake' ORDER BY sim DESC`)).rows;
    ok("fuzzy search 'fantsy remake' (typo) finds FF7 Remake", fuzzy.length > 0 && fuzzy.every((r) => r.item_id.startsWith("B0FF7R")),
       fuzzy.map((r) => `${r.item_id} ${r.sim}`).join(", "));

    // --- GDPR delete: cascades to user data, the audit record stays with actor_id = NULL
    await q(`DELETE FROM users WHERE id = $1`, [U]);
    const n = (await q(`SELECT (SELECT count(*) FROM library WHERE user_id = $1) + (SELECT count(*) FROM play_sessions WHERE user_id = $1)
                        + (SELECT count(*) FROM sessions WHERE user_id = $1) + (SELECT count(*) FROM item_feedback WHERE user_id = $1) AS n`, [U])).rows[0].n;
    ok("delete cascades (library, plays, sessions, feedback)", Number(n) === 0, `${n} rows left`);
    const audit = (await q(`SELECT actor_id FROM admin_audit_log WHERE target = '20261111-171751'`)).rows;
    ok("audit log kept with actor_id = NULL", audit.length === 1 && audit[0].actor_id === null);
  } finally {
    await q("ROLLBACK").catch(() => {});   // nothing from this script stays in the database
    await c.end();
  }
  const failed = results.filter((p) => !p).length;
  log(`${results.length - failed}/${results.length} checks passed (all rows rolled back)`);
  return failed;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  checkSchema(process.env.DATABASE_URL_DIRECT || process.env.DATABASE_URL)
    .then((failed) => process.exit(failed ? 1 : 0))
    .catch((e) => { console.error("check_schema:", e.message); process.exit(1); });
}
