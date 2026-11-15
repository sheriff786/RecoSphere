-- Exercises every table and constraint of 001_init.sql. Each "expect error" block must fail.
-- Manual psql script: it WRITES rows (games, a demo user) and keeps them, so run it only on a throwaway
-- database (psql "$URL" -f web/db/test_schema.sql). On Neon use `make db-check` (web/db/check_schema.mjs):
-- same scenarios, one transaction, rolled back.
\set ON_ERROR_STOP 0
\echo '--- happy path'
INSERT INTO games (item_id, title, title_family, item_type, platforms, publisher) VALUES
 ('B07R4C1G6Z','Dragon Ball FighterZ - PlayStation 4','dragon ball fighterz','game','{PS4}','Bandai Namco'),
 ('B0FF7R0001','Final Fantasy VII Remake - PlayStation 4 Deluxe Edition','final fantasy vii remake','game','{PS4}','Square Enix'),
 ('B0FF7R0002','FINAL FANTASY VII REMAKE (PS4)','final fantasy vii remake','game','{PS4}','Square Enix'),
 ('B0MOUSE001','Logitech G502 HERO Gaming Mouse','logitech g502 hero','accessory','{PC}','Logitech');
INSERT INTO users (id, email, password_hash, display_name) VALUES
 ('11111111-1111-1111-1111-111111111111','Sheriff@Example.com','$2b$10$x','Sheriff');
INSERT INTO users (kind, dataset_user_id, display_name) VALUES ('demo','AGEYTHU7HPQ7VA','Player AGEYTHU7');
INSERT INTO library (user_id, item_id, source) VALUES
 ('11111111-1111-1111-1111-111111111111','B0FF7R0001','onboarding');
-- one play = one transaction: session row + library upsert
BEGIN;
INSERT INTO play_sessions (user_id, item_id, client_event_id, seconds_played, request_id, surface)
VALUES ('11111111-1111-1111-1111-111111111111','B07R4C1G6Z','aaaaaaaa-0000-0000-0000-000000000001',120,'978e1bcb','home:top_picks');
INSERT INTO library (user_id, item_id, source, last_played_at, play_count, seconds_played)
VALUES ('11111111-1111-1111-1111-111111111111','B07R4C1G6Z','play',now(),1,120)
ON CONFLICT (user_id, item_id) DO UPDATE SET last_played_at = now(),
  play_count = library.play_count + 1, seconds_played = library.seconds_played + EXCLUDED.seconds_played;
COMMIT;
INSERT INTO item_feedback VALUES ('11111111-1111-1111-1111-111111111111','B0MOUSE001','not_interested');
INSERT INTO sessions (user_id, expires_at) VALUES ('11111111-1111-1111-1111-111111111111', now() + interval '7 days');
INSERT INTO admin_audit_log (actor_id, action, target, details)
VALUES ('11111111-1111-1111-1111-111111111111','model.rollback','20261111-171751','{"from":"20261114-090000"}');

\echo '--- expect error: duplicate email differing only in case (citext)'
INSERT INTO users (email, password_hash, display_name) VALUES ('sheriff@example.com','$2b$10$y','Dup');
\echo '--- expect error: registered user without password'
INSERT INTO users (email, display_name) VALUES ('a@b.com','NoPass');
\echo '--- expect error: demo user without dataset id'
INSERT INTO users (kind, display_name) VALUES ('demo','Ghost');
\echo '--- expect error: play for a game not in the catalog'
INSERT INTO play_sessions (user_id, item_id, client_event_id) VALUES ('11111111-1111-1111-1111-111111111111','NOPE','aaaaaaaa-0000-0000-0000-000000000009');
\echo '--- expect error: retried play request (same client_event_id)'
INSERT INTO play_sessions (user_id, item_id, client_event_id) VALUES ('11111111-1111-1111-1111-111111111111','B07R4C1G6Z','aaaaaaaa-0000-0000-0000-000000000001');
\echo '--- expect error: negative seconds'
INSERT INTO play_sessions (user_id, item_id, client_event_id, seconds_played) VALUES ('11111111-1111-1111-1111-111111111111','B07R4C1G6Z','aaaaaaaa-0000-0000-0000-000000000003',-5);

\echo '--- queries the app and engine will run'
\echo 'history sent to the engine (most recent first):'
SELECT item_id, source, play_count FROM library
 WHERE user_id = '11111111-1111-1111-1111-111111111111'
 ORDER BY COALESCE(last_played_at, added_at) DESC;
\echo 'dedup against history by title family (B0FF7R0002 is the same game as the owned Deluxe edition):'
SELECT g.item_id, g.title FROM games g
 WHERE g.item_id IN ('B0FF7R0002','B07R4C1G6Z','B0MOUSE001')
   AND g.item_type IN ('game','dlc') AND g.is_active
   AND g.title_family NOT IN (SELECT g2.title_family FROM library l JOIN games g2 USING (item_id)
                              WHERE l.user_id = '11111111-1111-1111-1111-111111111111')
   AND g.item_id NOT IN (SELECT item_id FROM item_feedback
                         WHERE user_id = '11111111-1111-1111-1111-111111111111' AND kind IN ('dislike','not_interested'));
\echo 'fuzzy search "fantsy" (typo):'
SELECT item_id, title, round(similarity(title, 'fantsy remake')::numeric, 2) AS sim FROM games
 WHERE title % 'fantsy remake' ORDER BY sim DESC;
\echo 'GDPR delete cascades:'
DELETE FROM users WHERE id = '11111111-1111-1111-1111-111111111111';
SELECT (SELECT count(*) FROM library) lib, (SELECT count(*) FROM play_sessions) plays,
       (SELECT count(*) FROM sessions) sess, (SELECT count(*) FROM item_feedback) fb,
       (SELECT count(*) FROM admin_audit_log WHERE actor_id IS NULL) audit_kept;
