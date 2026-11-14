// App database (SQLite): accounts + play sessions. The engine keeps the *recommendation* state
// (user history in Redis/memory); this DB is only what a product needs on top of it.
import Database from "better-sqlite3";
import bcrypt from "bcryptjs";
import fs from "node:fs";
import path from "node:path";

export type User = {
  id: number;
  email: string | null;
  name: string;
  kind: "app" | "dataset";
  engine_user_id: string;
  onboarded: number;
  created_at: string;
};

const g = globalThis as unknown as { __db?: Database.Database };

function open(): Database.Database {
  const file = process.env.DATABASE_PATH || "./data/app.db";
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const db = new Database(file);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE,
      name TEXT NOT NULL,
      password_hash TEXT,
      kind TEXT NOT NULL CHECK (kind IN ('app','dataset')),
      engine_user_id TEXT UNIQUE NOT NULL,
      onboarded INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS plays (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL REFERENCES users(id),
      item_id TEXT NOT NULL,
      request_id TEXT,
      started_at TEXT NOT NULL DEFAULT (datetime('now')),
      seconds INTEGER
    );
    CREATE INDEX IF NOT EXISTS plays_user ON plays(user_id, started_at);
  `);
  const admin = (process.env.ADMIN_EMAILS || "admin@recosphere.dev").split(",")[0].trim();
  const exists = db.prepare("SELECT 1 FROM users WHERE email = ?").get(admin);
  if (!exists && process.env.ADMIN_PASSWORD) {
    const info = db.prepare(
      "INSERT INTO users (email, name, password_hash, kind, engine_user_id, onboarded) VALUES (?, ?, ?, 'app', ?, 1)"
    ).run(admin, "Admin", bcrypt.hashSync(process.env.ADMIN_PASSWORD, 10), `app:admin`);
    void info;
  }
  return db;
}

export function db(): Database.Database {
  if (!g.__db) g.__db = open();
  return g.__db;
}

export function createAppUser(email: string, name: string, password: string): User {
  const d = db();
  const tx = d.transaction(() => {
    const r = d.prepare("INSERT INTO users (email, name, password_hash, kind, engine_user_id) VALUES (?, ?, ?, 'app', ?)")
      .run(email.toLowerCase(), name, bcrypt.hashSync(password, 10), `tmp:${Date.now()}:${Math.random()}`);
    const id = Number(r.lastInsertRowid);
    d.prepare("UPDATE users SET engine_user_id = ? WHERE id = ?").run(`app:${id}`, id);   // engine id = app:<id>
    return id;
  });
  return getUser(tx())!;
}

export function getUser(id: number): User | undefined {
  return db().prepare("SELECT id, email, name, kind, engine_user_id, onboarded, created_at FROM users WHERE id = ?").get(id) as User | undefined;
}

export function checkPassword(email: string, password: string): User | undefined {
  const row = db().prepare("SELECT * FROM users WHERE email = ? AND kind = 'app'").get(email.toLowerCase()) as
    (User & { password_hash: string }) | undefined;
  if (!row || !bcrypt.compareSync(password, row.password_hash)) return undefined;
  return getUser(row.id);
}

export function datasetUser(engineUserId: string): User {
  const d = db();
  const row = d.prepare("SELECT id FROM users WHERE engine_user_id = ?").get(engineUserId) as { id: number } | undefined;
  if (row) return getUser(row.id)!;
  const r = d.prepare("INSERT INTO users (name, kind, engine_user_id, onboarded) VALUES (?, 'dataset', ?, 1)")
    .run(`Player ${engineUserId.slice(0, 8)}`, engineUserId);
  return getUser(Number(r.lastInsertRowid))!;
}

export function setOnboarded(id: number) {
  db().prepare("UPDATE users SET onboarded = 1 WHERE id = ?").run(id);
}

export function startPlay(userId: number, itemId: string, requestId: string | null): number {
  return Number(db().prepare("INSERT INTO plays (user_id, item_id, request_id) VALUES (?, ?, ?)")
    .run(userId, itemId, requestId).lastInsertRowid);
}

export function endPlay(userId: number, playId: number, seconds: number) {
  db().prepare("UPDATE plays SET seconds = ? WHERE id = ? AND user_id = ?").run(Math.round(seconds), playId, userId);
}

export function playStats(userId: number) {
  return db().prepare(
    "SELECT COUNT(*) AS sessions, COALESCE(SUM(seconds),0) AS seconds FROM plays WHERE user_id = ?"
  ).get(userId) as { sessions: number; seconds: number };
}

export function isAdmin(u: User | undefined | null): boolean {
  if (!u?.email) return false;
  const list = (process.env.ADMIN_EMAILS || "admin@recosphere.dev").split(",").map((s) => s.trim().toLowerCase());
  return list.includes(u.email.toLowerCase());
}
