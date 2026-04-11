#!/usr/bin/env python3
"""Migrate data from the old SQLite database to PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_pg.py

Reads DB_PATH (SQLite) and DATABASE_URL (PostgreSQL) from the .env / config.
Assumes the PostgreSQL tables already exist (run `alembic upgrade head` first).
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from config import DATABASE_URL, DB_PATH


def read_sqlite() -> dict[str, list[dict]]:
    """Read all rows from the SQLite database."""
    if not os.path.exists(DB_PATH):
        print(f"SQLite file not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    data: dict[str, list[dict]] = {"users": [], "sessions": [], "messages": []}

    for row in conn.execute("SELECT * FROM users").fetchall():
        data["users"].append(dict(row))

    for row in conn.execute("SELECT * FROM sessions").fetchall():
        data["sessions"].append(dict(row))

    for row in conn.execute("SELECT * FROM messages").fetchall():
        data["messages"].append(dict(row))

    conn.close()
    return data


async def write_postgres(data: dict[str, list[dict]]) -> None:
    engine = create_async_engine(DATABASE_URL)

    async with engine.begin() as conn:
        # Users
        for u in data["users"]:
            await conn.execute(
                text(
                    "INSERT INTO users (id, username, password_hash, display_name, created_at) "
                    "VALUES (:id, :username, :password_hash, :display_name, :created_at) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                u,
            )
        print(f"  Migrated {len(data['users'])} users")

        # Sessions
        for s in data["sessions"]:
            s.setdefault("user_id", "")
            await conn.execute(
                text(
                    "INSERT INTO sessions (id, course_id, user_id, title, created_at, updated_at) "
                    "VALUES (:id, :course_id, :user_id, :title, :created_at, :updated_at) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                s,
            )
        print(f"  Migrated {len(data['sessions'])} sessions")

        # Messages
        for m in data["messages"]:
            m.setdefault("msg_type", "text")
            m.setdefault("metadata", "{}")
            if isinstance(m.get("metadata"), dict):
                m["metadata"] = json.dumps(m["metadata"], ensure_ascii=False)
            await conn.execute(
                text(
                    "INSERT INTO messages (id, session_id, role, content, msg_type, metadata, created_at) "
                    "VALUES (:id, :session_id, :role, :content, :msg_type, :metadata, :created_at) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                m,
            )
        print(f"  Migrated {len(data['messages'])} messages")

    await engine.dispose()


def main():
    print(f"Reading SQLite: {DB_PATH}")
    data = read_sqlite()
    print(f"Found: {len(data['users'])} users, {len(data['sessions'])} sessions, {len(data['messages'])} messages")

    print(f"Writing to PostgreSQL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    asyncio.run(write_postgres(data))
    print("Migration complete!")


if __name__ == "__main__":
    main()
