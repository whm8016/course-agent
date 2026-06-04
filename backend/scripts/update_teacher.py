"""One-time script: ensure teacher account 游老师 exists with password you1234.

Handles two cases:
  - 游老师 already exists → update password only
  - teacher1 exists but 游老师 doesn't → rename + update password

Usage:
    cd backend && python -m scripts.update_teacher
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update
from core.db.database import AsyncSessionLocal, User, engine
from core.db.auth import hash_password


async def main():
    new_hash = hash_password("you1234")

    async with AsyncSessionLocal() as db:
        # Check if 游老师 already exists
        result = await db.execute(select(User).where(User.username == "游老师"))
        existing = result.scalar_one_or_none()

        if existing:
            await db.execute(
                update(User)
                .where(User.id == existing.id)
                .values(password_hash=new_hash)
            )
            await db.commit()
            print(f"OK: 游老师 (id={existing.id}) already exists → password updated to you1234")
        else:
            result = await db.execute(select(User).where(User.username == "teacher1"))
            teacher1 = result.scalar_one_or_none()
            if not teacher1:
                print("ERROR: neither 'teacher1' nor '游老师' found in users table")
                return
            await db.execute(
                update(User)
                .where(User.id == teacher1.id)
                .values(username="游老师", display_name="游老师", password_hash=new_hash)
            )
            await db.commit()
            print(f"OK: teacher1 (id={teacher1.id}) renamed → 游老师, password=you1234")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
