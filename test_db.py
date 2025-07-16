# backend/test_db.py
import asyncio
from datetime import datetime
from app.models import question_store  # ✅ make sure this import is correct

async def demo():
    recent = await question_store.get_recent()
    print("\n=== Last 10 Questions ===")
    for i, row in enumerate(recent, 1):
        print(f"{i}. {row['query']}  (created: {row['created']})")

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())  # makes sure imports work
    asyncio.run(demo())
