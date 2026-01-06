from typing import Generator
from sqlalchemy.orm import Session
from app.adapters.database import SessionLocal
# from fastapi import HTTPException, status

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# auth to be added here later