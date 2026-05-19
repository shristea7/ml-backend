"""MongoDB connection and utilities for the backend."""
import os
import threading

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/medley")

_client: MongoClient | None = None
_db = None
_lock = threading.Lock()  # safe for multi-threaded servers (Flask, FastAPI, etc.)


def get_db():
    """
    Return the MongoDB database instance, creating the connection if needed.
    Thread-safe — safe to call from concurrent request handlers.
    """
    global _client, _db

    if _db is not None:
        return _db

    with _lock:
        # Double-checked locking: another thread may have initialised while we waited
        if _db is not None:
            return _db

        try:
            _client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10_000,
            )
            _client.admin.command("ping")   # fail fast if unreachable
            _db = _client.get_database()
            print(f"[MongoDB] Connected → {MONGODB_URI}")
        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            print(f"[MongoDB] Could not connect to {MONGODB_URI}: {exc}")
            raise

    return _db


def close_db():
    """Close the MongoDB connection and reset the cached instances."""
    global _client, _db

    with _lock:
        if _client:
            _client.close()
            print("[MongoDB] Connection closed.")
        _client = None
        _db = None
