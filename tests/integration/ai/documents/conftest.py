# tests/integration/ai/documents/conftest.py
"""Shared fixtures for AI documents integration tests."""

import os
import time

import psycopg
from dotenv import load_dotenv

# Load environment variables from .env file (for DOCKER_HOST, etc.)
load_dotenv()

# Disable Ryuk container for Colima compatibility
os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"


def init_vector_schema(sync_url: str, embedding_dim: int = 384) -> None:
    """Initialize pgvector extension and vector_items table.

    Args:
        sync_url: PostgreSQL connection URL.
        embedding_dim: Dimension of the embedding vectors.
    """
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with psycopg.connect(sync_url) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS vector_items (
                        namespace TEXT NOT NULL,
                        id TEXT NOT NULL,
                        embedding VECTOR({embedding_dim}) NOT NULL,
                        metadata JSONB NOT NULL,
                        PRIMARY KEY (namespace, id)
                    );
                    """
                )
                conn.commit()
            break
        except psycopg.OperationalError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
