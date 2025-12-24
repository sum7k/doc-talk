#!/usr/bin/env python3
"""
Generate a JWT token for local development testing.

Usage:
    python scripts/generate_token.py
    python scripts/generate_token.py --email user@example.com
    python scripts/generate_token.py --email admin@example.com --role ADMIN
"""

import argparse
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jose import jwt

from core.settings import get_settings


def generate_token(
    email: str = "dev@example.com", role: str = "MEMBER", hours: int = 24
):
    """Generate a JWT token for testing"""
    settings = get_settings()

    if not settings.jwt_secret_key:
        print("❌ Error: JWT_SECRET_KEY not set in environment")
        print("Set it with: export JWT_SECRET_KEY='your-secret-key'")
        sys.exit(1)

    user_id = str(uuid.uuid4())
    expiry = datetime.now(timezone.utc) + timedelta(hours=hours)

    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": int(expiry.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "iss": "fastapi-skeleton",
    }

    token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")

    print("\n✅ JWT Token Generated Successfully")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Email:   {email}")
    print(f"Role:    {role}")
    print(f"User ID: {user_id}")
    print(f"Expires: {expiry.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\nToken:\n{token}\n")
    print("Use with curl:")
    print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/auth/me\n')

    return token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JWT token for local development"
    )
    parser.add_argument("--email", default="dev@example.com", help="User email")
    parser.add_argument(
        "--role",
        default="MEMBER",
        choices=["OWNER", "ADMIN", "MEMBER", "VIEWER", "API_KEY"],
        help="User role",
    )
    parser.add_argument("--hours", type=int, default=24, help="Token expiry in hours")

    args = parser.parse_args()
    generate_token(args.email, args.role, args.hours)
