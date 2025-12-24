# JWT Token Generation Scripts

This directory contains helper scripts for local development.

## generate_token.py

Generate JWT tokens for testing protected endpoints without needing to register/login.

### Usage

```bash
# Generate token with defaults (dev@example.com, MEMBER role, 24h expiry)
make token

# Or run directly
python scripts/generate_token.py

# Generate admin token
python scripts/generate_token.py --email admin@example.com --role ADMIN

# Custom expiry (48 hours)
python scripts/generate_token.py --email user@example.com --hours 48
```

### Available Roles

- `OWNER` - Full system access
- `ADMIN` - Administrative privileges
- `MEMBER` - Standard user (default)
- `VIEWER` - Read-only access

### Requirements

Set `JWT_SECRET_KEY` environment variable:

```bash
export JWT_SECRET_KEY="your-secret-key-min-32-characters-long"
```

### Example Output

```
✅ JWT Token Generated Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Email:   dev@example.com
Role:    MEMBER
User ID: 78c74c1d-b113-45da-8437-cc43f86b0912
Expires: 2025-12-14 16:38:09 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

Use with curl:
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/auth/me
```
