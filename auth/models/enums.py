# models/enums.py (NEW FILE)
from enum import Enum


class UserRole(str, Enum):
    """Simple role hierarchy for SaaS/AI apps"""

    OWNER = "OWNER"  # Organization owner
    ADMIN = "ADMIN"  # Can manage users, billing
    MEMBER = "MEMBER"  # Regular user
    VIEWER = "VIEWER"  # Read-only access
    API_KEY = "API_KEY"  # For programmatic access (AI apps need this!)


# Role hierarchy (owner can do everything)
ROLE_HIERARCHY = {
    UserRole.OWNER: 100,
    UserRole.ADMIN: 75,
    UserRole.MEMBER: 50,
    UserRole.VIEWER: 25,
    UserRole.API_KEY: 10,
}
