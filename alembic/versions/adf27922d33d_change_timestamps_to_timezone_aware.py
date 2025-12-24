"""change_timestamps_to_timezone_aware

Revision ID: adf27922d33d
Revises: 5bbef6a610a4
Create Date: 2025-12-14 19:28:18.990171

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "adf27922d33d"
down_revision: Union[str, Sequence[str], None] = "5bbef6a610a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Change timestamp columns to timezone-aware (TIMESTAMP WITH TIME ZONE)
    op.execute(
        "ALTER TABLE users ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "ALTER TABLE users ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "ALTER TABLE tokens ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "ALTER TABLE tokens ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "ALTER TABLE tokens ALTER COLUMN expires_at TYPE TIMESTAMP WITH TIME ZONE"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Revert to timezone-naive timestamps
    op.execute(
        "ALTER TABLE users ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE"
    )
    op.execute(
        "ALTER TABLE users ALTER COLUMN updated_at TYPE TIMESTAMP WITHOUT TIME ZONE"
    )
    op.execute(
        "ALTER TABLE tokens ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE"
    )
    op.execute(
        "ALTER TABLE tokens ALTER COLUMN updated_at TYPE TIMESTAMP WITHOUT TIME ZONE"
    )
    op.execute(
        "ALTER TABLE tokens ALTER COLUMN expires_at TYPE TIMESTAMP WITHOUT TIME ZONE"
    )
