from sqlalchemy import text

from core.database import get_db_session_context
from core.exceptions import ReadinessError


async def check_db_readiness():
    try:
        async with get_db_session_context() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        raise ReadinessError("DB not ready") from e
