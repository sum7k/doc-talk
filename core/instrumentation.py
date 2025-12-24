import time

from sqlalchemy import event
from sqlalchemy.engine import Engine
from structlog.contextvars import get_contextvars

from core.metrics import DB_ERROR_COUNT, DB_QUERY_DURATION


def get_operation(sql: str) -> str:
    sql = sql.strip().lower()
    if sql.startswith("select"):
        return "select"
    if sql.startswith("insert"):
        return "insert"
    if sql.startswith("update"):
        return "update"
    if sql.startswith("delete"):
        return "delete"
    return "other"


def setup_db_metrics(engine: Engine, db_system: str = "postgres"):
    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        context._query_start_time = time.perf_counter()
        context._operation = get_operation(statement)

    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        duration = time.perf_counter() - context._query_start_time

        ctx = get_contextvars()
        method = ctx.get("method", "unknown")
        path = ctx.get("path", "unknown")

        DB_QUERY_DURATION.labels(
            db_system=db_system,
            operation=context._operation,
            method=method,
            path=path,
        ).observe(duration)

    @event.listens_for(engine, "handle_error")
    def handle_error(exception_context):
        ctx = get_contextvars()
        method = ctx.get("method", "unknown")
        path = ctx.get("path", "unknown")

        DB_ERROR_COUNT.labels(
            db_system=db_system,
            operation="unknown",
            method=method,
            path=path,
        ).inc()
