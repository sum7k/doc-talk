from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "request_count",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

ERROR_COUNT = Counter(
    "error_count",
    "Total HTTP error responses (5xx)",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency",
    ["method", "path", "status_code"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    ["db_system", "operation", "method", "path"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2),
)

DB_ERROR_COUNT = Counter(
    "db_errors_total",
    "Database query errors",
    ["db_system", "operation", "method", "path"],
)
