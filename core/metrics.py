from prometheus_client import Counter, Gauge, Histogram

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


# ============================================================================
# LLM Metrics
# ============================================================================

LLM_COMPLETION_DURATION = Histogram(
    "llm_completion_duration",
    "LLM completion duration in seconds",
    ["model", "provider"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
)

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "provider"],
)

LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total LLM errors",
    ["model", "provider", "error_type"],
)

LLM_TOKENS_PROMPT = Counter(
    "llm_tokens_prompt",
    "Total prompt tokens used",
    ["model", "provider"],
)

LLM_TOKENS_COMPLETION = Counter(
    "llm_tokens_completion",
    "Total completion tokens used",
    ["model", "provider"],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens used (prompt + completion)",
    ["model", "provider"],
)


# ============================================================================
# Embeddings Metrics
# ============================================================================

EMBEDDINGS_DURATION = Histogram(
    "embeddings_duration",
    "Embeddings generation duration in seconds",
    ["model", "backend"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

EMBEDDINGS_OPENAI_DURATION = Histogram(
    "openai_embeddings_duration",
    "OpenAI embeddings duration in seconds",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

EMBEDDINGS_LOCAL_DURATION = Histogram(
    "local_embeddings_duration",
    "Local embeddings duration in seconds",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

EMBEDDINGS_REQUESTS_TOTAL = Counter(
    "embeddings_requests_total",
    "Total embeddings requests",
    ["model", "backend"],
)

EMBEDDINGS_ERRORS_TOTAL = Counter(
    "embeddings_errors_total",
    "Total embeddings errors",
    ["model", "backend", "error_type"],
)

EMBEDDINGS_BATCH_SIZE = Gauge(
    "embeddings_batch_size",
    "Current embeddings batch size",
    ["model", "backend"],
)


# ============================================================================
# Vector Store Metrics (PgVector)
# ============================================================================

PGVECTOR_UPSERT_DURATION = Histogram(
    "pgvector_upsert_duration",
    "PgVector upsert duration in seconds",
    ["namespace"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

PGVECTOR_QUERY_DURATION = Histogram(
    "pgvector_query_duration",
    "PgVector query duration in seconds",
    ["namespace"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

PGVECTOR_DELETE_DURATION = Histogram(
    "pgvector_delete_duration",
    "PgVector delete duration in seconds",
    ["namespace"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

PGVECTOR_OPERATIONS_TOTAL = Counter(
    "pgvector_operations_total",
    "Total PgVector operations",
    ["operation", "namespace"],
)

PGVECTOR_ERRORS_TOTAL = Counter(
    "pgvector_errors_total",
    "Total PgVector errors",
    ["operation", "namespace", "error_type"],
)


# ============================================================================
# Vector Store Metrics (Qdrant)
# ============================================================================

QDRANT_UPSERT_DURATION = Histogram(
    "qdrant_upsert_duration",
    "Qdrant upsert duration in seconds",
    ["collection"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

QDRANT_QUERY_DURATION = Histogram(
    "qdrant_query_duration",
    "Qdrant query duration in seconds",
    ["collection"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

QDRANT_DELETE_DURATION = Histogram(
    "qdrant_delete_duration",
    "Qdrant delete duration in seconds",
    ["collection"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

QDRANT_OPERATIONS_TOTAL = Counter(
    "qdrant_operations_total",
    "Total Qdrant operations",
    ["operation", "collection"],
)

QDRANT_ERRORS_TOTAL = Counter(
    "qdrant_errors_total",
    "Total Qdrant errors",
    ["operation", "collection", "error_type"],
)


# ============================================================================
# Tool Metrics
# ============================================================================

TOOL_CALL_DURATION = Histogram(
    "tool_call_duration",
    "Tool call duration in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
)

TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool_name"],
)

TOOL_ERRORS_TOTAL = Counter(
    "tool_errors_total",
    "Total tool errors",
    ["tool_name", "error_type"],
)


# ============================================================================
# Chunking Metrics
# ============================================================================

CHUNKING_DURATION = Histogram(
    "chunking_duration",
    "Chunking duration in seconds",
    ["chunker_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)

CHUNKING_CHUNKS_CREATED = Counter(
    "chunking_chunks_created",
    "Total chunks created",
    ["chunker_type"],
)


# ============================================================================
# Metric Registries
# ============================================================================

HISTOGRAMS = {
    "llm_completion_duration": LLM_COMPLETION_DURATION,
    "embeddings_duration": EMBEDDINGS_DURATION,
    "openai_embeddings_duration": EMBEDDINGS_OPENAI_DURATION,
    "local_embeddings_duration": EMBEDDINGS_LOCAL_DURATION,
    "pgvector_upsert_duration": PGVECTOR_UPSERT_DURATION,
    "pgvector_query_duration": PGVECTOR_QUERY_DURATION,
    "pgvector_delete_duration": PGVECTOR_DELETE_DURATION,
    "qdrant_upsert_duration": QDRANT_UPSERT_DURATION,
    "qdrant_query_duration": QDRANT_QUERY_DURATION,
    "qdrant_delete_duration": QDRANT_DELETE_DURATION,
    "tool_call_duration": TOOL_CALL_DURATION,
    "chunking_duration": CHUNKING_DURATION,
}

COUNTERS = {
    "llm_requests_total": LLM_REQUESTS_TOTAL,
    "llm_errors_total": LLM_ERRORS_TOTAL,
    "llm_tokens_prompt": LLM_TOKENS_PROMPT,
    "llm_tokens_completion": LLM_TOKENS_COMPLETION,
    "llm_tokens_total": LLM_TOKENS_TOTAL,
    "embeddings_requests_total": EMBEDDINGS_REQUESTS_TOTAL,
    "embeddings_errors_total": EMBEDDINGS_ERRORS_TOTAL,
    "pgvector_operations_total": PGVECTOR_OPERATIONS_TOTAL,
    "pgvector_errors_total": PGVECTOR_ERRORS_TOTAL,
    "qdrant_operations_total": QDRANT_OPERATIONS_TOTAL,
    "qdrant_errors_total": QDRANT_ERRORS_TOTAL,
    "tool_calls_total": TOOL_CALLS_TOTAL,
    "tool_errors_total": TOOL_ERRORS_TOTAL,
    "chunking_chunks_created": CHUNKING_CHUNKS_CREATED,
}

GAUGES = {
    "embeddings_batch_size": EMBEDDINGS_BATCH_SIZE,
}


class PrometheusMetricsHook:
    def record_latency(self, name, value_ms, labels=None):
        try:
            HISTOGRAMS[name].labels(**(labels or {})).observe(value_ms / 1000)
        except KeyError:
            raise RuntimeError(f"Histogram not registered: {name}")

    def increment(self, name, value=1, labels=None):
        try:
            COUNTERS[name].labels(**(labels or {})).inc(value)
        except KeyError:
            raise RuntimeError(f"Counter not registered: {name}")

    def record_gauge(self, name, value, labels=None):
        try:
            GAUGES[name].labels(**(labels or {})).set(value)
        except KeyError:
            raise RuntimeError(f"Gauge not registered: {name}")
