"""Public API limit constants exposed through schema descriptions."""

MAX_QUERY_LENGTH = 4_000

AUTH_RATE_LIMIT = "5 requests per minute per IP for login/register"
CHAT_RATE_LIMIT = "30 chat requests per minute per authenticated user"
GLOBAL_QUEUE_LIMIT = "1000 queued or running RAG requests globally"

RATE_LIMIT_ERROR_CODE = "rate_limit_exceeded"
QUEUE_LIMIT_ERROR_CODE = "queue_limit_exceeded"
