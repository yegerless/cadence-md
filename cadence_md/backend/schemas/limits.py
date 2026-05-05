"""Public API limit constants exposed through schema descriptions."""

MAX_QUERY_LENGTH = 4_000

AUTH_RATE_LIMIT = (
    "Per-IP limits for POST /auth/register and POST /auth/login "
    "(configure via AUTH_REGISTER_RATE_LIMIT_IP and AUTH_LOGIN_RATE_LIMIT_IP)."
)
CHAT_RATE_LIMIT = "30 chat requests per minute per authenticated user"
GLOBAL_QUEUE_LIMIT = "1000 queued or running RAG requests globally"

RATE_LIMIT_ERROR_CODE = "rate_limit_exceeded"
QUEUE_LIMIT_ERROR_CODE = "queue_limit_exceeded"
