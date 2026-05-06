"""Public API limit constants exposed through schema descriptions."""

MAX_QUERY_LENGTH = 4_000

AUTH_RATE_LIMIT = (
    "Per-IP limits for POST /auth/register and POST /auth/login "
    "(configure via AUTH_REGISTER_RATE_LIMIT_IP and AUTH_LOGIN_RATE_LIMIT_IP)."
)
CHAT_RATE_LIMIT = (
    "Per-user limits for POST /api/v1/chat/messages and retry (configure via CHAT_RATE_LIMIT_USER)."
)
GLOBAL_QUEUE_LIMIT = (
    "Global cap on queued + running RAG requests (configure via GLOBAL_RAG_QUEUE_MAX)."
)

RATE_LIMIT_ERROR_CODE = "rate_limit_exceeded"
QUEUE_LIMIT_ERROR_CODE = "queue_limit_exceeded"
