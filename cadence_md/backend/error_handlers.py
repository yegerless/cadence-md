"""Map exceptions to ``ErrorResponse`` JSON for API clients."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from cadence_md.backend.exceptions import ApiError
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.limits import RATE_LIMIT_ERROR_CODE


def get_request_id(request: Request) -> str | None:
    """Return correlation id from middleware, if present."""
    return getattr(request.state, "request_id", None)


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Serialize ``ApiError`` to ``ErrorResponse``."""
    body = ErrorResponse(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        request_id=get_request_id(request),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(body.model_dump()),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Map validation errors to a stable contract."""
    body = ErrorResponse(
        code="validation_error",
        message="Request validation failed.",
        details={"errors": exc.errors()},
        request_id=get_request_id(request),
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(body.model_dump()),
    )


async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Normalize Starlette/FastAPI HTTP exceptions and SlowAPI rate limits."""
    rid = get_request_id(request)
    if isinstance(exc, RateLimitExceeded):
        body = ErrorResponse(
            code=RATE_LIMIT_ERROR_CODE,
            message=str(exc.detail),
            details={},
            request_id=rid,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(body.model_dump()),
        )

    detail = exc.detail
    if isinstance(detail, str):
        message = detail
        code = "http_error"
        details: dict[str, object] = {}
    elif isinstance(detail, dict):
        message = str(detail.get("message", "Request failed."))
        code = str(detail.get("code", "http_error"))
        details = {k: v for k, v in detail.items() if k not in {"message", "code"}}
    else:
        message = str(detail)
        code = "http_error"
        details = {}

    body = ErrorResponse(code=code, message=message, details=details, request_id=rid)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(body.model_dump()),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so clients always receive ``ErrorResponse``-shaped JSON."""
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
