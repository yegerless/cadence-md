"""Aggregate backend API routers."""

from fastapi import APIRouter

from cadence_md.backend.api.routers import auth, chat, health, users

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(chat.router)
api_router.include_router(health.router)

__all__ = ["api_router"]
