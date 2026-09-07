"""Stateless ASGI entry point for hosts that discover api/index.py."""

from model_lifecycle.api import app

__all__ = ["app"]

