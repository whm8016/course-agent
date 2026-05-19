"""Shared rate-limiter instance used by all routers."""
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import REDIS_URL

limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)
