"""TeleDuct correlation storage adapter."""
from teleduct.fusion.join import (
    correlate_requests,
    EventIngestionStore,
    ConflictingEventError,
    PersistenceError,
)

__all__ = [
    "correlate_requests",
    "EventIngestionStore",
    "ConflictingEventError",
    "PersistenceError",
]
