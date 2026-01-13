from datetime import datetime, timezone

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def seconds_since(ts: datetime) -> float:
    return (utcnow() - ts).total_seconds()
