import os
import ssl
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from celery import Celery


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


_OCR_TASK_NAMES = {
    "tasks.ocr_document",
    "tasks.ocr_page",
    "tasks.ocr_pdf_document",
}


def _default_queue() -> str:
    value = _env("CELERY_DEFAULT_QUEUE", "").strip()
    return value or "default"


def _ocr_queue() -> str:
    return _env("CELERY_OCR_QUEUE", "ocr")


def _build_task_routes(default_queue: str, ocr_queue: str) -> dict[str, dict[str, str]]:
    routes: dict[str, dict[str, str]] = {"tasks.*": {"queue": default_queue}}
    for name in _OCR_TASK_NAMES:
        routes[name] = {"queue": ocr_queue}
    return routes

def _canonicalize_redis_url(url: str) -> str:
    try:
        u = urlparse(url)
        if u.scheme in ("redis", "rediss") and u.hostname:
            # Prefer password-only auth (":password@host") for Celery/kombu compatibility
            password = u.password
            hostport = f"{u.hostname}{':' + str(u.port) if u.port else ''}"
            auth = f":{password}@" if password else ("@" if u.username else "")
            netloc = f"{auth}{hostport}" if (password or u.username) else hostport
            path = u.path or "/0"
            q = dict(parse_qsl(u.query, keep_blank_values=True))
            # Celery 5.3+ requires ssl_cert_reqs in rediss URL
            if u.scheme == "rediss" and "ssl_cert_reqs" not in q:
                q["ssl_cert_reqs"] = os.getenv("REDIS_SSL_CERT_REQS", "CERT_REQUIRED")
            query = urlencode(q)
            return urlunparse((u.scheme, netloc, path, u.params, query, u.fragment))
    except Exception:
        pass
    return url


broker = _canonicalize_redis_url(_env("CELERY_BROKER_URL", _env("REDIS_URL", "redis://localhost:6379/0")))
backend = _canonicalize_redis_url(_env("CELERY_RESULT_BACKEND", broker))

celery_app = Celery("lernio", broker=broker, backend=backend, include=["tasks"]) 

# Robustness: retry connecting on startup and allow optional TLS disable (not recommended)
celery_app.conf.broker_connection_retry_on_startup = True
if broker.startswith("rediss://"):
    # Default to strict certificate checks unless explicitly disabled
    if _env("REDIS_TLS_INSECURE", "0") == "1":
        celery_app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_NONE}
        celery_app.conf.redis_backend_use_ssl = {"ssl_cert_reqs": ssl.CERT_NONE}
    else:
        celery_app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
        celery_app.conf.redis_backend_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}

_DEFAULT_QUEUE = _default_queue()
_OCR_QUEUE = _ocr_queue()
celery_app.conf.task_default_queue = _DEFAULT_QUEUE
celery_app.conf.task_routes = _build_task_routes(_DEFAULT_QUEUE, _OCR_QUEUE)

# Log sanitized broker on import for debugging (passwords hidden)
_sanitized = broker
if ":" in _sanitized and "@" in _sanitized:
    try:
        # Replace password with ****** in netloc
        parsed = urlparse(_sanitized)
        if parsed.netloc:
            parts = parsed.netloc.split("@")
            auth, host = parts[0], parts[1]
            if ":" in auth:
                auth = auth.split(":")[0] + ":******"
            _sanitized = urlunparse((parsed.scheme, f"{auth}@{host}", parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        pass
print(f"[celery] broker={_sanitized} default_queue={_DEFAULT_QUEUE} ocr_queue={_OCR_QUEUE}")

# Ensure tasks module is imported so task definitions are registered
try:
    import tasks as _register_tasks  # noqa: F401
    print("[celery] tasks module imported")
except Exception as _e:  # pragma: no cover
    print(f"[celery] warning: could not import tasks module via absolute import: {_e}")
