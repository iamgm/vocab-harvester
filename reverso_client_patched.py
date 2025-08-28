import requests

try:
    # Import the original client to preserve full functionality
    from reverso_context_api import Client as _BaseClient
except ImportError as exc:  # pragma: no cover
    raise


class _SessionWithJson(requests.Session):
    """requests.Session with a json_request helper to match library expectations."""

    def json_request(self, *args, **kwargs):
        # Supports both signatures seen in the wild:
        # 1) json_request(url, payload, ...)
        # 2) json_request(method, url, payload, ...)
        if not args:
            raise TypeError("json_request requires at least url or method+url")

        method = 'POST'
        if isinstance(args[0], str) and args[0].upper() in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}:
            method = args[0].upper()
            if len(args) < 2:
                raise TypeError("json_request(method, url, ...) requires url")
            url = args[1]
            payload = args[2] if len(args) >= 3 else None
            rest = args[3:]
        else:
            url = args[0]
            payload = args[1] if len(args) >= 2 else None
            rest = args[2:]

        kw = dict(kwargs)
        if payload is not None and 'json' not in kw:
            kw['json'] = payload
        return self.request(method, url, *rest, **kw)


class Client(_BaseClient):
    """Patched Reverso Context Client enforcing a desktop User-Agent.

    This subclass wraps the original library's Client but injects a
    real-browser User-Agent into the underlying requests session to
    reduce 403/Forbidden responses.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "ru",
        proxies: dict | None = None,
        session: requests.Session | None = None,
        **kwargs,
    ):
        sess = session or _SessionWithJson()
        # Force a real User-Agent to look like a browser
        sess.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/100.0.0.0 Safari/537.36"
                )
            }
        )
        if proxies:
            sess.proxies.update(proxies)

        # Call base init with only the supported args
        super().__init__(
            source_lang=source_lang,
            target_lang=target_lang,
            **kwargs,
        )

        # Try to inject our prepared session into commonly used attributes
        try:
            setattr(self, "session", sess)
        except Exception:
            pass
        try:
            setattr(self, "_session", sess)
        except Exception:
            pass


