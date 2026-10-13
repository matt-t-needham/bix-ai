"""Cloudflare Access checks for the routes that reach the agent runner.

Agent-mode chats and command approvals end in a shell in the runner's sandbox,
so they don't rely on Access sitting in front: any container on the bix
network can reach ai-router directly and set any header. Same check as
rss_feed's auth.access_verified."""
import logging

import jwt
from fastapi import Request

import config

log = logging.getLogger("router")

_jwks: jwt.PyJWKClient | None = None


def _jwks_client() -> jwt.PyJWKClient:
    """Cached client for the Access team's signing keys."""
    global _jwks
    if _jwks is None:
        _jwks = jwt.PyJWKClient(
            f"https://{config.ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs",
            cache_keys=True, lifespan=3600, timeout=10,
        )
    return _jwks


def access_verified(request: Request) -> bool:
    """True only for a valid Access JWT for this app: good signature from the
    team's keys, our audience and issuer, not expired. Fails closed when
    ACCESS_AUD is unset."""
    if not config.ACCESS_AUD:
        log.warning("access check refused: ACCESS_AUD is not configured")
        return False
    token = request.headers.get("cf-access-jwt-assertion", "")
    if not token:
        return False
    try:
        key = _jwks_client().get_signing_key_from_jwt(token).key
        jwt.decode(
            token, key, algorithms=["RS256"],
            audience=config.ACCESS_AUD,
            issuer=f"https://{config.ACCESS_TEAM_DOMAIN}",
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except Exception as e:
        log.warning("access check refused: %s", e)
        return False
    return True


def allowed(request: Request) -> bool:
    """A verified Access JWT, from a browser on this origin. Sibling
    *.bix.computer pages are same-site and carry the Access cookie, so the JWT
    alone doesn't stop them (same rule as nginx/control-route.inc); clients
    that send no Sec-Fetch-Site are judged by the JWT alone."""
    site = request.headers.get("sec-fetch-site")
    return site in (None, "same-origin", "none") and access_verified(request)
