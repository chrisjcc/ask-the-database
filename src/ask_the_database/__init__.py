"""AskTheDatabase: Multi-turn reinforcement learning with verifiable rewards for NL2SQL."""

import os
import site
import sys


def sanitize_user_site() -> None:
    """Ensure ~/.local user site-packages do not shadow virtualenv / conda dependencies."""
    if os.environ.get("ENABLE_USER_SITE") == "1":
        return
    is_isolated_env = (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or "CONDA_PREFIX" in os.environ
        or "VIRTUAL_ENV" in os.environ
    )
    if is_isolated_env and hasattr(site, "getusersitepackages"):
        user_sites = site.getusersitepackages()
        targets = [user_sites] if isinstance(user_sites, str) else list(user_sites)
        sys.path[:] = [
            p for p in sys.path if not any(p == u or p.startswith(u + os.sep) for u in targets)
        ]


sanitize_user_site()

__version__ = "0.1.0"
__all__ = ["sanitize_user_site", "__version__"]
