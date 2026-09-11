"""Read the array name and the contact number off a clinical electrode label.

The only thing standing between a label string and the geometry the priors need, and it
depends on nothing beyond the standard library.
"""
from __future__ import annotations

import re

_ARRAY_RE = re.compile(r"^(.*?)(\d+)$")


def parse_array(channel_name: str) -> tuple[str, int | None]:
    """Split ``"OFa12"`` into ``("OFa", 12)``. Non-numeric suffix returns ``(name, None)``."""
    match = _ARRAY_RE.match(channel_name)
    if match is None:
        return channel_name, None
    return match.group(1), int(match.group(2))
