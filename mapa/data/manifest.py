"""Native BrainTreebank sample rates for converting event sample indices to seconds."""


def bt_subject_native_rate_hz(subject_id: int) -> int:
    """S9 is distributed at 1024 Hz; other subjects use 2048 Hz."""
    return 1024 if int(subject_id) == 9 else 2048
