"""ScoreEntry — the canonical data model for a single leaderboard entry."""

from __future__ import annotations

import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class ScoreEntry:
    """One player result for one game run.

    Attributes
    ----------
    game_id:
        Snake-case identifier, e.g. ``"le_messager"``.  Must be a key in
        ``client.GAME_REGISTRY`` for proper ranking direction.
    player_name:
        Display name typed by the player.
    score:
        For ``score_type == "time_asc"``   → elapsed seconds (float).
        For ``score_type == "points_desc"`` → points scored (float/int).
    data_id:
        Short identifier for the dataset used, e.g. ``"Hollow_test"``.
        Included so the leaderboard can show per-dataset sub-rankings later.
    timestamp:
        ISO-8601 UTC string, set automatically at instantiation.
    machine:
        Hostname, set automatically.  Helps diagnose duplicate submissions.
    metadata:
        Free-form dict for game-specific extras (circuit waypoints count,
        savior health remaining, etc.).
    """

    game_id: str
    player_name: str
    score: float
    data_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    machine: str = field(default_factory=socket.gethostname)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
