"""marvel_view.leaderboard — shared leaderboard backed by a GitHub-hosted JSON file."""

from .client import GAME_REGISTRY, fetch_scores, submit_score
from .models import ScoreEntry

__all__ = ["ScoreEntry", "GAME_REGISTRY", "submit_score", "fetch_scores"]
