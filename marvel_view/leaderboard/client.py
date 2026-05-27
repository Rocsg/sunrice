"""Leaderboard client — read/write scores to a GitHub-hosted leaderboard.json.

How it works
------------
Scores are stored in a single JSON file (``leaderboard.json``) in a dedicated
GitHub repository.  Each submission does a read → append → write cycle using
the GitHub Contents API, retrying on SHA conflicts (concurrent writes).

If the network is unavailable or ``SUNRICE_LB_TOKEN`` is not set, the score is
cached in a local SQLite database (``~/.cache/sunrice/leaderboard_pending.db``)
and flushed automatically on the next successful ``submit_score()`` call.

Required environment variable
------------------------------
``SUNRICE_LB_TOKEN``
    GitHub Personal Access Token with *Contents: Read and Write* permission
    on the ``SUNRICE_LB_REPO`` repository.  Fine-grained PATs are preferred
    (scope the token to that single repo only).

Optional environment variables
--------------------------------
``SUNRICE_LB_REPO``
    ``owner/repo`` of the leaderboard repository.
    Default: ``rfernandez/sunrice-leaderboard``

``SUNRICE_LB_LOCAL_DB``
    Path to the local SQLite fallback database.
    Default: ``~/.cache/sunrice/leaderboard_pending.db``
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .models import ScoreEntry

# ── game registry ─────────────────────────────────────────────────────────────
# Each entry defines how scores are ranked for that game.
#   "time_asc"    → lower elapsed seconds is better
#   "points_desc" → higher point count is better
#
# Add new games here as they are implemented.  The server-side JSON will pick
# up the score_type automatically on first submission.

GAME_REGISTRY: dict[str, dict] = {
    "le_messager":     {"label": "Le Messager",      "score_type": "time_asc"},
    "cross_the_beast": {"label": "Cross the Beast",  "score_type": "time_asc"},
    "catch_flux":      {"label": "Catch the Flux",    "score_type": "points_desc"},
    "circuit":         {"label": "Circuit",           "score_type": "time_asc"},
    "the_savior":      {"label": "The Savior",        "score_type": "time_asc"},
    # Suggested extras
    "root_dive":       {"label": "Root Dive",         "score_type": "time_asc"},
    "membrane_surfer": {"label": "Membrane Surfer",   "score_type": "points_desc"},
    "aero_census":     {"label": "Aero-Census",       "score_type": "points_desc"},
    "infection":       {"label": "Infection",         "score_type": "time_asc"},
}

# ── runtime configuration (env-var overridable) ───────────────────────────────

_REPO     = os.environ.get("SUNRICE_LB_REPO", "Rocsg/sunrice-leaderboard")
_TOKEN    = os.environ.get("SUNRICE_LB_TOKEN", "").strip()
_API_BASE = "https://api.github.com"
_FILE     = "leaderboard.json"
_LOCAL_DB = Path(
    os.environ.get(
        "SUNRICE_LB_LOCAL_DB",
        str(Path.home() / ".cache" / "sunrice" / "leaderboard_pending.db"),
    )
)
_MAX_RETRIES = 3


# ── local SQLite pending queue ────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    _LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_LOCAL_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pending ("
        "  id       INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  payload  TEXT    NOT NULL,"
        "  created  TEXT    NOT NULL"
        ")"
    )
    conn.commit()
    return conn


def _enqueue(entry: ScoreEntry) -> None:
    conn = _db()
    conn.execute(
        "INSERT INTO pending (payload, created) VALUES (?, ?)",
        (json.dumps(entry.to_dict()), entry.timestamp),
    )
    conn.commit()
    conn.close()


def _dequeue_all() -> list[tuple[int, dict]]:
    conn = _db()
    rows = conn.execute("SELECT id, payload FROM pending ORDER BY id").fetchall()
    conn.close()
    return [(row[0], json.loads(row[1])) for row in rows]


def _remove_ids(ids: list[int]) -> None:
    if not ids:
        return
    conn = _db()
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM pending WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()


# ── GitHub Contents API helpers ───────────────────────────────────────────────

def _request(method: str, url: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body).encode() if body else None
    headers = {
        "Authorization": f"Bearer {_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "sunrice-leaderboard/1.0",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _read_remote() -> tuple[dict, str]:
    """Download leaderboard.json from GitHub.  Returns (parsed_content, sha).

    If the file does not yet exist (HTTP 404 on the file itself) returns an
    empty leaderboard with sha='' so the caller can create it on first write.
    """
    url = f"{_API_BASE}/repos/{_REPO}/contents/{_FILE}"
    try:
        result = _request("GET", url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Could be: file missing, repo missing, or token lacks access.
            # Try a repo-level ping to distinguish the two cases.
            try:
                _request("GET", f"{_API_BASE}/repos/{_REPO}")
                # Repo exists → the file is simply missing; let caller create it
                return {}, ""
            except urllib.error.HTTPError as exc2:
                if exc2.code in (401, 403, 404):
                    raise PermissionError(
                        f"Cannot access repo '{_REPO}' (HTTP {exc2.code}). "
                        "Check that SUNRICE_LB_TOKEN is set to a valid GitHub PAT "
                        "with 'Contents: Read and Write' on that repo."
                    ) from exc2
            raise
        if exc.code in (401, 403):
            raise PermissionError(
                f"GitHub returned HTTP {exc.code} for repo '{_REPO}'. "
                "Check your SUNRICE_LB_TOKEN."
            ) from exc
        raise
    content = json.loads(base64.b64decode(result["content"]).decode())
    return content, result["sha"]


def _write_remote(content: dict, sha: str, message: str) -> bool:
    """Upload leaderboard.json back to GitHub.

    Returns True on success, False on SHA conflict (concurrent write detected).
    Re-raises any other HTTP error.
    """
    url = f"{_API_BASE}/repos/{_REPO}/contents/{_FILE}"
    encoded = base64.b64encode(
        json.dumps(content, indent=2, ensure_ascii=False).encode()
    ).decode()
    body: dict = {"message": message, "content": encoded}
    if sha:   # empty sha → new file; omit sha field so GitHub creates the file
        body["sha"] = sha
    try:
        _request("PUT", url, body)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            return False  # stale SHA — caller should retry
        raise


def _push(entry: ScoreEntry) -> None:
    """Atomic read → append → write with exponential back-off on SHA conflicts."""
    info = GAME_REGISTRY.get(entry.game_id, {})
    for attempt in range(_MAX_RETRIES):
        content, sha = _read_remote()
        games = content.setdefault("games", {})
        game_entry = games.setdefault(
            entry.game_id,
            {
                "label":      info.get("label", entry.game_id),
                "score_type": info.get("score_type", "time_asc"),
                "scores":     [],
            },
        )
        game_entry["scores"].append(entry.to_dict())
        if _write_remote(content, sha, f"score: {entry.game_id} — {entry.player_name}"):
            return
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(
        f"Could not push score after {_MAX_RETRIES} retries (persistent SHA conflict)"
    )


# ── public API ────────────────────────────────────────────────────────────────

def submit_score(entry: ScoreEntry) -> bool:
    """Submit a score to the shared GitHub leaderboard.

    If ``SUNRICE_LB_TOKEN`` is not set or the network is unreachable, the
    score is stored locally and will be flushed on the next successful call.

    Also flushes any previously cached pending scores before submitting the
    current one (best-effort; failures are silently left in the queue).

    Returns
    -------
    bool
        ``True``  — score written to GitHub.
        ``False`` — score cached locally (will retry next time).
    """
    if not _TOKEN:
        print(
            "[leaderboard] SUNRICE_LB_TOKEN not set — caching score locally.\n"
            f"  (target repo: {_REPO})"
        )
        _enqueue(entry)
        return False

    # Best-effort flush of pending queue
    pending = _dequeue_all()
    flushed: list[int] = []
    for pid, payload in pending:
        try:
            _push(ScoreEntry(**payload))
            flushed.append(pid)
        except Exception:
            pass  # leave in queue; network still flaky
    _remove_ids(flushed)

    # Submit current score
    try:
        _push(entry)
        print(f"[leaderboard] Score submitted to {_REPO}.")
        return True
    except Exception as exc:
        print(f"[leaderboard] Submission failed ({exc}) — caching locally.")
        _enqueue(entry)
        return False


def fetch_scores(game_id: str, limit: int = 10) -> list[dict]:
    """Return the top *limit* scores for *game_id*, sorted by score_type.

    Returns an empty list if offline or if ``SUNRICE_LB_TOKEN`` is not set.
    """
    if not _TOKEN:
        return []
    try:
        content, _ = _read_remote()
        game_entry = content.get("games", {}).get(game_id, {})
        scores = game_entry.get("scores", [])
        score_type = game_entry.get(
            "score_type",
            GAME_REGISTRY.get(game_id, {}).get("score_type", "time_asc"),
        )
        reverse = score_type == "points_desc"
        return sorted(scores, key=lambda s: s["score"], reverse=reverse)[:limit]
    except Exception:
        return []
