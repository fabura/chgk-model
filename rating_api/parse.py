"""Parse api.rating.chgk.info JSON blobs into structures shaped like
rows of the rating-dump Postgres tables.

Mapping (API → public.*):

  Tournament-read.id            → tournaments.id
  Tournament-read.name          → tournaments.title
  Tournament-read.type.name     → tournaments.type   (Russian text;
                                                    matches dump verbatim)
  Tournament-read.type.id       → tournaments.typeoft_id
  Tournament-read.dateStart     → tournaments.start_datetime
  Tournament-read.dateEnd       → tournaments.end_datetime
  Tournament-read.lastEditDate  → tournaments.last_edited_at
  sum(questionQty.values())     → tournaments.questions_count
  maiiRating                    → tournaments.maii_rating

  Results[].team.id             → tournament_results.team_id
  Results[].mask                → tournament_results.points_mask
  Results[].position            → tournament_results.position
  Results[].questionsTotal      → tournament_results.total
  Results[].team.name           → tournament_results.team_title

  Results[].teamMembers[].player.id  → tournament_rosters.player_id
  Results[].teamMembers[].flag       → tournament_rosters.flag
                                       (Б/К/Л/null — verbatim;
                                        is_captain = (flag == 'К'))

  Tournament-read.editors[].id  → tournament_editors.player_id

The bigint `id` / sequence-backed columns (tournament_results.id,
tournament_rosters.id, …) are left for Postgres to assign on INSERT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedTournament:
    id: int
    title: str | None
    start_datetime: str | None       # ISO 8601, as returned by API
    end_datetime: str | None
    last_edited_at: str | None
    questions_count: int | None
    typeoft_id: int | None
    type: str | None                  # Russian, e.g. "Синхрон"
    maii_rating: bool | None


@dataclass
class ParsedResult:
    tournament_id: int
    team_id: int
    team_title: str | None
    total: int | None
    position: float | None
    points_mask: str | None


@dataclass
class ParsedRoster:
    tournament_id: int
    team_id: int
    player_id: int
    flag: str | None
    is_captain: bool


@dataclass
class ParsedEditor:
    tournament_id: int
    player_id: int


@dataclass
class ParsedTournamentBundle:
    """Everything we mirror for one tournament."""
    tournament: ParsedTournament
    results: list[ParsedResult] = field(default_factory=list)
    rosters: list[ParsedRoster] = field(default_factory=list)
    editors: list[ParsedEditor] = field(default_factory=list)


# ----------------------------------------------------------------------
# parsers
# ----------------------------------------------------------------------


def _questions_count(blob: dict[str, Any]) -> int | None:
    """questionQty in the API is a dict {tour_idx: n_questions}; PG stores
    a single integer = sum across tours."""
    qq = blob.get("questionQty")
    if isinstance(qq, dict):
        total = 0
        for v in qq.values():
            try:
                total += int(v)
            except (TypeError, ValueError):
                continue
        return total or None
    if isinstance(qq, int):
        return qq
    return None


def _type_fields(blob: dict[str, Any]) -> tuple[int | None, str | None]:
    """type can be either {id,name,shortName} (per /tournaments/{id}) or
    a bare int (per /tournaments listing)."""
    t = blob.get("type")
    if isinstance(t, dict):
        tid = t.get("id")
        return (int(tid) if isinstance(tid, int) else None, t.get("name"))
    if isinstance(t, int):
        return (t, None)
    return (None, None)


def parse_tournament_blob(blob: dict[str, Any]) -> ParsedTournament:
    tid = blob.get("id")
    if not isinstance(tid, int):
        raise ValueError(f"tournament blob missing integer id: {blob!r}")
    typeoft_id, type_name = _type_fields(blob)
    return ParsedTournament(
        id=int(tid),
        title=blob.get("name"),
        start_datetime=blob.get("dateStart"),
        end_datetime=blob.get("dateEnd"),
        last_edited_at=blob.get("lastEditDate"),
        questions_count=_questions_count(blob),
        typeoft_id=typeoft_id,
        type=type_name,
        maii_rating=blob.get("maiiRating") if isinstance(blob.get("maiiRating"), bool) else None,
    )


def parse_results_blob(
    tournament_id: int,
    rows: list[dict[str, Any]],
) -> tuple[list[ParsedResult], list[ParsedRoster]]:
    """Split /results payload into (per-team results, per-player rosters).

    Rows without team.id are dropped. Rows without teamMembers contribute
    a result but no roster entries (and downstream load_from_db will
    silently skip them — same as today's behavior with empty rosters).
    """
    results: list[ParsedResult] = []
    rosters: list[ParsedRoster] = []

    for row in rows:
        team = row.get("team") or {}
        team_id = team.get("id")
        if not isinstance(team_id, int):
            continue
        team_id = int(team_id)

        position = row.get("position")
        if position is not None:
            try:
                position = float(position)
            except (TypeError, ValueError):
                position = None

        total = row.get("questionsTotal")
        if total is not None:
            try:
                total = int(total)
            except (TypeError, ValueError):
                total = None

        results.append(
            ParsedResult(
                tournament_id=tournament_id,
                team_id=team_id,
                team_title=team.get("name"),
                total=total,
                position=position,
                points_mask=row.get("mask"),
            )
        )

        for m in row.get("teamMembers") or []:
            player = m.get("player") or {}
            pid = player.get("id")
            if not isinstance(pid, int):
                continue
            flag = m.get("flag")
            rosters.append(
                ParsedRoster(
                    tournament_id=tournament_id,
                    team_id=team_id,
                    player_id=int(pid),
                    flag=flag if isinstance(flag, str) else None,
                    is_captain=(flag == "К"),
                )
            )

    return results, rosters


def parse_editors(tournament: ParsedTournament, blob: dict[str, Any]) -> list[ParsedEditor]:
    out: list[ParsedEditor] = []
    for ed in blob.get("editors") or []:
        pid = ed.get("id") if isinstance(ed, dict) else None
        if isinstance(pid, int):
            out.append(ParsedEditor(tournament_id=tournament.id, player_id=int(pid)))
    return out
