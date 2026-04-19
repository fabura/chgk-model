"""
Sequential training engine.

Processes tournaments in chronological order, updating player strengths
and question parameters online via analytic gradients of the noisy-OR
log-likelihood.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rating.decay import apply_calendar_decay, apply_decay
from rating.model import forward, process_batch_nb
from rating.players import PlayerState
from rating.questions import QuestionState
from rating.tournaments import TournamentState, TYPE_ASYNC, TYPE_OFFLINE, TYPE_SYNC, game_type_to_idx


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class Config:
    """Tunable hyperparameters for the sequential rating loop.

    `w_online` retains its original meaning: async tournaments should
    affect player ratings less than offline events. Question updates and
    tournament mode effects have their own weights because async results
    are useful for learning the easier mode, but less reliable as direct
    evidence of player strength and question discrimination.
    """

    eta0: float = 0.10
    rho: float = 0.9995
    w_online: float = 0.5

    w_offline: float = 1.0
    w_sync: float = 0.9
    w_online_questions: float = 0.45
    w_online_log_a: float = 0.05
    w_sync_mode: float = 1.0
    w_async_mode: float = 0.3
    w_sync_residual: float = 0.9
    w_async_residual: float = 0.6

    eta_mu: float = 0.005
    eta_eps: float = 0.03
    reg_mu_type: float = 0.10
    reg_eps: float = 0.20

    # L2-style shrinkage on player/question parameters.  Default 0.0 keeps
    # the previous behaviour exactly; positive values pull toward zero
    # after each gradient step (see ``process_batch_nb``).
    reg_theta: float = 0.0
    reg_b: float = 0.0
    reg_log_a: float = 0.0

    # Calendar-based decay.  When ``use_calendar_decay`` is True the
    # global per-tournament ``apply_decay(theta, rho)`` is replaced with
    # a per-player decay: ``theta_k *= rho_calendar ** (Δdays /
    # decay_period_days)`` applied lazily to each player when they next
    # appear in a tournament.  This decouples decay from the dataset
    # tournament cadence (which is highly uneven across weeks).
    #
    # Empirically (see ``docs/calendar_decay_experiments.md``), the per-
    # tournament global decay ``rho=0.9995`` was over-aggressive on the
    # rating-DB cadence (~21 tournaments/week × ~8 years ⇒ effective
    # multiplier ≈ 0.014).  Replacing it with calendar decay
    # (``rho_calendar=1.0`` ⇒ no decay) cut backtest logloss from 0.602
    # to 0.532.  ``rho_calendar=1.0`` is the new default; lower values
    # introduce mild long-term shrinkage at a small accuracy cost.
    use_calendar_decay: bool = True
    rho_calendar: float = 1.0  # 1.0 = disable; <1 = per-week multiplicative decay
    decay_period_days: float = 7.0

    # Cold-start shrinkage for first-time players.  ``cold_init_factor``
    # multiplies the team-mean θ when initialising a new player.  1.0
    # reproduces the previous behaviour ("inherit team average"); values
    # < 1.0 protect against a strong roster instantly inflating a
    # rookie's rating.
    cold_init_factor: float = 1.0

    # Cold-start prior θ for first-time players.  When
    # ``cold_init_use_team_mean=False`` every newcomer starts at
    # ``cold_init_theta`` regardless of their team.  Combined with
    # ``games_offset`` < 1 (rookie boost), this breaks the positive-
    # feedback loop where weak rookies → lower team means → even lower
    # starting θ for the next rookies, which causes the multi-year θ
    # drift visible on population plots.
    #
    # When ``cold_init_use_team_mean=True`` (default, legacy) the prior
    # is blended with the team mean as
    #     θ_new = cold_init_factor·mean(team) + (1−cold_init_factor)·prior
    # and reduces to "no teammates → θ_new = prior".
    #
    # Tuned defaults (see ``scripts/exp_cold_start_grid.py`` and
    # ``docs/cold_start_experiments.md``): a fixed prior of −1.0 with the
    # rookie boost gave the lowest backtest logloss (0.5129) over a 12-
    # cell sweep on the 2018-04 → 2025-12 hold-out, a ~1.3 % improvement
    # over the legacy team-mean inheritance, and effectively eliminated
    # the long-term drift of the top-1000 player median.
    cold_init_theta: float = -1.0
    cold_init_use_team_mean: bool = False

    # Adaptive learning-rate offset.  η_k = η0 / √(games_offset + games_k).
    # Default 0.25 gives a chess-Elo-style "rookie boost": at games=0 the
    # learning rate is η0/√0.25 = 2·η0, which lets the model quickly
    # find the right level for newcomers initialised at the fixed prior.
    # The previous default 1.0 (η_k = η0/√(1+games)) produced the same
    # asymptotic behaviour but reacted ~2× slower in the first 5 games.
    games_offset: float = 0.25

    # Team-size effect.  Adds a global, per-team-size shift to the
    # tournament difficulty:
    #     δ = mu_type[type] + eps[t] + delta_size[clip(team_size, 1, K)]
    # Anchored at ``team_size_anchor`` (delta_size at that index is
    # forced to zero so the parameter is identifiable against the
    # weekly-centered ε_t).  ``team_size_max`` clips uncommon roster
    # sizes (default 8 covers ~98% of the data; sizes 9+ are rare and
    # often roster errors).  ``w_size`` allows shutting the effect off
    # for async tournaments where rosters are noisier.
    use_team_size_effect: bool = True
    team_size_max: int = 8
    team_size_anchor: int = 6
    eta_size: float = 0.005
    reg_size: float = 0.10
    w_size_offline: float = 1.0
    w_size_sync: float = 1.0
    w_size_async: float = 0.5

    # Position-in-tour effect.  Adds a small per-position shift on
    # tournament difficulty:
    #     δ += delta_pos[(question_index_within_tournament) % tour_len]
    # Anchored at ``pos_anchor`` (mid-tour).  Default ``tour_len=12``
    # matches the standard ChGK tour length; tournaments with non-12
    # tour structures (≈30% of the data) get a noisier signal but the
    # effect is still informative on average.
    use_pos_effect: bool = True
    tour_len: int = 12
    # Anchored at position 0 (the easiest in the empirical take-rate
    # curve), so all other δ_pos values stay positive and the learned
    # vector reads naturally as "extra difficulty over question 1".
    pos_anchor: int = 0
    eta_pos: float = 0.005
    reg_pos: float = 0.10
    w_pos_offline: float = 1.0
    w_pos_sync: float = 1.0
    w_pos_async: float = 0.5

    use_tournament_delta: bool = True
    use_delta_type_prior: bool = False


# ======================================================================
# Result container
# ======================================================================

@dataclass
class SequentialResult:
    """Output of :func:`run_sequential`."""

    players: PlayerState
    questions: QuestionState
    total_loglik: float
    total_obs: int
    predictions: Optional[dict] = None
    history: Optional[list] = None
    canonical_q_map: Optional[np.ndarray] = None
    tournaments: Optional[TournamentState] = None
    delta_size: Optional[np.ndarray] = None  # learned team-size effect (index = team_size)
    team_size_anchor: int = 6
    delta_pos: Optional[np.ndarray] = None  # learned position-in-tour effect
    pos_anchor: int = 6


# ======================================================================
# Helpers
# ======================================================================

def _type_update_weights(
    game_type: str, cfg: Config
) -> tuple[float, float, float, float, float, float, float]:
    """Return per-parameter update weights for a tournament type.

    Returns ``(theta_w, b_w, log_a_w, mu_w, eps_w, size_w, pos_w)``.
    """
    if "async" in game_type:
        return (
            cfg.w_online,
            cfg.w_online_questions,
            cfg.w_online_log_a,
            cfg.w_async_mode,
            cfg.w_async_residual,
            cfg.w_size_async,
            cfg.w_pos_async,
        )
    if "sync" in game_type:
        return (
            cfg.w_sync,
            cfg.w_sync,
            cfg.w_sync,
            cfg.w_sync_mode,
            cfg.w_sync_residual,
            cfg.w_size_sync,
            cfg.w_pos_sync,
        )
    return (
        cfg.w_offline,
        cfg.w_offline,
        cfg.w_offline,
        0.0,
        cfg.w_offline,
        cfg.w_size_offline,
        cfg.w_pos_offline,
    )


def _prepare_arrays(
    arrays: dict[str, np.ndarray],
    maps,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise dtypes and derive game_idx / offsets if needed."""
    q_idx = arrays["q_idx"].astype(np.int32)
    taken = arrays["taken"].astype(np.float64)
    team_sizes = arrays["team_sizes"].astype(np.int32)
    player_flat = arrays["player_indices_flat"].astype(np.int32)

    game_idx = arrays.get("game_idx")
    if game_idx is None:
        qg = getattr(maps, "question_game_idx", None)
        if qg is not None:
            game_idx = qg[q_idx]
        else:
            raise ValueError(
                "game_idx not available in arrays or maps; "
                "sequential processing requires tournament assignment"
            )
    game_idx = game_idx.astype(np.int32)

    offsets = np.zeros(len(q_idx) + 1, dtype=np.int64)
    np.cumsum(team_sizes, out=offsets[1:])

    return q_idx, taken, team_sizes, player_flat, game_idx, offsets


# ======================================================================
# Main loop
# ======================================================================

def run_sequential(
    arrays: dict[str, np.ndarray],
    maps,
    cfg: Config = Config(),
    *,
    verbose: bool = True,
    collect_history: bool = False,
    collect_predictions: bool = False,
) -> SequentialResult:
    """Process tournaments chronologically, updating parameters online.

    Parameters
    ----------
    arrays : dict
        Packed observation arrays (q_idx, taken, team_sizes,
        player_indices_flat, and optionally game_idx).
    maps : IndexMaps
        Index maps with game metadata (game_date_ordinal, game_type, …).
    cfg : Config
        Hyperparameters.
    verbose : bool
        Print progress and summary.
    collect_history : bool
        Record (player_id, game_id, θ) after each tournament.
    collect_predictions : bool
        Record per-observation predictions made *before* parameter
        updates (needed for backtesting).

    Returns
    -------
    SequentialResult
    """
    q_idx, taken, team_sizes, player_flat, game_idx, offsets = (
        _prepare_arrays(arrays, maps)
    )
    n_obs = len(q_idx)
    num_players = maps.num_players
    num_questions = maps.num_questions

    # Canonical question mapping for paired tournaments (sync+async
    # sharing the same question package).  When available, multiple raw
    # question indices map to the same canonical index so they share b/a.
    _cq_raw = getattr(maps, "canonical_q_idx", None)
    _ncq = getattr(maps, "num_canonical_questions", None)
    if _cq_raw is not None and _ncq is not None and _ncq < num_questions:
        cq = _cq_raw.astype(np.int32)
        num_q_params = _ncq
    else:
        cq = np.arange(num_questions, dtype=np.int32)
        num_q_params = num_questions

    def _cqi(raw_qi: int) -> int:
        """Map raw question index → canonical (shared) index."""
        return int(cq[raw_qi])

    gdo = getattr(maps, "game_date_ordinal", None)
    gtype_arr = getattr(maps, "game_type", None)
    game_types: list[str] = (
        [str(x) for x in gtype_arr]
        if gtype_arr is not None
        else ["offline"] * len(getattr(maps, "idx_to_game_id", []))
    )

    # --- group observations by game ---
    obs_by_game: dict[int, list[int]] = defaultdict(list)
    for i in range(n_obs):
        obs_by_game[int(game_idx[i])].append(i)

    def _sort_key(g: int) -> tuple:
        if gdo is not None and g < len(gdo) and int(gdo[g]) >= 0:
            return (1, int(gdo[g]))
        return (0, g)

    game_order = sorted(obs_by_game, key=_sort_key)

    # --- state ---
    num_games = len(game_types) if game_types else (max(game_order) + 1 if game_order else 0)
    if cfg.use_tournament_delta:
        tournaments = TournamentState(
            num_games,
            game_type=game_types,
            use_type_prior=cfg.use_delta_type_prior,
        )
    else:
        tournaments = None

    players = PlayerState(num_players)
    questions = QuestionState(num_q_params)

    # Track games per week for centering δ
    last_week: int | None = None
    games_this_week: list[int] = []

    total_loglik = 0.0
    total_obs_count = 0
    zero_mu_type = np.zeros(3, dtype=np.float64)
    zero_eps = np.zeros(max(num_games, 1), dtype=np.float64)

    # Team-size effect: vector of length team_size_max + 1 (index 0 unused),
    # initialised to zero so the model starts identical to the previous one.
    # Anchored at team_size_anchor (delta_size at that index stays zero).
    team_size_max = max(2, int(cfg.team_size_max))
    team_size_anchor = max(1, min(int(cfg.team_size_anchor), team_size_max))
    delta_size = np.zeros(team_size_max + 1, dtype=np.float64)
    if not cfg.use_team_size_effect:
        eta_size_eff = 0.0
    else:
        eta_size_eff = float(cfg.eta_size)

    # Position-in-tour effect: vector of length tour_len, indexed by
    # raw_question_index_within_tournament % tour_len.  Anchored at
    # pos_anchor (delta_pos at that index stays zero).
    tour_len = max(2, int(cfg.tour_len))
    pos_anchor = max(0, min(int(cfg.pos_anchor), tour_len - 1))
    delta_pos = np.zeros(tour_len, dtype=np.float64)
    if not cfg.use_pos_effect:
        eta_pos_eff = 0.0
    else:
        eta_pos_eff = float(cfg.eta_pos)

    # Pre-compute per-(raw)-question position-in-tour from
    # idx_to_question_id, which stores (tournament_id, question_index)
    # tuples.  Falls back to ``q_raw % tour_len`` if the structure
    # differs (e.g. flat ints).
    qids = getattr(maps, "idx_to_question_id", None)
    q_pos_in_tour = np.zeros(num_questions, dtype=np.int32)
    if qids is not None and len(qids) > 0 and isinstance(qids[0], tuple):
        for raw_qi in range(min(num_questions, len(qids))):
            q_pos_in_tour[raw_qi] = int(qids[raw_qi][1]) % tour_len
    else:
        for raw_qi in range(num_questions):
            q_pos_in_tour[raw_qi] = raw_qi % tour_len

    history: list | None = [] if collect_history else None
    pred_p_list: list[float] | None = [] if collect_predictions else None
    pred_y_list: list[int] | None = [] if collect_predictions else None
    pred_g_list: list[int] | None = [] if collect_predictions else None

    try:
        from tqdm import tqdm
        game_iter = (
            tqdm(game_order, desc="Tournaments", unit=" tourn")
            if verbose
            else game_order
        )
    except ImportError:
        game_iter = game_order

    for g in game_iter:
        obs_indices = obs_by_game[g]
        gt = game_types[g] if g < len(game_types) else "offline"
        theta_w, b_w, log_a_w, mu_w, eps_w, size_w, pos_w = _type_update_weights(gt, cfg)
        gt_idx = game_type_to_idx(gt)

        # 0. Week boundary: center tournament residuals ------------------
        if tournaments is not None:
            current_week = (
                int(gdo[g]) // 7
                if gdo is not None and g < len(gdo) and int(gdo[g]) >= 0
                else g
            )
            if last_week is not None and current_week != last_week and games_this_week:
                tournaments.center(games_this_week)
                games_this_week = []
            last_week = current_week

        # 1. Decay -------------------------------------------------------
        # Calendar-based decay is applied later, *after* we collect the
        # set of players appearing in this tournament (so it touches
        # only those players and uses their personal Δdays).  When
        # disabled, fall back to the original global decay.
        if not cfg.use_calendar_decay:
            apply_decay(players.theta, cfg.rho)

        # 2. Initialise unseen players from teammates --------------------
        for i in obs_indices:
            s, e = int(offsets[i]), int(offsets[i + 1])
            pids = player_flat[s:e]
            for pidx in pids:
                if not players.seen[int(pidx)]:
                    players.initialize_new(
                        int(pidx),
                        pids.tolist(),
                        cold_factor=cfg.cold_init_factor,
                        prior=cfg.cold_init_theta,
                        use_team_mean=cfg.cold_init_use_team_mean,
                    )

        # 3. Initialise unseen questions from empirical take rates --------
        #    Keyed by canonical index so paired tournaments share params.
        q_takes: dict[int, list[int]] = defaultdict(list)
        for i in obs_indices:
            q_takes[_cqi(int(q_idx[i]))].append(int(taken[i]))
        for qi, takes in q_takes.items():
            if not questions.initialized[qi]:
                questions.init_from_take_rate(qi, sum(takes) / len(takes))

        # 4. Record predictions BEFORE updating --------------------------
        if collect_predictions:
            delta_g_base = tournaments.total_delta(g) if tournaments is not None else 0.0
            for i in obs_indices:
                qi = _cqi(int(q_idx[i]))
                s, e = int(offsets[i]), int(offsets[i + 1])
                th = players.theta[player_flat[s:e]]
                a_val = math.exp(
                    max(min(questions.log_a[qi], 3.0), -3.0)
                )
                ts_raw = e - s
                if ts_raw < 1:
                    ts_idx = 1
                elif ts_raw > team_size_max:
                    ts_idx = team_size_max
                else:
                    ts_idx = ts_raw
                pos_idx_pred = int(q_pos_in_tour[int(q_idx[i])])
                delta_g = delta_g_base
                if cfg.use_team_size_effect and ts_idx != team_size_anchor:
                    delta_g += float(delta_size[ts_idx])
                if cfg.use_pos_effect and pos_idx_pred != pos_anchor:
                    delta_g += float(delta_pos[pos_idx_pred])
                p, _, _ = forward(th, questions.b[qi], a_val, delta=delta_g)
                pred_p_list.append(p)
                pred_y_list.append(int(taken[i]))
                pred_g_list.append(g)

        # 5. Sequential updates (Numba-accelerated batch) ----------------
        by_q: dict[int, list[int]] = defaultdict(list)
        for i in obs_indices:
            by_q[int(q_idx[i])].append(i)
        obs_order = []
        for qi_raw in sorted(by_q):
            obs_order.extend(by_q[qi_raw])
        tourn_players: set[int] = set()
        for i in obs_indices:
            s, e = int(offsets[i]), int(offsets[i + 1])
            tourn_players.update(int(p) for p in player_flat[s:e])

        # 4b. Calendar-based decay: only touch participating players,
        #     proportional to their own days-since-last-game.
        if cfg.use_calendar_decay:
            current_ord = (
                int(gdo[g])
                if gdo is not None and g < len(gdo) and int(gdo[g]) >= 0
                else -1
            )
            if current_ord >= 0:
                pids_arr = np.fromiter(tourn_players, dtype=np.int64, count=len(tourn_players))
                apply_calendar_decay(
                    players.theta,
                    players.last_seen_ordinal,
                    current_ord,
                    pids_arr,
                    cfg.rho_calendar,
                    cfg.decay_period_days,
                )
                players.last_seen_ordinal[pids_arr] = current_ord

        mu_type_arr = tournaments.mu_type if tournaments is not None else zero_mu_type
        eps_arr = tournaments.eps if tournaments is not None else zero_eps
        obs_arr = np.array(obs_order, dtype=np.int64)
        total_loglik += process_batch_nb(
            obs_arr,
            offsets,
            player_flat,
            q_idx,
            taken,
            cq,
            q_pos_in_tour,
            players.theta,
            questions.b,
            questions.log_a,
            mu_type_arr,
            eps_arr,
            delta_size,
            delta_pos,
            players.games,
            g,
            gt_idx,
            cfg.eta0,
            theta_w,
            b_w,
            log_a_w,
            mu_w if tournaments is not None else 0.0,
            eps_w if tournaments is not None else 0.0,
            size_w if cfg.use_team_size_effect else 0.0,
            pos_w if cfg.use_pos_effect else 0.0,
            cfg.eta_mu if tournaments is not None else 0.0,
            cfg.eta_eps if tournaments is not None else 0.0,
            eta_size_eff,
            eta_pos_eff,
            cfg.reg_mu_type if tournaments is not None else 0.0,
            cfg.reg_eps if tournaments is not None else 0.0,
            cfg.reg_size,
            cfg.reg_pos,
            team_size_anchor,
            pos_anchor,
            cfg.reg_theta,
            cfg.reg_b,
            cfg.reg_log_a,
            20.0,
            cfg.games_offset,
        )
        total_obs_count += len(obs_order)

        np.clip(players.theta, -10.0, 10.0, out=players.theta)

        if tournaments is not None:
            games_this_week.append(g)

        # 6. Increment game counters -------------------------------------
        for pidx in tourn_players:
            players.games[pidx] += 1

        # 7. History snapshot --------------------------------------------
        if history is not None:
            gid = (
                maps.idx_to_game_id[g]
                if g < len(maps.idx_to_game_id)
                else g
            )
            for pidx in sorted(tourn_players):
                pid = (
                    maps.idx_to_player_id[pidx]
                    if pidx < len(maps.idx_to_player_id)
                    else pidx
                )
                history.append((pid, gid, float(players.theta[pidx])))

    # Center tournament residuals one last time (final week)
    if tournaments is not None and games_this_week:
        tournaments.center(games_this_week)

    # === Assemble predictions ==========================================
    predictions = None
    if collect_predictions and pred_p_list:
        predictions = {
            "pred_p": np.array(pred_p_list, dtype=np.float64),
            "actual_y": np.array(pred_y_list, dtype=np.int32),
            "game_idx": np.array(pred_g_list, dtype=np.int32),
        }

    # === Summary =======================================================
    if verbose:
        avg = total_loglik / max(total_obs_count, 1)
        shared = num_questions - num_q_params
        pair_msg = (
            f", {num_q_params} canonical questions "
            f"({shared} shared across paired tournaments)"
            if shared > 0
            else ""
        )
        print(
            f"\nDone: {len(game_order)} tournaments, "
            f"{total_obs_count} observations{pair_msg}"
        )
        print(f"Average log-likelihood: {avg:.4f}  (logloss: {-avg:.4f})")
        if tournaments is not None:
            print(
                "Mode offsets:"
                f" offline={tournaments.mu_type[TYPE_OFFLINE]:+.4f}"
                f" sync={tournaments.mu_type[TYPE_SYNC]:+.4f}"
                f" async={tournaments.mu_type[TYPE_ASYNC]:+.4f}"
            )
        if cfg.use_team_size_effect:
            parts = []
            for n in range(1, team_size_max + 1):
                marker = "*" if n == team_size_anchor else " "
                parts.append(f"n={n}{marker}{float(delta_size[n]):+.3f}")
            print("Team-size effects (δ added to difficulty; * = anchor at 0):")
            print("  " + "  ".join(parts))
        if cfg.use_pos_effect:
            parts = []
            for p in range(tour_len):
                marker = "*" if p == pos_anchor else " "
                parts.append(f"p={p:>2}{marker}{float(delta_pos[p]):+.3f}")
            print(
                f"Position-in-tour effects (tour_len={tour_len}; "
                f"δ added to difficulty; * = anchor at 0):"
            )
            print("  " + "  ".join(parts))

        # Parameter-space diagnostics: how many params hit the hard
        # clamps?  A large fraction signals saturation — usually
        # caused by too-large lr, too-weak regularisation, or
        # genuinely under-identified questions.
        init_q = questions.initialized
        if init_q.any():
            b_init = questions.b[init_q]
            la_init = questions.log_a[init_q]
            n_q = int(init_q.sum())
            b_clamp = int(np.sum(np.abs(b_init) >= 9.999))
            la_clamp = int(np.sum(np.abs(la_init) >= 2.999))
            print(
                "\nParameter diagnostics:"
                f"\n  b      : mean={float(b_init.mean()):+.3f} "
                f"std={float(b_init.std()):.3f} "
                f"clamped={b_clamp}/{n_q} ({100.0 * b_clamp / n_q:.2f}%)"
                f"\n  log_a  : mean={float(la_init.mean()):+.3f} "
                f"std={float(la_init.std()):.3f} "
                f"clamped={la_clamp}/{n_q} ({100.0 * la_clamp / n_q:.2f}%)"
            )
        seen_p = players.seen
        if seen_p.any():
            th_seen = players.theta[seen_p]
            n_p = int(seen_p.sum())
            th_clamp = int(np.sum(np.abs(th_seen) >= 9.999))
            print(
                f"  theta  : mean={float(th_seen.mean()):+.3f} "
                f"std={float(th_seen.std()):.3f} "
                f"clamped={th_clamp}/{n_p} ({100.0 * th_clamp / n_p:.2f}%)"
            )

        experienced = players.games >= 30
        if experienced.any():
            exp_idx = np.where(experienced)[0]
            top = exp_idx[np.argsort(players.theta[exp_idx])[::-1][:10]]
            print(f"\nTop 10 players (≥ 30 games):")
            for rank, idx in enumerate(top, 1):
                pid = (
                    maps.idx_to_player_id[idx]
                    if idx < len(maps.idx_to_player_id)
                    else idx
                )
                print(
                    f"  {rank:2d}. id={pid}  "
                    f"θ={players.theta[idx]:.4f}  "
                    f"games={players.games[idx]}"
                )

    return SequentialResult(
        players=players,
        questions=questions,
        total_loglik=total_loglik,
        total_obs=total_obs_count,
        predictions=predictions,
        history=history,
        canonical_q_map=cq if num_q_params < num_questions else None,
        tournaments=tournaments,
        delta_size=delta_size if cfg.use_team_size_effect else None,
        team_size_anchor=team_size_anchor,
        delta_pos=delta_pos if cfg.use_pos_effect else None,
        pos_anchor=pos_anchor,
    )
