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


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class Config:
    """Tunable hyperparameters for the sequential rating loop.

    `w_online` retains its original meaning: async tournaments should
    affect player ratings less than offline events. Question updates
    have their own weights because async results are still useful for
    learning question difficulty/discrimination, just noisier.
    """

    # 2026-04 lean defaults.  In April 2026 we removed the per-mode
    # offset μ_type and the per-tournament residual ε_t (8 746 params)
    # after an ablation showed they were net-negative for backtest
    # quality, then added a small teammate-θ shrinkage and re-tuned
    # eta0 for the leaner model.  Cumulative gain over the previous
    # defaults on the 20 % hold-out: logloss 0.5365 → 0.5309
    # (-0.0056), AUC 0.8065 → 0.8115 (+0.0050).  See `/tmp/exp_*` and
    # the cleanup commit message for the full sweep table.
    eta0: float = 0.07
    rho: float = 0.9995
    w_online: float = 0.5

    w_offline: float = 1.0
    w_sync: float = 0.7
    w_online_questions: float = 0.30
    w_online_log_a: float = 0.05

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
    # question difficulty:
    #     δ = delta_size[clip(team_size, 1, K)]
    # Anchored at ``team_size_anchor`` (delta_size at that index is
    # forced to zero for identifiability).  ``team_size_max`` clips
    # uncommon roster sizes (default 8 covers ~98% of the data; sizes
    # 9+ are rare and often roster errors).  ``w_size`` allows shutting
    # the effect off for async tournaments where rosters are noisier.
    use_team_size_effect: bool = True
    team_size_max: int = 8
    team_size_anchor: int = 6
    eta_size: float = 0.005
    reg_size: float = 0.10
    w_size_offline: float = 1.0
    w_size_sync: float = 1.0
    w_size_async: float = 0.5

    # Solo-mode update weights.  When ``use_solo_channel=True``,
    # samples with ``team_size == 1`` are routed through a separate
    # "solo" channel regardless of the tournament's nominal type
    # (offline/sync/async).  This isolates the population of online
    # solo quizzes (M-Лига, "Гостиный двор", etc.) whose take-rate
    # distribution is sharply different from 5–6-player team play and
    # would otherwise inflate strong soloists' θ via the noisy-OR
    # identifiability shortcut.
    #
    # Default ``use_solo_channel=False`` reproduces the legacy
    # behaviour exactly (solo samples use their tournament-type
    # weights).  Flip to True together with the ``w_solo*`` knobs
    # below to opt in.
    #
    # Recommended starting weights (when opted in):
    #   - ``w_solo`` ≪ ``w_offline`` so a single solo result tugs θ
    #     much less than a team result;
    #   - ``w_solo_questions`` / ``w_solo_log_a`` = 0 so the narrow,
    #     self-selected population of soloists does not bias question
    #     difficulty / discrimination estimates;
    #   - ``w_size_solo`` = 1 so ``delta_size[1]`` is still learned
    #     (otherwise it stays at 0 and breaks the noisy-OR forward
    #     pass for solo predictions);
    #   - ``w_pos_solo`` = 0 (positional structure on solo packs is
    #     atypical — most are 36-question online quizzes).
    use_solo_channel: bool = False
    w_solo: float = 0.1
    w_solo_questions: float = 0.0
    w_solo_log_a: float = 0.0
    w_size_solo: float = 1.0
    w_pos_solo: float = 0.0

    # Position-in-tour effect.  Adds a small per-position shift on
    # question difficulty:
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

    # Per-tournament teammate shrinkage.  After each gradient step,
    # pull every roster member's θ toward the per-team mean of θ on
    # this tournament:
    #     θ_k -= eta_teammate * (θ_k - mean_team(θ))
    # Motivation: noisy-OR on team-level data has an identifiability
    # problem for stable rosters — a small early θ-gap between
    # long-term teammates is otherwise locked in for life because the
    # credit attribution is proportional to current λ_k.  This
    # shrinkage adds a soft pull that lets joint games slowly close
    # such artificial intra-team gaps while preserving real differences
    # (which are independently supported by solo games).  0.005 was
    # picked from a sweep (logloss −0.0014, AUC +0.0011 over 0.0).
    eta_teammate: float = 0.005

    # Periodic gauge re-centering to neutralise the multi-year θ drift.
    #
    # The noisy-OR model is gauge-invariant: shifting θ → θ+Δ and
    # b → b + a·Δ leaves predictions unchanged.  Without an anchor the
    # cold-start of new rookies (every newcomer enters at
    # ``cold_init_theta=-1.0``) slowly drags the population mean
    # downward over years, so a top player's θ in 2017 (≈ 1.4) and in
    # 2026 (≈ 1.0) are not directly comparable even when their relative
    # rank is similar.
    #
    # When ``recenter_period_days > 0`` we apply, every ~year, a
    # gauge transform that pins the median θ of "active veterans"
    # (``games >= recenter_min_games`` and seen within
    # ``recenter_active_days``) to ``recenter_target``:
    #   Δ = recenter_target - median(θ_active_veterans)
    #   θ_k    += Δ            for every player
    #   b_i    += a_i · Δ      for every (canonical) question
    # Predictions are exactly invariant; the only effect is to keep the
    # absolute θ scale stable across years.
    recenter_period_days: float = 365.0  # 0 = disable
    recenter_min_games: int = 200
    recenter_active_days: int = 365
    recenter_target: float = -0.70  # tuned via backtest sweep (best logloss/AUC)


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
    delta_size: Optional[np.ndarray] = None  # learned team-size effect (index = team_size)
    team_size_anchor: int = 6
    delta_pos: Optional[np.ndarray] = None  # learned position-in-tour effect
    pos_anchor: int = 6
    # Yearly gauge re-centering events: list of (ord_day, median_before,
    # delta_applied, n_active_veterans).  Used by build_db.py to retroactively
    # bring all historical θ rows into a single (final) gauge so the displayed
    # player-history graph does not show a one-time "shr" cliff at the date of
    # the first re-center event.
    recenter_events: Optional[list[tuple[int, float, float, int]]] = None


# ======================================================================
# Helpers
# ======================================================================

def _type_update_weights(
    game_type: str, cfg: Config
) -> tuple[float, float, float, float, float]:
    """Return per-parameter update weights for a tournament type.

    Returns ``(theta_w, b_w, log_a_w, size_w, pos_w)``.
    """
    if "async" in game_type:
        return (
            cfg.w_online,
            cfg.w_online_questions,
            cfg.w_online_log_a,
            cfg.w_size_async,
            cfg.w_pos_async,
        )
    if "sync" in game_type:
        return (
            cfg.w_sync,
            cfg.w_sync,
            cfg.w_sync,
            cfg.w_size_sync,
            cfg.w_pos_sync,
        )
    return (
        cfg.w_offline,
        cfg.w_offline,
        cfg.w_offline,
        cfg.w_size_offline,
        cfg.w_pos_offline,
    )


def _solo_update_weights(
    cfg: Config,
) -> tuple[float, float, float, float, float]:
    """Return per-parameter update weights for solo (team_size==1) samples.

    Solo samples are routed through their own channel regardless of
    the tournament's nominal type — see ``Config.w_solo``.
    """
    return (
        cfg.w_solo,
        cfg.w_solo_questions,
        cfg.w_solo_log_a,
        cfg.w_size_solo,
        cfg.w_pos_solo,
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
    players = PlayerState(num_players)
    questions = QuestionState(num_q_params)

    # Track epoch boundary for periodic gauge re-centering (drift fix).
    last_recenter_epoch: int | None = None
    recenter_period = float(getattr(cfg, "recenter_period_days", 0.0) or 0.0)
    recenter_enabled = recenter_period > 0
    recenter_log: list[tuple[int, float, float, int]] = []

    total_loglik = 0.0
    total_obs_count = 0

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
        theta_w, b_w, log_a_w, size_w, pos_w = _type_update_weights(gt, cfg)

        # 0b. Year/epoch boundary: gauge re-center to keep median θ of
        #     active veterans pinned to cfg.recenter_target.  Strictly a
        #     gauge transform (θ ↑Δ, b ↑a·Δ): predictions are invariant.
        if recenter_enabled and gdo is not None and g < len(gdo) and int(gdo[g]) >= 0:
            current_ord_for_year = int(gdo[g])
            current_epoch = int(current_ord_for_year // recenter_period)
            if last_recenter_epoch is None:
                last_recenter_epoch = current_epoch
            elif current_epoch > last_recenter_epoch:
                cutoff_ord = current_ord_for_year - cfg.recenter_active_days
                seen_mask = players.seen if hasattr(players, "seen") else None
                games_arr = players.games
                last_seen_arr = players.last_seen_ordinal
                active_mask = (
                    (games_arr >= cfg.recenter_min_games)
                    & (last_seen_arr >= cutoff_ord)
                )
                if seen_mask is not None:
                    active_mask &= seen_mask
                n_active = int(active_mask.sum())
                if n_active >= 50:
                    med = float(np.median(players.theta[active_mask]))
                    delta = float(cfg.recenter_target) - med
                    if abs(delta) > 1e-9:
                        players.theta += delta
                        # b → b + a·Δ to preserve predictions exactly.
                        a_vals = np.exp(np.clip(questions.log_a, -3.0, 3.0))
                        questions.b += a_vals * delta
                        recenter_log.append(
                            (current_ord_for_year, med, delta, n_active)
                        )
                        if verbose:
                            print(
                                f"  [recenter] ord={current_ord_for_year} "
                                f"epoch={current_epoch} "
                                f"median_active={med:+.4f} → target={cfg.recenter_target:+.2f} "
                                f"Δ={delta:+.4f} n_active={n_active}"
                            )
                last_recenter_epoch = current_epoch

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
                delta_g = 0.0
                if cfg.use_team_size_effect and ts_idx != team_size_anchor:
                    delta_g += float(delta_size[ts_idx])
                if cfg.use_pos_effect and pos_idx_pred != pos_anchor:
                    delta_g += float(delta_pos[pos_idx_pred])
                p, _, _ = forward(th, questions.b[qi], a_val, delta=delta_g)
                pred_p_list.append(p)
                pred_y_list.append(int(taken[i]))
                pred_g_list.append(g)

        # 5. Sequential updates (Numba-accelerated batch) ----------------
        # When ``use_solo_channel`` is enabled, samples with
        # team_size==1 get a separate update pass with their own
        # (theta_w, b_w, log_a_w, size_w, pos_w) weights.  Otherwise
        # everything flows through the legacy single-batch path.
        by_q: dict[int, list[int]] = defaultdict(list)
        for i in obs_indices:
            by_q[int(q_idx[i])].append(i)
        obs_order_team: list[int] = []
        obs_order_solo: list[int] = []
        if cfg.use_solo_channel:
            for qi_raw in sorted(by_q):
                for i in by_q[qi_raw]:
                    if int(team_sizes[i]) == 1:
                        obs_order_solo.append(i)
                    else:
                        obs_order_team.append(i)
        else:
            for qi_raw in sorted(by_q):
                obs_order_team.extend(by_q[qi_raw])
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

        if obs_order_team:
            obs_arr = np.array(obs_order_team, dtype=np.int64)
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
                delta_size,
                delta_pos,
                players.games,
                cfg.eta0,
                theta_w,
                b_w,
                log_a_w,
                size_w if cfg.use_team_size_effect else 0.0,
                pos_w if cfg.use_pos_effect else 0.0,
                eta_size_eff,
                eta_pos_eff,
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
            total_obs_count += len(obs_order_team)
        if obs_order_solo:
            solo_theta_w, solo_b_w, solo_log_a_w, solo_size_w, solo_pos_w = (
                _solo_update_weights(cfg)
            )
            obs_arr_solo = np.array(obs_order_solo, dtype=np.int64)
            total_loglik += process_batch_nb(
                obs_arr_solo,
                offsets,
                player_flat,
                q_idx,
                taken,
                cq,
                q_pos_in_tour,
                players.theta,
                questions.b,
                questions.log_a,
                delta_size,
                delta_pos,
                players.games,
                cfg.eta0,
                solo_theta_w,
                solo_b_w,
                solo_log_a_w,
                solo_size_w if cfg.use_team_size_effect else 0.0,
                solo_pos_w if cfg.use_pos_effect else 0.0,
                eta_size_eff,
                eta_pos_eff,
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
            total_obs_count += len(obs_order_solo)

        np.clip(players.theta, -10.0, 10.0, out=players.theta)

        # 5b. Teammate θ-shrinkage (experimental, see Config docstring).
        # One pull per (team, tournament): collect distinct rosters from
        # observations and shrink each roster's θ toward its mean.
        if cfg.eta_teammate > 0.0:
            seen_rosters: set[tuple[int, ...]] = set()
            for i in obs_indices:
                s, e = int(offsets[i]), int(offsets[i + 1])
                if e - s < 2:
                    continue
                roster = tuple(sorted(int(p) for p in player_flat[s:e]))
                if roster in seen_rosters:
                    continue
                seen_rosters.add(roster)
                idx = np.fromiter(roster, dtype=np.int64, count=len(roster))
                th = players.theta[idx]
                mean_th = float(th.mean())
                players.theta[idx] = th - cfg.eta_teammate * (th - mean_th)

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
        delta_size=delta_size if cfg.use_team_size_effect else None,
        team_size_anchor=team_size_anchor,
        delta_pos=delta_pos if cfg.use_pos_effect else None,
        pos_anchor=pos_anchor,
        recenter_events=list(recenter_log) if recenter_log else None,
    )
