"""Neuroprobe union windows, per-task labels, and upstream-faithful split indices.

One encoded window serves all tasks that share its neural onset. Each task’s
interleaved dataset order determines its validation/test split on that union axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mapa.data.word_events import (
    _assign_cross_subject_split,
    _assign_within_session_split,
)

__all__ = [
    "SessionTargets",
    "build_session_targets",
]


@dataclass(frozen=True)
class SessionTargets:
    """One session's union clip axis + per-task labels + per-task splits.

    ``clip_starts``/``clip_durations``/``clip_movie_onsets`` are ``(N_union,)`` — the neural
    windows Stage 1 forwards (``start`` is the neural-clock onset ``est_idx/SR``). ``labels``,
    ``ws_split``, ``cs_split`` key by task; split arrays index the union axis."""

    subject_id: int
    trial_id: int
    clip_starts: np.ndarray
    clip_durations: np.ndarray
    clip_movie_onsets: np.ndarray
    labels: dict[str, np.ndarray]
    ws_split: dict[str, dict[int, dict[str, np.ndarray]]]
    cs_split: dict[str, dict[str, np.ndarray]]


def _union_axis(events: pd.DataFrame) -> tuple[np.ndarray, dict[float, int]]:
    """Unique neural windows (by ``start``), sorted — the forward grid's clip axis."""
    starts = np.unique(events["start"].astype(float).to_numpy())
    return starts, {float(s): i for i, s in enumerate(starts)}


def _rows_to_union(
    split_df: pd.DataFrame, task: str, split_name: str, index: dict[float, int]
) -> np.ndarray:
    sel = split_df[(split_df["task"] == task) & (split_df["split"] == split_name)]
    return np.array(
        [index[float(s)] for s in sel["start"].astype(float)], dtype=np.int64
    )


def build_session_targets(
    events: pd.DataFrame,
    *,
    subject_id: int,
    trial_id: int,
    n_folds: int = 2,
) -> SessionTargets:
    """Assemble one session's targets from its balanced word-event rows.

    ``events`` is ``_word_event_rows(..., balance=True)`` for ONE session, all 15 tasks
    (columns ``start, duration, task, label, subject_id, trial_id, movie_onset_s``)."""
    starts, index = _union_axis(events)
    n = len(starts)

    # Per-clip duration / movie-onset (a clip's window is task-independent → first wins).
    dur = np.full(n, np.nan)
    mov = np.full(n, np.nan)
    for s, d, m in zip(
        events["start"].astype(float),
        events["duration"].astype(float),
        events["movie_onset_s"].astype(float),
    ):
        i = index[float(s)]
        if np.isnan(dur[i]):
            dur[i], mov[i] = d, m

    tasks = list(events["task"].unique())

    labels: dict[str, np.ndarray] = {}
    for task in tasks:
        arr = np.full(n, np.nan)
        rows = events[events["task"] == task]
        for s, lab in zip(rows["start"].astype(float), rows["label"]):
            arr[index[float(s)]] = float(lab)
        labels[task] = arr

    # WS: KFold2 held-out fold halved (one split_df per fold; per-task → union indices).
    ws_split: dict[str, dict[int, dict[str, np.ndarray]]] = {t: {} for t in tasks}
    for fold in range(n_folds):
        sdf = _assign_within_session_split(
            events, test_subject_id=subject_id, test_trial_id=trial_id,
            fold_index=fold, n_folds=n_folds,
        )
        for task in tasks:
            ws_split[task][fold] = {
                name: _rows_to_union(sdf, task, name, index)
                for name in ("train", "val", "test")
            }

    # CS: the test session's per-task 50/50 chronological halves (train = anchor, elsewhere).
    csdf = _assign_cross_subject_split(
        events, test_subject_id=subject_id, test_trial_id=trial_id,
        train_subject_id=-1, train_trial_id=-1,  # sentinel anchor → only val/test assigned
    )
    cs_split: dict[str, dict[str, np.ndarray]] = {}
    for task in tasks:
        cs_split[task] = {
            name: _rows_to_union(csdf, task, name, index) for name in ("val", "test")
        }

    return SessionTargets(
        subject_id=subject_id, trial_id=trial_id,
        clip_starts=starts, clip_durations=dur, clip_movie_onsets=mov,
        labels=labels, ws_split=ws_split, cs_split=cs_split,
    )


PROBE_CLIP_DUR_S = 1.0   # the leaderboard-faithful 1 s word window


def build_label_events(subject_id: int, trial_id: int, timeline: str, tasks, bt_root,
                  *, lite_cap: bool = True) -> pd.DataFrame:
    """Balanced word-event rows for one session, using the seeded Lite sample cap.

    The cap controls trial selection only; contact selection happens separately.
    """
    from mapa.data.word_events import (
        _load_neural_to_movie_map,
        _load_pitch_volume_features,
        _load_words_and_nonverbal,
        _tasks_need_pitch_volume,
        _word_event_rows,
    )

    words_df, nonverbal_df = _load_words_and_nonverbal(
        subject_id, trial_id, bt_root=bt_root, enrich=True
    )
    neural_to_movie = _load_neural_to_movie_map(subject_id, trial_id, bt_root)
    pvf = (
        _load_pitch_volume_features(subject_id, trial_id)
        if _tasks_need_pitch_volume(tasks) else None
    )
    return _word_event_rows(
        subject_id=subject_id, trial_id=trial_id, timeline=timeline,
        words_df=words_df, nonverbal_df=nonverbal_df, tasks=tuple(tasks),
        binary_tasks=True, lite=lite_cap, nano=False, random_seed=42,
        duration=PROBE_CLIP_DUR_S, balance=True,
        pitch_volume_features=pvf, neural_to_movie=neural_to_movie,
    )
