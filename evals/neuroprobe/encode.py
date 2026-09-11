"""Encode the 12 Neuroprobe-Lite sessions into frozen per-region and per-contact features.

The three normalized 32 Hz magnitude bands receive Guard 3 at the model
input. Tap 0 uses the same clipped, decimated inputs as the encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import torch

from mapa.models.stem import clip_band

# The 15 Neuroprobe leaderboard tasks (upstream ``neuroprobe.config.NEUROPROBE_TASKS``).
# --tasks board re-labels the SAME task-agnostic features, so the only extra encode cost is
# materializing 15 label vectors instead of 4.
BOARD_TASKS: tuple[str, ...] = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
# The 12 Neuroprobe-Lite sessions (upstream ``NEUROPROBE_LITE_SUBJECT_TRIALS``): the board's
# CS anchor (2,4) + the 10 CS test cells + (2,0). --sessions board evaluates these.
BOARD_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
    (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1),
)
FPS = 32.0
CLIP_DUR_S = 1.0
N_REGIONS = 75
GPU_TAPS: tuple[int, ...] = (3, 6, 9, 12)   # raw block outputs read in one encoder forward


def _load_ckpt(ckpt_path: str) -> tuple[dict, bool | None]:
    """The encoder weights and, when the file records it, whether relative position is on.

    Reads both layouts through the same helper the public loader uses, so there is one
    implementation of what counts as a checkpoint.
    """
    from mapa.encoder import _tower_state

    return _tower_state(torch.load(ckpt_path, map_location="cpu", weights_only=True))


def _freeze(m, *, device):
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _load_encoder(tsd: dict, *, device: torch.device, space_rope: bool):
    """Build the encoder shell and load the weights into it.

    Everything the shell needs is read off the weights rather than passed in, so a wrong guess
    fails the strict check below instead of being absorbed. Deep supervision shows up as
    ``norms_block.*`` with no ``norm_out``, and a single tap as the reverse. Width is read off a
    LayerNorm, and a wrong width cannot load at all, whatever ``strict`` says.

    Relative position is the one exception, handled by the caller: see ``--no-relative-position``.
    """
    from mapa.models.tower import MaeEncoder

    peek = [v.shape[0] for kk, v in tsd.items() if kk.endswith("region_embed.embed.weight")]
    if peek and int(peek[0]) != N_REGIONS:
        raise ValueError(f"ckpt region table {peek[0]} != expected {N_REGIONS}")
    deep_sup = any(kk.startswith("encoder.norms_block.") for kk in tsd)
    # region_embed is READ OFF the ckpt for the same reason deep_sup is: the --no-region-embed
    # arm ships a tower with no region_embed submodule at all, so a shell built with one puts
    # region_embed.embed.weight in `missing` and the check below raises. Inferring it keeps that
    # check as the verifier — a wrong inference still fails loud, it is never silently absorbed.
    region_embed = bool(peek)
    dkey = "encoder.blocks.0.norm1.weight"
    if dkey not in tsd:
        raise RuntimeError(f"ckpt has no '{dkey}'; cannot infer encoder width")
    d_model = int(tsd[dkey].shape[0])
    print(f"[encode] encoder deep_sup={deep_sup} region_embed={region_embed} d_model={d_model} "
          f"(read off the weights) relative_position={space_rope}")
    tower = MaeEncoder(n_regions=N_REGIONS, deep_sup=deep_sup, region_embed=region_embed,
                       space_rope=space_rope, d_model=d_model)
    missing, unexpected = tower.load_state_dict(tsd, strict=False)
    bad = [m for m in missing if "num_batches_tracked" not in m]
    if bad or unexpected:
        raise RuntimeError(f"encoder state_dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    return _freeze(tower, device=device)


def _load_targets(session, bt_root, tasks=BOARD_TASKS):
    from mapa.probe.labels import build_label_events as _label_events
    from mapa.probe.labels import build_session_targets

    subject_id, trial_id = session
    events = _label_events(subject_id, trial_id, f"btbank{subject_id}_trial{trial_id}",
                           tasks, bt_root, lite_cap=True)
    # build_session_targets derives its task list from events["task"].unique(), so passing
    # `tasks` to _label_events is what widens 4 -> 15.
    return build_session_targets(events, subject_id=subject_id, trial_id=trial_id)


def _lite_keep_labels_fn(bt_root):
    """``keep_labels_fn`` restricting a session to its Neuroprobe-Lite montage.

    Injected into ``load_v3_sessions`` (NOT applied to the built spec) so keep_idx / region_id
    / sidecar / geom / band_stats are all constructed on the Lite axis by the normal code path
    — ``geom`` cannot be masked post-hoc, its gather_idx stores indices into the survivor axis.
    The realized montage is a SET intersection (``voltage_order ∩ lite_labels``), matching
    upstream ``datasets.py`` ``[full.index(e) for e in lite if e in full]``; we keep OUR voltage
    order, which is free for the encoder (the per-region pool is permutation-invariant within a
    region) and is what index-RoPE expects."""
    from mapa.data.anatomy import lite_electrode_set

    def fn(subject_id, trial_id, labels):
        return set(lite_electrode_set(subject_id))

    return fn


def _window_bands(spec, starts, clip_frames):
    """Slice normalized evaluation windows from three continuous 32 Hz band caches.

    Load selected contacts once per band to avoid repeated scattered disk reads."""
    keep = spec.keep_idx.numpy()
    starts = np.asarray(starts, dtype=float)
    n_frames_native = spec.n_frames
    t0 = np.rint(starts * FPS).astype(np.int64)
    end = t0 + clip_frames
    oob = np.where((t0 < 0) | (end > n_frames_native))[0]
    if len(oob):
        raise RuntimeError(
            f"{spec.session_key}: {len(oob)} union windows out of cache bounds "
            f"(n_frames_native={n_frames_native}, first bad start={float(starts[oob[0]]):.4f}s)"
        )
    bands = []
    for path, norm in zip(spec.band_paths, spec.band_norms):
        mm = np.load(path, mmap_mode="r")
        full = np.asarray(mm[keep], dtype=np.float32)              # (N, F, T_total) bulk → RAM
        del mm
        clips = np.stack([full[:, :, a:b] for a, b in zip(t0.tolist(), end.tolist())], axis=0)
        t = torch.from_numpy(clips)
        bands.append(norm.transform(t))       # (n, N, F_b, T32) vectorized
    return bands                                                  # 3 × (n, N, F_b, T32)


def _canon_regions(grid, region_id):
    """Canonical (grid-order) contact indices + their region ids + present region atlas ids.

    build_r4_grid lays tokens contact-major (k_full block per contact); the first token of
    each block carries that contact's index (``grid.contact``), so reshaping to (n, k_full)
    and taking column 0 recovers the n canonical contacts and their regions."""
    k = grid.k_full
    canon = grid.contact.reshape(-1, k)[:, 0].cpu().numpy()         # (n,) contact index into N
    region_canon = region_id.cpu().numpy()[canon]                   # (n,) DKT tag per canon contact
    present = np.unique(region_canon)                              # sorted present atlas ids
    return canon, region_canon, present


def _pool_regions(x, region_canon, present):
    """Pool electrodes→regions: x (B, n, *feat) → (B, |P|, prod(feat)) flattened last dim.

    Per present region, MEAN over its electrodes. Returns (B, |P|, F) fp16, region order == present."""
    B = x.shape[0]
    blocks = []
    for p in present:
        cols = np.where(region_canon == p)[0]
        sub = x[:, cols]                                          # (B, |cols|, *feat)
        blocks.append(sub.mean(1).reshape(B, -1))                 # (B, prod(feat))
    return torch.stack(blocks, dim=1).to(torch.float16)           # (B, |P|, F)


def _enc0_band_lengths(bands, strides):
    """Frame counts after the model’s per-band decimation."""
    st = _enc0_strides(bands, strides)
    return tuple(int(-(-b.shape[-1] // s)) for b, s in zip(bands, st))  # len of x[..., ::s]


def _enc0_strides(bands, strides):
    """Resolve/validate the per-band decimation strides shared by enc0's pooled + elec paths."""
    from mapa.models.pack_r4 import BAND_STRIDES

    if strides is None:
        strides = BAND_STRIDES
    if len(strides) != len(bands):
        raise ValueError(f"enc0 got {len(bands)} bands but {len(strides)} strides")
    return strides


def _enc0_bands_canon(bands, canon, strides):
    """Per band: decimate ``x[..., ::stride]`` (the model's own input frames), reorder to
    canonical contacts, time-major. Yields (n_win, n_canon, T_b, F_b) tensors, one per band —
    the shared prefix of enc0's pooled and unpooled (elec) paths."""
    for b, (x, st) in enumerate(zip(bands, strides)):             # x (n, N, F_b, T_clock)
        xd = clip_band(x[..., ::st], b)                           # (n, N, F_b, T_b)
        xd = xd.transpose(-1, -2).contiguous()                   # (n, N, T_b, F_b) time-major
        yield xd[:, canon]                                       # (n, n_canon, T_b, F_b)


def _enc0_pooled(bands, canon, region_canon, present, strides=None):
    """Model-clipped input baseline, decimated and pooled by atlas region."""
    strides = _enc0_strides(bands, strides)
    per_band = [_pool_regions(xd, region_canon, present)
                for xd in _enc0_bands_canon(bands, canon, strides)]
    return torch.cat(per_band, dim=-1)                            # (n, |P|, F0)


def _enc0_elec(bands, canon, strides=None):
    """Unpooled enc0: same canonical-contact axis and band-concat order as ``_enc0_pooled``'s
    pre-pool input — the depth-0 sibling of ``_encode_taps``' ``enc{t}_elec`` (GPU taps
    3/6/12), which stores the equivalent unpooled tensor for those taps."""
    strides = _enc0_strides(bands, strides)
    per_band = []
    for xd in _enc0_bands_canon(bands, canon, strides):
        B = xd.shape[0]
        per_band.append(xd.reshape(B, xd.shape[1], -1).to(torch.float16))
    return torch.cat(per_band, dim=-1)                            # (n, n_canon, F0)


@torch.no_grad()
def _encode_taps(encoder, bands, grid, region_packed, region_canon, present,
                 *, device, batch_size, elec_taps=()):
    """One forward of the encoder over all windows → per-tap region-pooled keep-time features.

    Cache stores the raw region-mean feature (n,|P|,k_full·d) — the most flexible storage: a
    readout can standardize columns on train stats (the FM linear-probe convention) or feed it
    raw (r1/M9-comparable), but neither is recoverable from a baked per-token LN. So we keep raw
    only. Returns {tap: {'raw': ...}}."""
    n = bands[0].shape[0]
    k = grid.k_full
    # Per-electrode keep-time: WS keeps ALL electrodes; the region-mean is the
    # comparison. Stored UNPOOLED on the canonical-contact axis (same order as region_canon), so
    # it is the pooled tap's exact pre-mean input — the diff is the pooling and nothing else.
    #
    # Each tap is preallocated at full length and written slice-by-slice. Accumulating a list of
    # per-batch tensors and torch.cat-ing at the end held the list AND its concatenation alive
    # simultaneously — a 2x peak on the ~40 GB enc12_elec tap, which is what forced --mem=300G
    # (measured 230-245 GiB on the 12-session board encode). Values, dtypes and row order are
    # unchanged; only the allocation pattern differs.
    acc: dict = {key: None for key in list(GPU_TAPS) + [f"elec{t}" for t in elec_taps]}

    def _write(key, lo, hi, x):
        if acc[key] is None:
            acc[key] = torch.empty((n, *x.shape[1:]), dtype=x.dtype)
        acc[key][lo:hi] = x

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bb = [b[s:e].to(device) for b in bands]
        Bb = e - s
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            _z, taps = encoder.forward(bb, grid, region_packed, tap_blocks=GPU_TAPS)
        for t in GPU_TAPS:
            enc = taps[t].float().reshape(Bb, -1, k, taps[t].shape[-1]).cpu()  # (Bb, n, k, d)
            _write(t, s, e, _pool_regions(enc, region_canon, present))
            if t in elec_taps:
                # (Bb, n_contacts, k·d) fp16 — the SAME tensor, just unpooled.
                _write(f"elec{t}", s, e, enc.reshape(Bb, enc.shape[1], -1).to(torch.float16))
    return {t: {"raw": v} for t, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="required unless --enc0-only (enc0 never reads weights)")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True,
                   help="3× in v3 concat order: slow, mid, hga")
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sessions", choices=("board",), default="board",
                   help="the 12 Neuroprobe-Lite sessions")
    p.add_argument("--session-index", type=int, default=None,
                   help="encode ONLY cohort[i], one Slurm array task per session. "
                        "Default = all.")
    p.add_argument("--tasks", choices=("board15",), default="board15",
                   help="board15 = the 15 leaderboard tasks (re-labels the same features)")
    p.add_argument("--electrode-set", choices=("lite",), default="lite",
                   help="lite = the Neuroprobe-Lite montage (leaderboard parity)")
    p.add_argument("--no-relative-position", "--no-space-rope", dest="no_space_rope",
                   action="store_true",
                   help="the checkpoint was trained without the relative positional encoding. "
                        "A released checkpoint records this itself and needs no flag. A raw "
                        "training checkpoint records nothing, and the setting leaves no trace in "
                        "the weights, so it has to be passed by hand: get it wrong and the encode "
                        "rotates by a contact index the trained model never saw, with no error.")
    p.add_argument("--enc0-only", action="store_true",
                   help="compute ONLY the enc0 input floor (raw decimated band bins pooled to "
                        "regions). No ckpt, no model, no GPU — enc0 never touched weights. Lets "
                        "the published frontend baseline run without loading a checkpoint.")
    p.add_argument("--elec-taps", default="0,12",
                   help="comma-separated taps to ALSO write per-electrode (unpooled), e.g. "
                        "'0,12' -> feats['enc0_elec'], feats['enc12_elec']. 0 = the |STFT| "
                        "frontend (enc0); 3/6/9/12 route through the encoder forward (GPU_TAPS). "
                        "WS keeps all electrodes by default; each costs "
                        "~N/|P| (~5x) the pooled tap on disk.")
    args = p.parse_args()

    from mapa.data.region_fn import make_bt_region_fn
    from mapa.data.session_loader import load_v3_sessions
    from mapa.models.pack_r4 import build_r4_grid

    n_cache = 3
    if len(args.band_cache_dirs) != n_cache:
        raise SystemExit(
            f"need {n_cache} --band-cache-dir (slow, mid, hga), got {len(args.band_cache_dirs)}")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_frames = round(CLIP_DUR_S * FPS)
    region_fn = make_bt_region_fn(args.bt_root)
    cohort = BOARD_SESSIONS
    if args.session_index is not None:
        if not 0 <= args.session_index < len(cohort):
            raise SystemExit(f"--session-index {args.session_index} out of range [0,{len(cohort)})")
        cohort = (cohort[args.session_index],)
    for s, t in cohort:
        path = os.path.join(args.out_dir, f"enc_s{s}_t{t}_{args.tag}.pt")
        if os.path.exists(path):
            raise SystemExit(
                f"output already exists: {path}. Use a new tag/directory, or --session-index "
                "to encode only a missing session; existing files are not verified or resumed."
            )
    tasks = BOARD_TASKS
    keep_labels_fn = _lite_keep_labels_fn(args.bt_root)
    elec_taps = tuple(int(t) for t in args.elec_taps.split(",") if t.strip())
    bad = [t for t in elec_taps if t not in (0,) + GPU_TAPS]
    if bad:
        raise SystemExit(f"--elec-taps {bad} not in {(0,) + GPU_TAPS} (0 = enc0_elec, off the "
                          f"|STFT| frontend directly; the rest route through GPU_TAPS {GPU_TAPS})")
    # tap 0 never goes through the encoder forward (_encode_taps/GPU_TAPS) — it's the frontend
    # input, handled by _enc0_elec below instead of _encode_taps' elec_taps loop.
    gpu_elec_taps = tuple(t for t in elec_taps if t != 0)
    want_enc0_elec = 0 in elec_taps
    checkpoint = None
    if args.enc0_only:
        # enc0 is a pure function of the CACHED BANDS (no weights ever touched), so there is
        # nothing to load and nothing to run on a GPU.
        encoder = None
        tower_note = "ENC0-ONLY (no ckpt, no model, CPU)"
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --enc0-only")
        sd, recorded = _load_ckpt(args.ckpt)
        if args.no_space_rope:
            space_rope = False
        elif recorded is not None:
            space_rope = recorded
        else:
            raise SystemExit(
                f"{args.ckpt} does not record whether the relative positional encoding is "
                "active, and it cannot be read off the weights. Pass --no-relative-position "
                "for an ablation checkpoint, or re-export the checkpoint with its metadata.")
        encoder = _load_encoder(sd, device=device, space_rope=space_rope)
        digest = hashlib.sha256()
        with open(args.ckpt, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        checkpoint = {
            "sha256": digest.hexdigest(), "space_rope": space_rope,
            "region_embed": any(k.endswith("region_embed.embed.weight") for k in sd),
            "d_model": encoder.d_model,
        }
        del sd
        tower_note = "encoder"
    print(f"[encode-r4] tag={args.tag} device={device} gpu_taps={GPU_TAPS} + enc0 "
          f"{tower_note}, sessions={args.sessions}({len(cohort)}) "
          f"tasks={args.tasks}({len(tasks)}) electrodes={args.electrode_set}", flush=True)

    for session in cohort:
        subject_id, trial_id = session
        path = os.path.join(args.out_dir, f"enc_s{subject_id}_t{trial_id}_{args.tag}.pt")
        spec = load_v3_sessions(
            sessions=[session], band_cache_dirs=args.band_cache_dirs, span_dir=args.span_dir,
            region_fn=region_fn, lof_report_path=None,
            keep_labels_fn=keep_labels_fn,
        )[0]
        if keep_labels_fn is not None:
            # The montage is the WHOLE parity claim — print what was realized, per session.
            from mapa.data.anatomy import lite_electrode_set
            lite = lite_electrode_set(subject_id)
            kept = spec.setup.sidecar.labels
            ok = set(kept) <= set(lite)
            print(f"[check] lite montage s{subject_id}_t{trial_id}: kept={len(kept)} "
                  f"of lite-list {len(lite)} subset-of-lite={ok} -> {'OK' if ok else 'VIOLATED'}",
                  flush=True)
            if not ok:
                raise RuntimeError("lite montage kept a non-Lite electrode — refusing to write")
        targets = _load_targets(session, args.bt_root, tasks)
        bands = _window_bands(spec, targets.clip_starts, clip_frames)

        geom = spec.setup.geom.to(device)
        region_id = spec.setup.region_id.to(device)
        grid = build_r4_grid(geom, n_time=clip_frames)
        region_true = region_fn(subject_id, trial_id, list(spec.setup.sidecar.labels)).long()
        if not torch.equal(region_true, spec.setup.region_id.cpu()):
            raise RuntimeError(
                f"region tag parity FAILED for s{subject_id}_t{trial_id}: the recomputed atlas "
                f"tag disagrees with the session spec on "
                f"{int((region_true != spec.setup.region_id.cpu()).sum())} of "
                f"{region_true.numel()} contacts. Refusing to encode."
            )
        region_packed = region_id[grid.contact]                        # MODEL-side tag
        canon, region_canon, present = _canon_regions(grid, region_true)  # TRUE atlas ids

        feats = {"enc0": {"raw": _enc0_pooled(bands, canon, region_canon, present)}}
        if want_enc0_elec:
            feats["enc0_elec"] = {"raw": _enc0_elec(bands, canon)}
        enc0_lengths = _enc0_band_lengths(bands, None)
        if enc0_lengths != tuple(int(x) for x in grid.band_lengths):
            raise RuntimeError(f"enc0 band lengths {enc0_lengths} != model grid {grid.band_lengths}")
        if args.enc0_only:
            tap_pooled = {}
        else:
            tap_pooled = _encode_taps(encoder, bands, grid, region_packed, region_canon,
                                      present, device=device, batch_size=args.batch_size,
                                      elec_taps=gpu_elec_taps)
            for t in GPU_TAPS:
                feats[f"enc{t}"] = tap_pooled[t]
            for t in gpu_elec_taps:
                feats[f"enc{t}_elec"] = tap_pooled[f"elec{t}"]

        payload = {
            "subject_id": subject_id, "trial_id": trial_id, "ckpt_tag": args.tag,
            "checkpoint": checkpoint,
            "present_parcels": np.asarray(present, dtype=np.int64),   # (|P|,) atlas ids, feature order
            # (n_contacts,) atlas id per CANONICAL contact — the same axis the ``enc*_elec`` taps
            # are stored on, so a readout can re-pool electrodes→regions itself instead of being
            # stuck with the mean baked in here. present_regions alone is not enough: it gives the
            # region ORDER but not the membership.
            "parcel_canon": np.asarray(region_canon, dtype=np.int64),
            "elec_labels": np.asarray(spec.setup.sidecar.labels)[canon],
            "band_lengths": enc0_lengths,
            # Frequency bins per band. band_lengths alone does NOT let a consumer slice enc0 by
            # band: the encoder taps are (k_full, d) so band_lengths is enough there, but enc0 is
            # the raw spectrogram at Σ_b F_b·T_b, and F_b is not recoverable from that total.
            "band_fdims": tuple(int(b.shape[-2]) for b in bands),
            "feats": {k: {v: t for v, t in d.items()} for k, d in feats.items()},
            "clip_starts": np.asarray(targets.clip_starts),
            "labels": {lt: np.asarray(v) for lt, v in targets.labels.items()},
            "ws_split": targets.ws_split,
            "cs_split": targets.cs_split,
            "n_windows": int(bands[0].shape[0]),
        }
        with open(path, "xb") as f:
            torch.save(payload, f)
        shp = {k: tuple(next(iter(d.values())).shape) for k, d in feats.items()}
        print(f"[encode-r4] {session}: |P|={len(present)} n={payload['n_windows']} "
              f"shapes={shp} -> {path}", flush=True)
        del bands, feats, tap_pooled, payload


if __name__ == "__main__":
    main()
