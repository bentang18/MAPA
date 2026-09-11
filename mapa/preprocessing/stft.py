"""Whole-recording magnitude STFT, ported from the research cache producer."""

import torch


def _single_stft_raw_view(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    nperseg: int,
    hop_length: int,
    k0: int,
    k1: int,
    log_eps: float,
    apply_log: bool = False,
    cartesian: bool = False,
) -> torch.Tensor:
    """2STFT single-band raw |STFT| (§3a): ONE Hann STFT, keep the inclusive
    rfft-bin slice ``[k0, k1]``. ``cartesian=True`` (3STFT slow band, §4) instead
    returns the two real components stacked on the freq axis ([Re ++ Im], F →
    2·n_bins) to preserve phase; magnitude/log are bypassed.

    Same STFT call as :func:`_multi_stft_raw_view` (``center=True``,
    ``normalized=False``, value axis = raw ``|STFT|``) — only the bin selection
    differs. Input ``(..., C, T_samples)`` → output ``(..., C, k1-k0+1, T_bin)``
    with ``T_bin = 1 + T_samples // hop_length``.
    """
    win = torch.hann_window(nperseg, device=waveform.device)
    wf = waveform
    if wf.shape[-1] < nperseg:
        # NeuralSet's prepare() probes with sub-second inputs — pad so
        # torch.stft(center=True) can reflect-pad without crashing. Real
        # 1 s windows never hit this branch.
        wf = torch.nn.functional.pad(wf, (0, nperseg - wf.shape[-1]))
    spec = torch.stft(
        wf,
        n_fft=nperseg,
        hop_length=hop_length,
        win_length=nperseg,
        window=win,
        return_complex=True,
        normalized=False,
        center=True,
    )
    band = spec[..., k0 : k1 + 1, :]  # (..., n_bins, T) complex
    if cartesian:
        # Phase-bearing slow band (§4): two real channels stacked on the freq
        # axis as [Re of n_bins ++ Im of n_bins] → F doubles to 2·n_bins. Keeps
        # phase; apply_log is rejected for cartesian upstream so it never applies.
        return torch.cat([band.real, band.imag], dim=-2)  # (..., 2·n_bins, T)
    mag = band.abs()  # (..., n_bins, T)
    if apply_log:
        return torch.log(mag + log_eps)
    return mag


def _single_stft_raw_view_chunked(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    nperseg: int,
    hop_length: int,
    k0: int,
    k1: int,
    log_eps: float,
    apply_log: bool = False,
    cartesian: bool = False,
    chunk_frames: int = 960,
) -> torch.Tensor:
    """Memory-bounded whole-recording single-band raw |STFT|, byte-identical to
    one un-chunked :func:`_single_stft_raw_view` over the full ``waveform``.

    The single-band analog of :func:`_multi_stft_raw_view_chunked`: overlap-save
    in ``chunk_frames``-wide output blocks, each reading ``half = nperseg // 2``
    extra REAL samples of context per side, then keeping only the frames whose
    full analysis window was real (the ``center=True`` reflect frames survive
    only at the TRUE recording ends, exactly as the un-chunked call produces).

    Correctness needs ``half`` to be a multiple of ``hop_length`` so every
    segment starts on the global hop grid — true for both 2STFT bands
    (low 256 = 1·256; high 64 = 1·64). Pinned by the acceptance test.

    Input ``(..., C, T_samples)`` → ``(..., C, k1-k0+1, T_bin)`` with
    ``T_bin = 1 + T_samples // hop_length``.
    """
    if hop_length <= 0 or nperseg % hop_length != 0:
        raise ValueError(
            f"chunked single-band STFT needs nperseg ({nperseg}) divisible by "
            f"hop_length ({hop_length}) so segment starts land on the hop grid"
        )
    half = nperseg // 2
    if half % hop_length != 0:
        raise ValueError(
            f"chunked single-band STFT needs half=nperseg//2 ({half}) divisible "
            f"by hop_length ({hop_length}) for segment→global frame alignment"
        )
    n_samples = int(waveform.shape[-1])
    total_frames = 1 + n_samples // hop_length

    def _view(seg: torch.Tensor) -> torch.Tensor:
        return _single_stft_raw_view(
            seg,
            sample_rate=sample_rate,
            nperseg=nperseg,
            hop_length=hop_length,
            k0=k0,
            k1=k1,
            log_eps=log_eps,
            apply_log=apply_log,
            cartesian=cartesian,
        )

    # Whole recording fits in one pass (or is too short to chunk): direct call.
    if total_frames <= chunk_frames or n_samples <= 2 * half + nperseg:
        return _view(waveform)

    blocks: list[torch.Tensor] = []
    f = 0
    while f < total_frames:
        f_end = min(f + chunk_frames, total_frames)
        lo = max(0, (f * hop_length) - half)
        hi = min(n_samples, (f_end - 1) * hop_length + half + 1)
        if hi - lo < nperseg:
            lo = ((max(0, hi - nperseg)) // hop_length) * hop_length
        seg = waveform[..., lo:hi]
        spec_seg = _view(seg)
        g0 = f - lo // hop_length
        g1 = f_end - lo // hop_length
        blocks.append(spec_seg[..., g0:g1])
        f = f_end
    return torch.cat(blocks, dim=-1)
