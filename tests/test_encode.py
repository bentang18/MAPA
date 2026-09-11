"""Feature-cache contact identity survives canonical packing and serialization."""
import sys
from types import SimpleNamespace

import numpy as np
import torch

from evals.neuroprobe import encode
from evals.neuroprobe.readout import _elec_cols
from mapa.data.dataset import build_session_spec
from mapa.data.session_setup import build_session_setup


def test_saved_contact_labels_match_encoded_feature_order(tmp_path, monkeypatch):
    labels = ['RB4', 'LA3', 'RB1', 'LA1']
    setup = build_session_setup(labels, torch.zeros(4, dtype=torch.long), drop_labels=set())
    paths, stats = [], []
    for band, bins in enumerate((7, 6, 7)):
        path = tmp_path / f'band{band}.npy'
        values = np.arange(1, 5, dtype=np.float32)[:, None, None]
        np.save(path, np.broadcast_to(values, (4, bins, 64)))
        paths.append(str(path))
        stats.append((torch.zeros(4, bins, 1), torch.ones(4, bins, 1)))
    spec = build_session_spec(session_key=(1, 1), band_paths=paths, band_stats=stats,
                              setup=setup, n_frames=64, bad_spans_s=[])
    targets = SimpleNamespace(clip_starts=np.array([0., 1.]), labels={'onset': np.array([0, 1])},
                              ws_split={}, cs_split={})
    monkeypatch.setattr('mapa.data.session_loader.load_v3_sessions', lambda **kw: [spec])
    monkeypatch.setattr('mapa.data.region_fn.make_bt_region_fn',
                        lambda root: lambda s, t, labs: torch.zeros(len(labs), dtype=torch.long))
    monkeypatch.setattr('mapa.data.anatomy.lite_electrode_set', lambda subject: frozenset(labels))
    monkeypatch.setattr(encode, '_load_targets', lambda *a: targets)
    monkeypatch.setattr(sys, 'argv', ['encode', '--enc0-only', '--tag', 'fixture',
                        '--out-dir', str(tmp_path), '--span-dir', str(tmp_path),
                        '--bt-root', str(tmp_path), '--session-index', '0',
                        '--band-cache-dir', 'slow', '--band-cache-dir', 'mid',
                        '--band-cache-dir', 'fast'])
    encode.main()
    rec = torch.load(tmp_path / 'enc_s1_t1_fixture.pt', weights_only=False)
    assert rec['elec_labels'].tolist() == ['RB4', 'RB1', 'LA3', 'LA1']
    expected = torch.tensor([1., 3., 2., 4.], dtype=torch.float16)[None, :, None].expand(2, 4, 348)
    torch.testing.assert_close(rec['feats']['enc0_elec']['raw'], expected, rtol=0, atol=0)
    sibling = {'elec_labels': np.array(['RB4', 'LA1', 'XX1', 'LA3'])}
    assert _elec_cols(rec, sibling) == ([3, 2, 0], [1, 3, 0], 3)


def test_encode_refuses_existing_unverified_cache(tmp_path, monkeypatch):
    import pytest

    path = tmp_path / 'enc_s1_t1_fixture.pt'
    path.write_bytes(b'partial interrupted feature cache')
    monkeypatch.setattr(sys, 'argv', ['encode', '--enc0-only', '--tag', 'fixture',
                        '--out-dir', str(tmp_path), '--span-dir', 'missing',
                        '--bt-root', 'missing', '--session-index', '0',
                        '--band-cache-dir', 'slow', '--band-cache-dir', 'mid',
                        '--band-cache-dir', 'fast'])
    with pytest.raises(SystemExit, match='output already exists'):
        encode.main()
    assert path.read_bytes() == b'partial interrupted feature cache'


def test_checkpoint_loader_accepts_release_metadata_and_rejects_custom_objects(tmp_path):
    import datetime
    import pickle

    import pytest

    path = tmp_path / "checkpoint.pt"
    model = {"encoder.blocks.0.norm1.weight": torch.ones(4)}
    torch.save({"model": model, "meta": {"space_rope": False, "step": 55000}}, path)
    state, rope = encode._load_ckpt(path)
    assert rope is False
    torch.testing.assert_close(state["encoder.blocks.0.norm1.weight"], torch.ones(4))

    # Harmless unsupported metadata stands in for arbitrary pickle objects. The
    # checkpoint loader must reject it before extracting the model dictionary.
    torch.save({"model": model, "meta": {"space_rope": False,
                                       "custom": datetime.datetime(2026, 1, 1)}}, path)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        encode._load_ckpt(path)
