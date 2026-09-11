# Synthetic frontend reference

`frontend_reference.npz` contains synthetic reference outputs from the research STFT,
normalization, CAR and NeuralSet harmonic-notch functions. Source hashes are in
`reference_sources.json`; the reference environment used MNE 1.11.0 and PyTorch 2.10.0.

The fixture uses NumPy RNG seed 81, 7 channels × 20 seconds at 1024 Hz, scale 100.
Channel 4 is zero and is explicitly excluded by the caller; channel 6 is excluded from
the model montage. The test recreates the input and passes these selections explicitly.
Expected values include whole-recording magnitude bands, median/MAD statistics,
unclipped normalized values for the 4–5 second window, and surviving contact indices.

There are no participant recordings or artifact-detection rules in these fixtures.
Cross-version filtering/FFT comparisons use small floating-point tolerances; pure Torch
chunking is checked exactly against a whole-recording STFT.
