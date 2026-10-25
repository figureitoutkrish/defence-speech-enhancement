"""Hand-built spectral subtraction baseline (Boll 1979 / Berouti et al. 1979).

Deliberately NOT using the `noisereduce` library -- this baseline exists to
demonstrate the classical failure mode (musical noise, residual artifacts)
that the fine-tuned model is pitched against, so it has to be the real
STFT-domain algorithm, not a black-box wrapper.

Algorithm, per pair:
  1. STFT the noisy signal -> complex spectrogram.
  2. Estimate the noise magnitude spectrum by averaging magnitude over the
     first `noise_estimate_frames` STFT frames of the SAME noisy signal.
     This is the standard no-VAD spectral-subtraction convention (Boll,
     1979) used when a separate noise-only recording isn't available.
     KNOWN LIMITATION for this dataset specifically: `mixing.py` (Phase 1)
     overlays the noise clip under the FULL utterance duration, so the
     leading frames are not guaranteed silence-plus-noise-only -- some
     pairs will have speech onset very early. This is documented here
     rather than hidden; it is exactly the kind of classical-baseline
     weakness the project's pitch argues a learned model avoids. It is not
     a bug to fix in this file -- fixing it would mean building a real VAD,
     which is out of scope for a classical baseline by definition.
  3. Subtract the noise magnitude estimate from the noisy magnitude,
     scaled by an oversubtraction factor alpha (Berouti et al., 1979) that
     over-subtracts to suppress residual noise at the cost of some speech
     distortion -- alpha and the spectral floor beta are both in
     configs/finetune.yaml, not hardcoded.
  4. Floor the result at `beta * noisy_magnitude` per frequency bin instead
     of clamping to zero -- a hard zero floor is what causes "musical
     noise" (isolated bins flickering to zero and back), so a small floor
     is the standard mitigation, though the artifact is still audible,
     which is the point of this baseline.
  5. Recombine the enhanced magnitude with the ORIGINAL NOISY PHASE (phase
     is not re-estimated -- this is the standard spectral-subtraction
     simplification; phase errors are known to be less perceptually
     important than magnitude errors at these SNRs) and inverse-STFT back
     to a waveform.

I/O goes through `soundfile` only (see the development log: torchaudio.load/save is
not used in this project due to the torchcodec/libavutil linking issue
found in Phase 0).

Run standalone: `python -m src.baseline.spectral_subtraction` from repo root.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

CONFIG_PATH = "configs/finetune.yaml"
TEST_MANIFEST = Path("manifests/test.json")
OUTPUT_DIR = Path("results/baseline")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def spectral_subtract(noisy: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Runs spectral subtraction on a single-channel waveform.

    Args:
        noisy: 1D float32 tensor, shape [samples].
        cfg: the full loaded finetune.yaml dict.

    Returns:
        1D float32 tensor, shape [samples] (same length as input).
    """
    stft_cfg = cfg["baseline"]["stft"]
    n_fft = stft_cfg["n_fft"]
    hop_length = stft_cfg["hop_length"]
    win_length = stft_cfg["win_length"]
    noise_estimate_frames = cfg["baseline"]["noise_estimate_frames"]
    alpha = cfg["baseline"]["oversubtraction_factor"]
    beta = cfg["baseline"]["spectral_floor"]

    window = torch.hann_window(win_length)

    spec = torch.stft(
        noisy,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )  # [freq_bins, frames]

    magnitude = spec.abs()
    phase = torch.angle(spec)

    num_frames = magnitude.shape[1]
    frames_for_estimate = min(noise_estimate_frames, num_frames)
    noise_mag_estimate = magnitude[:, :frames_for_estimate].mean(dim=1, keepdim=True)  # [freq_bins, 1]

    subtracted = magnitude - alpha * noise_mag_estimate
    floor = beta * magnitude
    enhanced_magnitude = torch.maximum(subtracted, floor)

    enhanced_spec = torch.polar(enhanced_magnitude, phase)

    enhanced = torch.istft(
        enhanced_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=noisy.shape[-1],
    )
    return enhanced


def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1).contiguous()  # [channels, samples]
    return wav[0], sr  # mono


def _save_wav(path: Path, wav: torch.Tensor, sr: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.numpy(), sr, subtype=subtype)


def run(manifest_path: Path = TEST_MANIFEST, output_dir: Path = OUTPUT_DIR) -> None:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    subtype = cfg["output"]["bit_depth"]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(manifest):
        noisy, sr = _load_wav(Path(entry["noisy_path"]))
        if sr != sample_rate:
            raise RuntimeError(
                f"{entry['noisy_path']} has sample rate {sr}, expected {sample_rate}"
            )

        enhanced = spectral_subtract(noisy, cfg)

        out_path = output_dir / f"{entry['pair_id']}_enhanced.wav"
        _save_wav(out_path, enhanced, sample_rate, subtype)

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"[spectral_subtraction] {i + 1}/{len(manifest)} done")

    print(f"[spectral_subtraction] wrote {len(manifest)} enhanced files to {output_dir}/")


if __name__ == "__main__":
    run()
