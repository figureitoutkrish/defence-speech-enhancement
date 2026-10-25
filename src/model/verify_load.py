"""Phase 3a verification: loads dns48 and runs one real inference pass.

Loads the model via src/model/load.py, picks ONE real audio file from
data/processed/test/ via the test manifest (not a hardcoded path), runs it
through the model in inference mode (no gradient, no training), confirms
output shape matches input shape, and times the single inference call as a
sanity number (not a rigorous warm/cold benchmark -- that was already
verified separately per the development log).

Does NOT train, does NOT compute loss/optimizer state -- inference only.

Run standalone: `python -m src.model.verify_load` from repo root.
"""

import json
import logging
import time
from pathlib import Path

import soundfile as sf
import torch

from src.model.load import load_config, load_dns48

TEST_MANIFEST = Path("manifests/test.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_one_noisy_wav(manifest_path: Path = TEST_MANIFEST) -> tuple[torch.Tensor, int, dict]:
    """Picks the first entry from the test manifest and loads its noisy wav
    via soundfile (per project convention -- no torchaudio.load/save).

    Returns: (waveform [1, samples] float32 tensor, sample_rate, manifest_entry).
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    entry = manifest[0]

    data, sr = sf.read(entry["noisy_path"], dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()  # [channels, samples]
    return waveform, sr, entry


def main() -> None:
    cfg = load_config()

    model, device = load_dns48(cfg)

    waveform, sr, entry = load_one_noisy_wav()
    expected_sr = cfg["sample_rate"]
    if sr != expected_sr:
        raise RuntimeError(
            f"Loaded test audio sample rate {sr} != configs/finetune.yaml "
            f"sample_rate {expected_sr} -- refusing to run inference on mismatched rate."
        )

    logger.info(f"Loaded real test audio: {entry['noisy_path']} (pair_id={entry['pair_id']})")
    logger.info(f"Input shape: {tuple(waveform.shape)}, sample_rate: {sr}")

    input_tensor = waveform.unsqueeze(0).to(device)  # [batch=1, channels, samples]

    with torch.no_grad():
        start = time.perf_counter()
        output = model(input_tensor)
        elapsed = time.perf_counter() - start

    output = output.squeeze(0).cpu()  # back to [channels, samples]

    logger.info(f"Output shape: {tuple(output.shape)}")
    logger.info(f"Inference time (single run, sanity number only): {elapsed * 1000:.2f} ms")

    if tuple(output.shape) != tuple(waveform.shape):
        logger.error(
            f"OUTPUT SHAPE MISMATCH -- input {tuple(waveform.shape)} vs "
            f"output {tuple(output.shape)}. dns48 is expected to output "
            f"enhanced audio at the same length/sample-rate as input."
        )
        raise RuntimeError("Output shape does not match input shape -- see error above.")

    logger.info("Output shape matches input shape -- PASS.")
    logger.info("Phase 3a verification complete: model loads, architecture matches spec, "
                 "runs inference on real audio, output shape correct.")


if __name__ == "__main__":
    main()
