"""Loads the pretrained dns48 checkpoint (Facebook Research Denoiser).

Uses denoiser's own loading mechanism (`denoiser.pretrained.dns48()`) rather
than reimplementing checkpoint download/deserialization. Model architecture
and pretrained-loading code come from the pip package `denoiser==0.1.5`
as-is -- ONLY `src/vendor/denoiser_patched/{audio.py,stft_loss.py}` are
patched (see that README), and neither is imported anywhere in this module.
Verified directly: `denoiser.pretrained` imports only `.demucs` and `.utils`
(checked its import lines in the installed package), so loading the
checkpoint here does not touch either patched file -- confirmed, not assumed.

Per project rule 8: checks `torch.backends.mps.is_available()` and falls
back to CPU with an explicit, visible warning -- never fails silently.

Run standalone: `python -m src.model.load` from repo root (prints param
count / spec comparison / device, does not run inference).
"""

import logging

import torch
import yaml
from denoiser.pretrained import dns48

CONFIG_PATH = "configs/finetune.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Returns the MPS device if available, else CPU with a visible warning.

    Per project rule 8: never assume CUDA, never fail silently on missing
    MPS -- an M4 Pro without MPS available is a real, visible problem, not a
    silently-tolerated fallback.
    """
    if torch.backends.mps.is_available():
        logger.info("MPS backend available -- using device 'mps'.")
        return torch.device("mps")
    logger.warning(
        "MPS backend NOT available on this machine -- falling back to CPU. "
        "This target is an M4 Pro MacBook; CPU fallback will be significantly "
        "slower and should not happen in the expected environment. Check "
        "torch install / macOS version if this is unexpected."
    )
    return torch.device("cpu")


def _check_architecture(model: torch.nn.Module, cfg: dict) -> None:
    """Compares the loaded model's actual architecture against the locked
    spec in configs/finetune.yaml['model']['expected']. Flags ANY mismatch
    explicitly via logger.error -- never silently assumes it's fine.
    """
    expected = cfg["model"]["expected"]
    mismatches = []

    total_params = sum(p.numel() for p in model.parameters())
    if total_params != expected["total_params"]:
        mismatches.append(
            f"total_params: got {total_params}, expected {expected['total_params']}"
        )

    n_encoder = len(model.encoder)
    if n_encoder != expected["encoder_layers"]:
        mismatches.append(
            f"encoder_layers: got {n_encoder}, expected {expected['encoder_layers']}"
        )

    n_decoder = len(model.decoder)
    if n_decoder != expected["decoder_layers"]:
        mismatches.append(
            f"decoder_layers: got {n_decoder}, expected {expected['decoder_layers']}"
        )

    if model.kernel_size != expected["kernel_size"]:
        mismatches.append(
            f"kernel_size: got {model.kernel_size}, expected {expected['kernel_size']}"
        )

    if model.stride != expected["stride"]:
        mismatches.append(f"stride: got {model.stride}, expected {expected['stride']}")

    lstm = model.lstm.lstm
    if lstm.num_layers != expected["lstm_layers"]:
        mismatches.append(
            f"lstm_layers: got {lstm.num_layers}, expected {expected['lstm_layers']}"
        )
    if lstm.hidden_size != expected["lstm_hidden"]:
        mismatches.append(
            f"lstm_hidden: got {lstm.hidden_size}, expected {expected['lstm_hidden']}"
        )
    if lstm.input_size != expected["chout"]:
        mismatches.append(
            f"lstm_input (chout): got {lstm.input_size}, expected {expected['chout']}"
        )

    logger.info(f"Loaded dns48 -- total params: {total_params}")
    logger.info(f"Encoder layers: {n_encoder}, decoder layers: {n_decoder}")
    logger.info(f"kernel_size={model.kernel_size}, stride={model.stride}")
    logger.info(
        f"LSTM: num_layers={lstm.num_layers}, hidden_size={lstm.hidden_size}, "
        f"input_size={lstm.input_size}"
    )

    if mismatches:
        for m in mismatches:
            logger.error(f"ARCHITECTURE MISMATCH -- {m}")
        raise RuntimeError(
            f"dns48 architecture does not match the locked spec in "
            f"configs/finetune.yaml['model']['expected']: {mismatches}"
        )
    logger.info("Architecture matches locked spec exactly -- no mismatches.")


def load_dns48(cfg: dict | None = None) -> tuple[torch.nn.Module, torch.device]:
    """Loads the pretrained dns48 model, verifies its architecture against
    the locked spec, moves it to the best available device, and sets it to
    eval mode.

    Returns: (model, device).
    """
    if cfg is None:
        cfg = load_config()

    logger.info("Loading dns48 pretrained checkpoint via denoiser.pretrained.dns48()...")
    model = dns48(pretrained=True)

    _check_architecture(model, cfg)

    device = get_device()
    model = model.to(device)
    model.eval()

    return model, device


if __name__ == "__main__":
    load_dns48()
