"""v5 TASK 2: dynamic (on-the-fly) mixing Dataset -- TRAINING SPLIT ONLY.

Root cause this responds to (see configs/finetune.yaml training_run_v5
comment, not repeated here): v1/v2/v4 all plateau because the training set
is a STATIC set of 2162 pre-generated pairs, identical every epoch. This
Dataset instead draws a fresh (clean utterance, noise clip, SNR,
augmentation) combination on every __getitem__ call, so every epoch sees new
combinations. Reuses src/data/mixing.py (mix_pair) and src/data/augment.py
(augment) exactly as-is -- does NOT reimplement the SNR-mixing or
reverb/clipping math, per explicit task instruction.

val/test are explicitly NEVER touched by this module -- they continue to use
the EXISTING static src/model/dataset.py's NoisyCleanDataset over
manifests/val.json / manifests/test.json, completely unchanged. This module
only ever constructs a training-split dataset; there is no val/test variant
here, by construction (no split parameter to misuse).

Run standalone: `python -m src.data.dynamic_dataset` from repo root (draws
and prints a few sample mixtures, does not train).
"""

import json
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as taF
import yaml
from torch.utils.data import Dataset

from src.data.augment import augment
from src.data.mixing import mix_pair

CONFIG_PATH = "configs/finetune.yaml"
DYNAMIC_POOL_MANIFEST = Path("manifests/dynamic_train_pool.json")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_wav(path: str, target_sr: int) -> torch.Tensor:
    """Strict loader for clean speech, matching build_dataset.py's
    _load_wav() convention exactly (LibriSpeech is always 16kHz natively --
    a mismatch means a real problem, not expected raw-source variation)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1).contiguous()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        raise RuntimeError(
            f"{path} has sample rate {sr}, expected {target_sr}. "
            "Dynamic pool sources (dev-clean, train-clean-100) should both "
            "be natively 16kHz -- a mismatch here is a real problem."
        )
    return wav


def _load_noise_wav(path: str, target_sr: int) -> torch.Tensor:
    """Loader for noise sources, matching build_dataset.py's
    _load_noise_wav() convention exactly (noise sources span several
    native sample rates -- resampling on-the-fly is expected preprocessing,
    not a sign of a bug)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1).contiguous()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = taF.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav


def load_noise_pool(cfg: dict) -> dict:
    """Identical noise-pool loading to build_dataset.py's load_noise_pool()
    -- same source directories, same MUSAN sound-bible/free-sound ->
    stationary/general split convention. Duplicated here (not imported from
    build_dataset.py) because build_dataset.py's version is coupled to its
    own module-level RAW_DIR constant and project rule 7 keeps src/
    modules from depending on each other in ways that would make this
    module's noise pool implicitly track build_dataset.py's unrelated
    dataset-orchestration changes; the actual glob logic is identical and
    trivial to keep in sync by inspection.
    """
    raw_dir = Path("data/raw")
    pool = {"gunshot": [], "stationary": [], "general": []}

    urbansound8k_dir = raw_dir / "urbansound8k" / cfg["dataset"]["urbansound8k"]["target_class"]
    pool["gunshot"].extend(sorted(str(p) for p in urbansound8k_dir.glob("*.wav")))

    gunshots_dir = raw_dir / "gunshots"
    pool["gunshot"].extend(sorted(str(p) for p in gunshots_dir.rglob("*.wav")))

    musan_noise_dir = raw_dir / "musan" / "noise"
    for wav in sorted(musan_noise_dir.rglob("*.wav")):
        if "sound-bible" in wav.parts:
            pool["stationary"].append(str(wav))
        else:
            pool["general"].append(str(wav))

    for category, files in pool.items():
        if not files:
            raise RuntimeError(
                f"Noise category '{category}' has 0 clips. "
                "Run src/data/download.py (and complete manual gunshots step) first."
            )
    return pool


class DynamicMixDataset(Dataset):
    """Draws a fresh (clean, noise, SNR, augmentation) mixture per
    __getitem__ call. TRAIN SPLIT ONLY -- there is no val/test mode.

    __len__ returns training_run_v5.steps_per_epoch * batch_size, so a
    DataLoader(shuffle=False) over this dataset naturally produces exactly
    steps_per_epoch batches per epoch (the fixed "epoch size" TASK 2/5
    requires for wall-clock comparability with v1/v2/v4) while each index
    still resolves to an independently-drawn random mixture (index value
    itself is unused beyond satisfying Dataset's __getitem__(idx) contract
    -- the mixture drawn does NOT depend on idx).

    Seeding: one `random.Random(seed)` instance, constructed once in
    __init__ from the top-level config `seed` -- NOT reseeded per epoch or
    per __getitem__. This makes a full re-run with the same seed
    reproduce the same overall draw sequence (the "seedable for
    reproducibility of a given run" requirement) while still drawing
    genuinely different mixtures epoch-to-epoch within one run, since the
    RNG state simply keeps advancing (the "new mixtures every epoch"
    requirement) -- reseeding per epoch would instead make every epoch
    IDENTICAL, defeating the entire point of dynamic mixing.
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.sample_rate = cfg["sample_rate"]
        self.segment_samples = int(cfg["training"]["segment_seconds"] * self.sample_rate)

        run_cfg = cfg["training_run_v5"]
        self._epoch_size = run_cfg["steps_per_epoch"] * run_cfg["batch_size"]

        with open(DYNAMIC_POOL_MANIFEST, "r") as f:
            self.pool_entries = json.load(f)

        self.noise_pool = load_noise_pool(cfg)

        category_weights_cfg = run_cfg["noise_category_weights"]
        self.categories = list(category_weights_cfg.keys())
        self.category_weights = [category_weights_cfg[c] for c in self.categories]

        # Single RNG instance, seeded once -- see class docstring.
        self._rng = random.Random(cfg["seed"])

    def __len__(self) -> int:
        return self._epoch_size

    def _draw_clean(self) -> torch.Tensor:
        entry = self._rng.choice(self.pool_entries)
        return _load_wav(entry["utterance_path"], self.sample_rate)

    def _draw_noise_category(self) -> str:
        return self._rng.choices(self.categories, weights=self.category_weights, k=1)[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_full = self._draw_clean()

        # Crop/pad clean to the fixed training segment length FIRST (same
        # convention as src/model/dataset.py's NoisyCleanDataset), then mix
        # -- mixing.py's fit_noise_to_length() then only ever needs to fit
        # noise to exactly segment_samples, matching how build_dataset.py's
        # static pairs were sized relative to mixing (mix at full utterance
        # length, static dataset then crops at train time) closely enough
        # for a fair comparison, while avoiding loading/mixing full-length
        # utterances just to discard most of them post-crop.
        length = clean_full.shape[-1]
        if length >= self.segment_samples:
            max_start = length - self.segment_samples
            start = self._rng.randint(0, max_start)
            clean = clean_full[..., start:start + self.segment_samples]
        else:
            pad = self.segment_samples - length
            clean = torch.nn.functional.pad(clean_full, (0, pad))

        noise_category = self._draw_noise_category()
        noise_path = self._rng.choice(self.noise_pool[noise_category])
        noise = _load_noise_wav(noise_path, self.sample_rate)

        result = mix_pair(clean, noise, noise_category, self.cfg, self._rng)
        noisy = augment(result["noisy"], self.cfg, self._rng, self.sample_rate)

        # [1, samples] -> [samples], matching NoisyCleanDataset's 1D
        # per-item convention (DataLoader's default collate then produces
        # [batch, samples]).
        return noisy.squeeze(0), result["clean"].squeeze(0)


if __name__ == "__main__":
    cfg = load_config()
    dataset = DynamicMixDataset(cfg)
    print(f"DynamicMixDataset: {len(dataset)} items/epoch "
          f"(steps_per_epoch={cfg['training_run_v5']['steps_per_epoch']} x "
          f"batch_size={cfg['training_run_v5']['batch_size']}), "
          f"pool size={len(dataset.pool_entries)} utterances, "
          f"categories={dataset.categories} weights={dataset.category_weights}")
    for i in range(3):
        noisy, clean = dataset[i]
        print(f"  sample {i}: noisy shape={tuple(noisy.shape)}, clean shape={tuple(clean.shape)}")
