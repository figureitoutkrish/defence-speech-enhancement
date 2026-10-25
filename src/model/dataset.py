"""PyTorch Dataset/DataLoader for noisy/clean pairs from Phase 1 manifests.

Reads `manifests/{train,val,test}.json` (written by `src/data/build_dataset.py`
in Phase 1). Loads audio via `soundfile` only -- matching the Phase 0
soundfile-only-for-I/O convention (`torchaudio.load`/`save` are not used
anywhere in this project, per the development log).

Manifest clips have variable duration (verified in the Phase 3a follow-up:
3.8s-32.3s across test-split files alone). To collate into a batch tensor,
each item is cropped (or padded, if shorter) to a fixed
`training.segment_seconds` window from `configs/finetune.yaml` -- the SAME
crop offset is applied to both the noisy and clean waveform of a pair, so
they stay time-aligned.

Run standalone: `python -m src.model.dataset` from repo root (loads one
batch and prints shapes, does not train).
"""

import json
import random
from pathlib import Path

import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CONFIG_PATH = "configs/finetune.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class NoisyCleanDataset(Dataset):
    """Loads (noisy, clean) waveform pairs from a Phase 1 manifest.

    Returns tensors shaped [samples] (1D, mono -- matches what `load.py`'s
    `Demucs.forward()` accepts, which auto-unsqueezes a channel dim for 2D
    input; batching via DataLoader's default collate then produces
    [batch, samples], which the model also accepts directly).
    """

    def __init__(self, manifest_path: str, cfg: dict):
        with open(manifest_path, "r") as f:
            self.entries = json.load(f)
        self.sample_rate = cfg["sample_rate"]
        self.segment_samples = int(cfg["training"]["segment_seconds"] * self.sample_rate)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_wav(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise RuntimeError(
                f"{path} has sample rate {sr}, expected {self.sample_rate} "
                f"(configs/finetune.yaml['sample_rate']) -- Phase 1 output "
                f"should already be at the target rate, so this indicates a "
                f"real problem, not expected variation."
            )
        # soundfile returns [samples, channels]; Phase 1 output is mono
        # PCM_16 (confirmed in the development log), so squeeze to 1D [samples].
        return torch.from_numpy(data).transpose(0, 1).contiguous().squeeze(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[index]
        noisy = self._load_wav(entry["noisy_path"])
        clean = self._load_wav(entry["clean_path"])

        length = noisy.shape[-1]
        if length != clean.shape[-1]:
            raise RuntimeError(
                f"noisy/clean length mismatch for pair_id={entry['pair_id']}: "
                f"{length} vs {clean.shape[-1]} -- Phase 1 mixing should "
                f"always produce time-aligned same-length pairs."
            )

        if length >= self.segment_samples:
            # Same random crop offset for both, so noisy/clean stay aligned.
            max_start = length - self.segment_samples
            start = random.randint(0, max_start)
            noisy = noisy[start:start + self.segment_samples]
            clean = clean[start:start + self.segment_samples]
        else:
            pad = self.segment_samples - length
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))

        return noisy, clean


def _make_oversampling_weights(dataset: NoisyCleanDataset, cfg: dict) -> list[float]:
    """Per-pair sampling weights for v2's train-only hard-mixture oversampling.

    Pairs with manifest snr_db < snr_hard_threshold_db (the harder,
    lower-SNR half of the -5..15dB range) get `oversample_factor` x the
    weight of easier pairs -- WeightedRandomSampler then draws harder
    pairs more often per epoch. snr_db already exists per-pair in the
    Phase 1 manifest, no new data/labels needed.
    """
    oversample_cfg = cfg["training_run"]["oversample_hard_mixtures"]
    threshold = oversample_cfg["snr_hard_threshold_db"]
    factor = oversample_cfg["oversample_factor"]

    weights = []
    for entry in dataset.entries:
        if entry["snr_db"] < threshold:
            weights.append(factor)
        else:
            weights.append(1.0)
    return weights


def _curriculum_progress(epoch: int, max_epochs: int, ramp_epoch_fraction: float) -> float:
    """Linear ramp progress in [0.0, 1.0], per v4's curriculum schedule.

    `epoch` is 1-indexed (matches finetune.py's training loop convention).
    Progress is 0.0 at epoch 1 (uniform/true distribution), reaches 1.0 at
    epoch `ramp_epoch_fraction * max_epochs` (full end-weights), and stays
    at 1.0 for all epochs after that ("full difficulty thereafter").
    """
    ramp_epochs = max(1, round(ramp_epoch_fraction * max_epochs))
    progress = (epoch - 1) / ramp_epochs
    return min(1.0, max(0.0, progress))


def make_curriculum_sampler(
    dataset: "NoisyCleanDataset", cfg: dict, epoch: int
) -> WeightedRandomSampler:
    """Builds a fresh WeightedRandomSampler for the given (1-indexed) epoch,
    per v4's curriculum schedule (configs/finetune.yaml
    ['training_run_v4']['curriculum']).

    Two independent, multiplicative per-pair weight axes (both driven by
    the same linear ramp progress value, see `_curriculum_progress()`):
      - SNR axis: pairs with snr_db < snr_hard_threshold_db get a weight
        that ramps from 1.0 (epoch 1) to snr_end_weight (end of ramp).
      - Category axis: gunshot pairs get a weight that ramps from 1.0 to
        category_end_weight over the same schedule.
    A pair that is BOTH low-SNR AND gunshot (the diagnosed worst cell,
    gunshot x -5..0dB) gets both factors multiplied together -- the two
    axes compound exactly where the diagnosis says the failure is
    concentrated, not just additively.

    Called fresh each epoch by train_v4() (not cached) since the weights
    themselves change every epoch during the ramp -- recomputing per-pair
    weights over ~2162 train pairs is a negligible cost next to one epoch
    of forward/backward passes.
    """
    curriculum_cfg = cfg["training_run_v4"]["curriculum"]
    max_epochs = cfg["training_run_v4"]["max_epochs"]
    progress = _curriculum_progress(epoch, max_epochs, curriculum_cfg["ramp_epoch_fraction"])

    snr_threshold = curriculum_cfg["snr_hard_threshold_db"]
    snr_end_weight = curriculum_cfg["snr_end_weight"]
    category_end_weight = curriculum_cfg["category_end_weight"]

    snr_weight_now = 1.0 + progress * (snr_end_weight - 1.0)
    category_weight_now = 1.0 + progress * (category_end_weight - 1.0)

    weights = []
    for entry in dataset.entries:
        w = 1.0
        if entry["snr_db"] < snr_threshold:
            w *= snr_weight_now
        if entry["noise_category"] == "gunshot":
            w *= category_weight_now
        weights.append(w)

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def make_dataloader(
    split: str,
    cfg: dict | None = None,
    batch_size: int | None = None,
    use_oversampling: bool = False,
) -> DataLoader:
    """Builds a DataLoader for the given split ('train', 'val', or 'test').

    `batch_size` overrides `cfg["training"]["batch_size"]` (the 3b
    smoke-test value, still used by `smoke_test_train()`) -- Phase 3c's
    real run passes `cfg["training_run"]["batch_size"]` instead.

    `use_oversampling`: v2-only, train-split-only lever. When True AND
    split == "train" AND cfg["training_run"]["oversample_hard_mixtures"]
    ["enabled"] is True, replaces uniform shuffling with a
    WeightedRandomSampler biased toward lower-SNR (harder) pairs. val/test
    dataloaders NEVER apply this (this function only checks the sampler
    path when split == "train"), so they always stay uniform/representative
    of the true -5..15dB spread -- required for honest evaluation.
    """
    if cfg is None:
        cfg = load_config()
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]

    manifest_path = Path("manifests") / f"{split}.json"
    dataset = NoisyCleanDataset(str(manifest_path), cfg)

    oversample_cfg = cfg.get("training_run", {}).get("oversample_hard_mixtures", {})
    if split == "train" and use_oversampling and oversample_cfg.get("enabled", False):
        weights = _make_oversampling_weights(dataset, cfg)
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg["training"]["num_workers"],
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg["training"]["num_workers"],
    )


def make_curriculum_train_dataloader(cfg: dict, epoch: int, batch_size: int) -> DataLoader:
    """v4-only: builds the TRAIN DataLoader for one specific (1-indexed)
    epoch, using make_curriculum_sampler()'s epoch-dependent weights.

    Train-split only, by construction -- there is no val/test variant of
    this function. val/test always go through the existing make_dataloader()
    with its default uniform shuffling (use_oversampling left False), same
    as every prior phase -- this function does not touch or wrap that path.
    """
    manifest_path = Path("manifests") / "train.json"
    dataset = NoisyCleanDataset(str(manifest_path), cfg)
    sampler = make_curriculum_sampler(dataset, cfg, epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg["training"]["num_workers"],
    )


if __name__ == "__main__":
    cfg = load_config()
    loader = make_dataloader("train", cfg)
    noisy_batch, clean_batch = next(iter(loader))
    print(f"noisy_batch: shape={tuple(noisy_batch.shape)}, dtype={noisy_batch.dtype}")
    print(f"clean_batch: shape={tuple(clean_batch.shape)}, dtype={clean_batch.dtype}")
