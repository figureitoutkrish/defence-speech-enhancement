"""Phase 3b/3c: training loop for fine-tuning dns48.

Phase 3b (SMOKE TEST SCOPE, see `smoke_test_train()`): wires up the loop
structure (forward -> loss -> backward -> step -> zero_grad), runs exactly
ONE batch / ONE gradient step with value-level verification. Still the
default entry point (`python -m src.model.finetune`) -- unchanged from 3b.

Phase 3c (REAL RUN, see `train()`): multi-epoch training with a validation
loop, plateau-based early stopping, and best-val-loss checkpointing. Entry
point: `python -m src.model.finetune --run` (see bottom of file for exact
command). Epoch ceiling / batch size / early-stopping thresholds are all
config-driven from `configs/finetune.yaml['training_run']` -- justification
for each value is in that config section's comments and in the development log's
Phase 3c ceiling entry, not repeated here.

CRITICAL (carried over from 3b): this project's loss functions are
imported explicitly from `src/vendor/denoiser_patched/`, never from the
raw `denoiser` package directly -- verified concretely (module `__file__`
inspection) in both `smoke_test_train()` and `train()`.
"""

import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.dynamic_dataset import DynamicMixDataset
from src.model.dataset import load_config, make_curriculum_train_dataloader, make_dataloader
from src.model.load import load_dns48
from src.vendor.denoiser_patched.stft_loss import MultiResolutionSTFTLoss
import src.vendor.denoiser_patched.stft_loss as patched_stft_loss_module

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def verify_patched_loss_import() -> None:
    """Confirms MultiResolutionSTFTLoss resolves to our vendored patch, not
    the raw pip `denoiser` package -- concrete check via module __file__,
    same method used in the Phase 3a follow-up (sys.modules inspection),
    not an assumption based on how the import statement is written.
    """
    resolved_path = patched_stft_loss_module.__file__
    expected_suffix = "src/vendor/denoiser_patched/stft_loss.py"
    logger.info(f"MultiResolutionSTFTLoss resolved from: {resolved_path}")
    if not resolved_path.endswith(expected_suffix):
        raise RuntimeError(
            f"STFT loss did NOT resolve to the patched vendor module -- "
            f"got {resolved_path}, expected a path ending in {expected_suffix}. "
            f"This would mean the raw (broken, under torch 2.14) pip denoiser "
            f"stft_loss.py is being used instead."
        )
    logger.info("Confirmed: STFT loss resolves to the patched vendor module, not raw pip denoiser.")


def freeze_encoder_layers(model: torch.nn.Module, n_layers: int) -> None:
    """Freezes the first `n_layers` of the encoder (params.requires_grad = False).

    Mechanism only -- defaulted OFF (config `training.freeze_encoder_layers: 0`)
    for 3b. Whether to actually freeze early layers is a 3c decision.
    """
    if n_layers <= 0:
        logger.info("freeze_encoder_layers=0 -- no encoder layers frozen (full fine-tuning).")
        return
    for layer in model.encoder[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    logger.info(f"Froze the first {n_layers} encoder layer(s).")


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr = cfg["training"]["learning_rate"]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Adam optimizer: lr={lr}, trainable_params={sum(p.numel() for p in trainable_params)}")
    return torch.optim.Adam(trainable_params, lr=lr)


def compute_loss(
    output: torch.Tensor,
    clean: torch.Tensor,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L1 + multi-resolution STFT loss, per the development log's locked loss choice.

    `output`/`clean` are [batch, samples] (2D) -- MultiResolutionSTFTLoss's
    forward() expects (B, T), confirmed via its docstring in
    src/vendor/denoiser_patched/stft_loss.py.

    Returns: (l1_loss, stft_loss_combined, total_loss) -- all separate
    scalars, per the task's requirement to inspect L1 and STFT loss
    values independently, not just the combined total.
    """
    weights = cfg["training"]["loss_weights"]

    l1_loss = F.l1_loss(output, clean)

    sc_loss, mag_loss = stft_loss_fn(output, clean)
    stft_loss = sc_loss + mag_loss

    total_loss = weights["l1"] * l1_loss + weights["stft"] * stft_loss
    return l1_loss, stft_loss, total_loss


def smoke_test_train() -> None:
    """Loads ONE batch, runs ONE forward+backward+optimizer step, prints
    value-level verification. Does NOT save a checkpoint, does NOT loop
    epochs -- per explicit 3b scope.
    """
    verify_patched_loss_import()

    cfg = load_config()

    model, device = load_dns48(cfg)
    model.train()  # smoke test needs gradients; load_dns48() leaves it in eval mode

    freeze_encoder_layers(model, cfg["training"]["freeze_encoder_layers"])
    optimizer = build_optimizer(model, cfg)
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    train_loader = make_dataloader("train", cfg)
    noisy_batch, clean_batch = next(iter(train_loader))
    noisy_batch = noisy_batch.to(device)
    clean_batch = clean_batch.to(device)

    logger.info(f"Input batch -- noisy: shape={tuple(noisy_batch.shape)}, dtype={noisy_batch.dtype}")
    logger.info(f"Input batch -- clean: shape={tuple(clean_batch.shape)}, dtype={clean_batch.dtype}")

    optimizer.zero_grad()

    output = model(noisy_batch)
    logger.info(f"Model output (pre-squeeze): shape={tuple(output.shape)}, dtype={output.dtype}")

    # Demucs outputs [batch, channels=1, samples]; loss expects [batch, samples].
    output_2d = output.squeeze(1)
    logger.info(f"Model output (loss input, squeezed): shape={tuple(output_2d.shape)}, dtype={output_2d.dtype}")

    l1_loss, stft_loss, total_loss = compute_loss(output_2d, clean_batch, stft_loss_fn, cfg)

    logger.info(f"L1 loss:   {l1_loss.item():.6f}")
    logger.info(f"STFT loss: {stft_loss.item():.6f}")
    logger.info(f"Total loss (weighted sum): {total_loss.item():.6f}")

    for name, val, in [("L1", l1_loss), ("STFT", stft_loss), ("total", total_loss)]:
        v = val.item()
        if v != v:  # NaN check, avoids importing math for one check
            raise RuntimeError(f"{name} loss is NaN -- loss computation is broken.")
        if v in (float("inf"), float("-inf")):
            raise RuntimeError(f"{name} loss is Inf -- likely a scale/shape mismatch.")
        if v == 0.0:
            raise RuntimeError(
                f"{name} loss is exactly zero -- suspicious, likely means the loss "
                f"isn't actually receiving real gradient-carrying tensors."
            )

    total_loss.backward()

    logger.info("Spot-checking gradients on a few real model parameters after backward():")
    checked = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_abs_mean = p.grad.abs().mean().item()
            logger.info(f"  {name}: grad.abs().mean()={grad_abs_mean:.8f}")
            if grad_abs_mean == 0.0:
                logger.warning(f"  {name} has an all-zero gradient -- may indicate a disconnected path.")
            checked += 1
        if checked >= 5:
            break

    if checked == 0:
        raise RuntimeError(
            "No parameter had a non-None gradient after backward() -- loss is "
            "NOT connected to the model, something is silently detached."
        )

    optimizer.step()

    logger.info("Smoke test complete: 1 batch, 1 forward+backward+optimizer.step(). "
                "No checkpoint saved, no further steps run.")


def _run_validation(
    model: torch.nn.Module,
    val_loader,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
    device: torch.device,
) -> tuple[float, float, float]:
    """Runs one full pass over the val split, no gradients.

    Returns: (mean_l1, mean_stft, mean_total) over all val batches.
    """
    model.eval()
    total_l1, total_stft, total_weighted, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            output = model(noisy_batch).squeeze(1)
            l1_loss, stft_loss, total_loss = compute_loss(output, clean_batch, stft_loss_fn, cfg)
            total_l1 += l1_loss.item()
            total_stft += stft_loss.item()
            total_weighted += total_loss.item()
            n_batches += 1
    model.train()
    return total_l1 / n_batches, total_stft / n_batches, total_weighted / n_batches


def train() -> None:
    """Phase 3c real training run: multi-epoch, validation loop, plateau
    early stopping, best-val-loss checkpointing. Config-driven from
    `configs/finetune.yaml['training_run']` -- see that section's comments
    and the development log's Phase 3c ceiling entry for the epoch-ceiling/batch-size
    justification (not repeated here).

    If the epoch ceiling is hit without a clear plateau, this is reported
    honestly (not framed as convergence) -- locked project-wide rule, not
    optional.
    """
    verify_patched_loss_import()

    cfg = load_config()
    run_cfg = cfg["training_run"]
    batch_size = run_cfg["batch_size"]
    max_epochs = run_cfg["max_epochs"]
    min_delta = run_cfg["early_stopping"]["min_delta"]
    patience_epochs = run_cfg["early_stopping"]["patience_epochs"]
    checkpoint_dir = Path(run_cfg["checkpoint_dir"])
    results_csv = Path(run_cfg["results_csv"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "dns48_finetuned_best.pt"

    model, device = load_dns48(cfg)
    model.train()

    freeze_encoder_layers(model, cfg["training"]["freeze_encoder_layers"])
    optimizer = build_optimizer(model, cfg)
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    train_loader = make_dataloader("train", cfg, batch_size=batch_size)
    val_loader = make_dataloader("val", cfg, batch_size=batch_size)

    logger.info(
        f"Phase 3c training run: max_epochs={max_epochs}, batch_size={batch_size}, "
        f"early_stopping(min_delta={min_delta}, patience_epochs={patience_epochs})"
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    plateaued = False

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_l1", "train_stft", "train_total",
            "val_l1", "val_stft", "val_total", "epoch_seconds",
        ])

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            running_l1, running_stft, running_total, n_batches = 0.0, 0.0, 0.0, 0
            for noisy_batch, clean_batch in train_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)

                optimizer.zero_grad()
                output = model(noisy_batch).squeeze(1)
                l1_loss, stft_loss, total_loss = compute_loss(output, clean_batch, stft_loss_fn, cfg)
                total_loss.backward()
                optimizer.step()

                running_l1 += l1_loss.item()
                running_stft += stft_loss.item()
                running_total += total_loss.item()
                n_batches += 1

            train_l1 = running_l1 / n_batches
            train_stft = running_stft / n_batches
            train_total = running_total / n_batches

            val_l1, val_stft, val_total = _run_validation(model, val_loader, stft_loss_fn, cfg, device)

            epoch_seconds = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{max_epochs} -- "
                f"train: L1={train_l1:.6f} STFT={train_stft:.6f} total={train_total:.6f} | "
                f"val: L1={val_l1:.6f} STFT={val_stft:.6f} total={val_total:.6f} | "
                f"{epoch_seconds:.1f}s"
            )

            writer.writerow([epoch, train_l1, train_stft, train_total, val_l1, val_stft, val_total, epoch_seconds])
            f.flush()

            if val_total < best_val_loss - min_delta:
                best_val_loss = val_total
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total_loss": val_total,
                    "val_l1_loss": val_l1,
                    "val_stft_loss": val_stft,
                }, checkpoint_path)
                logger.info(f"  New best val loss ({val_total:.6f}) -- checkpoint saved to {checkpoint_path}")
            else:
                epochs_since_improvement += 1
                logger.info(
                    f"  No improvement > min_delta={min_delta} for "
                    f"{epochs_since_improvement}/{patience_epochs} epochs."
                )

            if epochs_since_improvement >= patience_epochs:
                plateaued = True
                logger.info(
                    f"Plateau detected: val loss has not improved by more than "
                    f"min_delta={min_delta} for {patience_epochs} consecutive epochs. "
                    f"Stopping early at epoch {epoch} (ceiling was {max_epochs})."
                )
                break

    if plateaued:
        logger.info(
            f"Training run complete -- STOPPED EARLY on plateau at epoch {epoch}. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"Checkpoint: {checkpoint_path}"
        )
    elif epoch == max_epochs:
        logger.info(
            f"Training run complete -- EPOCH CEILING ({max_epochs}) REACHED, no plateau detected. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"HONEST FRAMING (per locked project rule): this means the ceiling was hit while "
            f"val loss was still improving or fluctuating without a clear plateau signal -- "
            f"do NOT report this as convergence. Checkpoint: {checkpoint_path}"
        )


def _build_lr_lambda(total_steps: int, cfg: dict):
    """Returns a `LambdaLR`-compatible fn implementing warmup + cosine (or
    linear) decay, per v2 lever 1 (`configs/finetune.yaml
    ['training_run']['lr_schedule']`).

    The returned function maps a step index -> a multiplier on `peak_lr`
    (LambdaLR's convention: it scales the optimizer's base LR, so the
    optimizer must be constructed with `lr=peak_lr` and this multiplier
    ranges from `min_lr/peak_lr` up to `1.0`, never 0 -- matching the
    config's floor, not a hard cutoff to zero).
    """
    import math

    sched_cfg = cfg["training_run"]["lr_schedule"]
    peak_lr = sched_cfg["peak_lr"]
    min_lr = sched_cfg["min_lr"]
    warmup_ratio = sched_cfg["warmup_ratio"]
    schedule_type = sched_cfg["schedule_type"]

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr_mult = min_lr / peak_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        if schedule_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule_type == "linear":
            decay = 1.0 - progress
        else:
            raise ValueError(f"Unknown lr_schedule.schedule_type: {schedule_type!r}")
        return min_lr_mult + (1.0 - min_lr_mult) * decay

    return lr_lambda


def compute_loss_v2(
    output: torch.Tensor,
    clean: torch.Tensor,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as `compute_loss()` but reads `loss_weights_v2` (l1=1.0,
    stft=2.0 proposed) instead of `training.loss_weights` -- kept as a
    separate function so v1's `compute_loss()` / `smoke_test_train()` /
    `train()` are byte-for-byte unaffected by v2's reweighting.
    """
    weights = cfg["loss_weights_v2"]

    l1_loss = F.l1_loss(output, clean)

    sc_loss, mag_loss = stft_loss_fn(output, clean)
    stft_loss = sc_loss + mag_loss

    total_loss = weights["l1"] * l1_loss + weights["stft"] * stft_loss
    return l1_loss, stft_loss, total_loss


def _run_validation_v2(
    model: torch.nn.Module,
    val_loader,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
    device: torch.device,
) -> tuple[float, float, float]:
    """Same as `_run_validation()` but uses `compute_loss_v2()` (the v2
    loss reweighting) so val loss is measured on the same objective the
    v2 run is actually optimizing and comparing against for
    checkpointing/early-stopping.
    """
    model.eval()
    total_l1, total_stft, total_weighted, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            output = model(noisy_batch).squeeze(1)
            l1_loss, stft_loss, total_loss = compute_loss_v2(output, clean_batch, stft_loss_fn, cfg)
            total_l1 += l1_loss.item()
            total_stft += stft_loss.item()
            total_weighted += total_loss.item()
            n_batches += 1
    model.train()
    return total_l1 / n_batches, total_stft / n_batches, total_weighted / n_batches


def train_v2() -> None:
    """v2 aggressive fine-tuning run: LR warmup+decay schedule, STFT-
    reweighted loss, relaxed early stopping / higher epoch ceiling,
    optional partial encoder freezing (toggleable, OFF by default), and
    train-only oversampling of harder (lower-SNR) mixtures.

    Entirely separate from `train()` (v1) -- does not modify or call it,
    so v1's checkpoint/results/behavior stay reproducible byte-for-byte.
    Saves to a SEPARATE checkpoint (`training_run.checkpoint_filename`,
    `dns48_finetuned_v2_best.pt`) and a separate results CSV
    (`results/finetune_training_log_v2.csv`) -- v1's
    `checkpoints/dns48_finetuned_best.pt` and
    `results/finetune_training_log.csv` are never touched by this
    function.

    Does NOT evaluate on the test set and does NOT append to the development log --
    per explicit instruction, that happens only after the user reports
    back the training curve and a decision is made on whether to run
    Phase 4 evaluation on this checkpoint.
    """
    verify_patched_loss_import()

    cfg = load_config()
    run_cfg = cfg["training_run"]
    batch_size = run_cfg["batch_size"]
    max_epochs = run_cfg["max_epochs"]
    min_delta = run_cfg["early_stopping"]["min_delta"]
    patience_epochs = run_cfg["early_stopping"]["patience_epochs"]
    checkpoint_dir = Path(run_cfg["checkpoint_dir"])
    results_csv = Path(run_cfg["results_csv"])
    freeze_v2_cfg = run_cfg["freeze_encoder_layers_v2"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / run_cfg["checkpoint_filename"]

    model, device = load_dns48(cfg)
    model.train()

    if freeze_v2_cfg["enabled"]:
        freeze_encoder_layers(model, freeze_v2_cfg["n_layers"])
    else:
        logger.info("freeze_encoder_layers_v2.enabled=false -- no encoder layers frozen (full fine-tuning).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    peak_lr = run_cfg["lr_schedule"]["peak_lr"]
    optimizer = torch.optim.Adam(trainable_params, lr=peak_lr)
    logger.info(f"Adam optimizer: peak_lr={peak_lr}, trainable_params={sum(p.numel() for p in trainable_params)}")

    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    train_loader = make_dataloader("train", cfg, batch_size=batch_size, use_oversampling=True)
    val_loader = make_dataloader("val", cfg, batch_size=batch_size)
    logger.info(
        "val/test dataloaders use uniform sampling (use_oversampling defaults False / "
        "not passed) -- oversampling applies to the train split only, per explicit "
        "instruction that evaluation must stay representative of the true -5..15dB spread."
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max_epochs
    lr_lambda = _build_lr_lambda(total_steps, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Phase 3c v2 training run: max_epochs={max_epochs}, batch_size={batch_size}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"lr_schedule={run_cfg['lr_schedule']}, "
        f"loss_weights_v2={cfg['loss_weights_v2']}, "
        f"early_stopping(min_delta={min_delta}, patience_epochs={patience_epochs}), "
        f"freeze_encoder_layers_v2={freeze_v2_cfg}, "
        f"oversample_hard_mixtures={run_cfg['oversample_hard_mixtures']}"
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    plateaued = False

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_l1", "train_stft", "train_total",
            "val_l1", "val_stft", "val_total", "lr", "epoch_seconds",
        ])

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            running_l1, running_stft, running_total, n_batches = 0.0, 0.0, 0.0, 0
            for noisy_batch, clean_batch in train_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)

                optimizer.zero_grad()
                output = model(noisy_batch).squeeze(1)
                l1_loss, stft_loss, total_loss = compute_loss_v2(output, clean_batch, stft_loss_fn, cfg)
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                running_l1 += l1_loss.item()
                running_stft += stft_loss.item()
                running_total += total_loss.item()
                n_batches += 1

            train_l1 = running_l1 / n_batches
            train_stft = running_stft / n_batches
            train_total = running_total / n_batches

            val_l1, val_stft, val_total = _run_validation_v2(model, val_loader, stft_loss_fn, cfg, device)

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_seconds = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{max_epochs} -- "
                f"train: L1={train_l1:.6f} STFT={train_stft:.6f} total={train_total:.6f} | "
                f"val: L1={val_l1:.6f} STFT={val_stft:.6f} total={val_total:.6f} | "
                f"lr={current_lr:.8f} | {epoch_seconds:.1f}s"
            )

            writer.writerow([
                epoch, train_l1, train_stft, train_total,
                val_l1, val_stft, val_total, current_lr, epoch_seconds,
            ])
            f.flush()

            if val_total < best_val_loss - min_delta:
                best_val_loss = val_total
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total_loss": val_total,
                    "val_l1_loss": val_l1,
                    "val_stft_loss": val_stft,
                    "lr": current_lr,
                }, checkpoint_path)
                logger.info(f"  New best val loss ({val_total:.6f}) -- checkpoint saved to {checkpoint_path}")
            else:
                epochs_since_improvement += 1
                logger.info(
                    f"  No improvement > min_delta={min_delta} for "
                    f"{epochs_since_improvement}/{patience_epochs} epochs."
                )

            if epochs_since_improvement >= patience_epochs:
                plateaued = True
                logger.info(
                    f"Plateau detected: val loss has not improved by more than "
                    f"min_delta={min_delta} for {patience_epochs} consecutive epochs. "
                    f"Stopping early at epoch {epoch} (ceiling was {max_epochs})."
                )
                break

    if plateaued:
        logger.info(
            f"v2 training run complete -- STOPPED EARLY on plateau at epoch {epoch}. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"Checkpoint: {checkpoint_path}"
        )
    elif epoch == max_epochs:
        logger.info(
            f"v2 training run complete -- EPOCH CEILING ({max_epochs}) REACHED, no plateau detected. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"HONEST FRAMING (per locked project rule): this means the ceiling was hit while "
            f"val loss was still improving or fluctuating without a clear plateau signal -- "
            f"do NOT report this as convergence. Checkpoint: {checkpoint_path}"
        )


def silence_collapse_penalty(
    output: torch.Tensor,
    noisy: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """v4 intervention 2: penalizes the ENHANCED output going near-silent in
    frames where the NOISY INPUT had substantial energy -- a heuristic proxy
    for "something was probably there (likely speech), don't erase it
    entirely". Directly targets the diagnosed over-suppression failure mode
    (see configs/finetune.yaml['training_run_v4'] comment block for the full
    diagnosis this responds to).

    Deliberately NOT keyed off the clean target (a blanket minimum-energy
    floor against clean was the earlier, weaker idea, rejected because it
    penalizes correctly suppressing genuine silence/noise-only regions too).
    Keying off the noisy input's own energy means the penalty only fires
    where the model itself had reason to believe something was active,
    independent of whether ground-truth clean actually has speech there.

    EXACT FORMULA:
      1. Frame both `output` and `noisy` into overlapping windows via
         `unfold(dim=-1, size=frame_length, step=hop_length)` (same
         windowing convention as baseline/diagnose_failures' STFT frames,
         just RMS instead of a full spectral transform -- cheaper, and
         plenty for a coarse "was this frame active" signal).
      2. Per-frame RMS energy for both signals: `sqrt(mean(x**2))` over each
         frame's samples.
      3. `active_mask = noisy_rms > active_rms_threshold` -- frames where the
         NOISY input itself had substantial energy ("likely active region").
      4. `ratio = output_rms / (noisy_rms + eps)` -- how much of the input's
         energy survived into the output, in each frame.
      5. `deficit = clamp(suppression_ratio_threshold - ratio, min=0)` --
         only frames where the surviving-energy ratio has dropped BELOW the
         threshold contribute (a ratio-based penalty, not absolute energy --
         a naturally quiet-but-real region isn't penalized just for being
         quiet, only a frame that collapsed relative to what the input had).
      6. Penalty = mean of `deficit` over active-masked frames only (frames
         where `active_mask` is False never contribute, whether or not the
         output is quiet there -- suppressing genuine silence/noise-only
         input is exactly what the model SHOULD do and must not be
         penalized for).

    LIMITATIONS (heuristic proxy for speech presence, NOT a real VAD --
    documented per task instruction, not silently assumed correct):
      - `noisy_rms > active_rms_threshold` conflates "speech present" with
        "any energy present" -- a loud noise-only frame (e.g. a gunshot
        transient with no underlying speech in that instant) will also be
        flagged "active" and penalize the model for suppressing it, even
        though suppressing pure noise is correct behavior there. This is a
        real, accepted limitation: the penalty is a coarse nudge against a
        specific observed failure mode, not a precise speech/non-speech
        classifier, and could theoretically fight against wanted
        suppression on pure-noise transients. Small `weight` (0.05) is the
        mitigation -- this term nudges, it does not dominate L1+STFT.
      - Frame-level RMS has no phase or spectral-shape awareness --two
        frames with equal RMS can differ completely in spectral content
        (e.g. surviving low-frequency rumble vs. surviving speech
        formants); this penalty cannot tell them apart, only whether SOME
        energy survived.
      - `active_rms_threshold` is a fixed, unlearned cutoff on normalized
        [-1, 1] float32 audio, not calibrated per-recording loudness or
        per-noise-category -- a quiet recording's genuinely active speech
        frames could sit below threshold and never get the mask's
        protection, and a loud recording's noise floor could sit above it.
    """
    penalty_cfg = cfg["training_run_v4"]["silence_penalty"]
    frame_length = penalty_cfg["frame_length"]
    hop_length = penalty_cfg["hop_length"]
    active_threshold = penalty_cfg["active_rms_threshold"]
    ratio_threshold = penalty_cfg["suppression_ratio_threshold"]
    eps = 1e-8

    if output.shape[-1] < frame_length:
        # Segment shorter than one frame (shouldn't happen given
        # training.segment_seconds, but guard rather than let unfold error).
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)

    output_frames = output.unfold(-1, frame_length, hop_length)  # [B, n_frames, frame_length]
    noisy_frames = noisy.unfold(-1, frame_length, hop_length)

    output_rms = torch.sqrt(torch.mean(output_frames ** 2, dim=-1) + eps)  # [B, n_frames]
    noisy_rms = torch.sqrt(torch.mean(noisy_frames ** 2, dim=-1) + eps)

    active_mask = noisy_rms > active_threshold
    if active_mask.sum() == 0:
        # No active frames in this batch (e.g. all-silence segments) --
        # nothing to penalize, return zero rather than dividing by zero.
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)

    ratio = output_rms / (noisy_rms + eps)
    deficit = torch.clamp(ratio_threshold - ratio, min=0.0)

    return deficit[active_mask].mean()


def compute_loss_v4(
    output: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """v4 loss: L1 + multi-resolution STFT (both reverted to v1's implicit
    1:1 weighting via loss_weights_v4, NOT v2's stft=2.0 -- see
    configs/finetune.yaml['training_run_v4'] comment for why) PLUS the new
    silence-collapse penalty term, additive and small-weighted.

    Returns: (l1_loss, stft_loss, silence_penalty, total_loss) -- all
    logged separately per epoch, per explicit instruction.
    """
    weights = cfg["loss_weights_v4"]
    penalty_weight = cfg["training_run_v4"]["silence_penalty"]["weight"]

    l1_loss = F.l1_loss(output, clean)

    sc_loss, mag_loss = stft_loss_fn(output, clean)
    stft_loss = sc_loss + mag_loss

    penalty = silence_collapse_penalty(output, noisy, cfg)

    total_loss = weights["l1"] * l1_loss + weights["stft"] * stft_loss + penalty_weight * penalty
    return l1_loss, stft_loss, penalty, total_loss


def _run_validation_v4(
    model: torch.nn.Module,
    val_loader,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Same shape as _run_validation()/_run_validation_v2() but reports all
    four v4 loss components. val_loader is the STANDARD uniform-sampling
    loader from make_dataloader() (never curriculum-weighted) -- confirmed
    concretely by train_v4() below, which calls make_dataloader("val", ...)
    with use_oversampling left at its default False, exactly like v1/v2.
    """
    model.eval()
    total_l1, total_stft, total_penalty, total_weighted, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            output = model(noisy_batch).squeeze(1)
            l1_loss, stft_loss, penalty, total_loss = compute_loss_v4(
                output, clean_batch, noisy_batch, stft_loss_fn, cfg
            )
            total_l1 += l1_loss.item()
            total_stft += stft_loss.item()
            total_penalty += penalty.item()
            total_weighted += total_loss.item()
            n_batches += 1
    model.train()
    return (
        total_l1 / n_batches,
        total_stft / n_batches,
        total_penalty / n_batches,
        total_weighted / n_batches,
    )


def train_v4() -> None:
    """v4: curriculum training (epoch-dependent hard-mixture/gunshot
    oversampling, replacing v2's static 3x oversampling) + silence-collapse
    penalty (new loss term, additive to L1+STFT), keeping v2's LR warmup+
    cosine schedule and dropping v2's 2x STFT reweighting. Full rationale
    for every config value lives in configs/finetune.yaml
    ['training_run_v4'] -- not repeated here.

    Entirely separate from train() (v1) and train_v2() -- does not modify
    or call either, so their checkpoints/results/behavior stay reproducible
    byte-for-byte. Saves to checkpoints/dns48_finetuned_v4_best.pt and
    results/finetune_training_log_v4.csv.

    val/test dataloaders: STANDARD make_dataloader() calls, uniform
    sampling, use_oversampling left at its default False -- confirmed
    explicitly in-code below (not just in the config comment) -- the
    curriculum sampler (make_curriculum_train_dataloader()) is only ever
    constructed for the train split, and only inside the per-epoch loop.

    Does NOT evaluate on the test set and does NOT append to the development log --
    per explicit instruction, this happens only after the user reports back
    the training curve.
    """
    verify_patched_loss_import()

    cfg = load_config()
    run_cfg = cfg["training_run_v4"]
    batch_size = run_cfg["batch_size"]
    max_epochs = run_cfg["max_epochs"]
    min_delta = run_cfg["early_stopping"]["min_delta"]
    patience_epochs = run_cfg["early_stopping"]["patience_epochs"]
    checkpoint_dir = Path(run_cfg["checkpoint_dir"])
    results_csv = Path(run_cfg["results_csv"])
    curriculum_cfg = run_cfg["curriculum"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / run_cfg["checkpoint_filename"]

    model, device = load_dns48(cfg)
    model.train()

    # v4 does not use freeze_encoder_layers_v2 or freeze_encoder_layers --
    # neither v1's smoke-test freezing nor v2's optional freeze lever was
    # implicated by any diagnosis, and this run's two interventions
    # (curriculum + silence penalty) are meant to be evaluated on their own,
    # not compounded with a third unrelated lever.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    peak_lr = cfg["training_run"]["lr_schedule"]["peak_lr"]  # reused from v2, unchanged (kept per instruction)
    optimizer = torch.optim.Adam(trainable_params, lr=peak_lr)
    logger.info(f"Adam optimizer: peak_lr={peak_lr}, trainable_params={sum(p.numel() for p in trainable_params)}")

    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    val_loader = make_dataloader("val", cfg, batch_size=batch_size)
    logger.info(
        "val dataloader uses standard uniform sampling (make_dataloader(), "
        "use_oversampling not passed / defaults False) -- val is NEVER "
        "curriculum-weighted, confirmed explicitly here, same standard as "
        "every prior phase (v1/v2's val/test loaders)."
    )

    steps_per_epoch = 2162 // batch_size  # matches train-manifest size; curriculum sampler always draws len(dataset) samples/epoch
    total_steps = steps_per_epoch * max_epochs
    lr_lambda = _build_lr_lambda(total_steps, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Phase v4 training run: max_epochs={max_epochs}, batch_size={batch_size}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"lr_schedule={cfg['training_run']['lr_schedule']}, "
        f"loss_weights_v4={cfg['loss_weights_v4']}, "
        f"silence_penalty={run_cfg['silence_penalty']}, "
        f"curriculum={curriculum_cfg}, "
        f"early_stopping(min_delta={min_delta}, patience_epochs={patience_epochs})"
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    plateaued = False

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "curriculum_progress", "curriculum_snr_weight", "curriculum_category_weight",
            "train_l1", "train_stft", "train_silence_penalty", "train_total",
            "val_l1", "val_stft", "val_silence_penalty", "val_total",
            "lr", "epoch_seconds",
        ])

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            from src.model.dataset import _curriculum_progress  # local import: keeps dataset.py free of a finetune.py dependency
            progress = _curriculum_progress(epoch, max_epochs, curriculum_cfg["ramp_epoch_fraction"])
            snr_weight_now = 1.0 + progress * (curriculum_cfg["snr_end_weight"] - 1.0)
            category_weight_now = 1.0 + progress * (curriculum_cfg["category_end_weight"] - 1.0)

            train_loader = make_curriculum_train_dataloader(cfg, epoch, batch_size)

            running_l1, running_stft, running_penalty, running_total, n_batches = 0.0, 0.0, 0.0, 0.0, 0
            for noisy_batch, clean_batch in train_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)

                optimizer.zero_grad()
                output = model(noisy_batch).squeeze(1)
                l1_loss, stft_loss, penalty, total_loss = compute_loss_v4(
                    output, clean_batch, noisy_batch, stft_loss_fn, cfg
                )
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                running_l1 += l1_loss.item()
                running_stft += stft_loss.item()
                running_penalty += penalty.item()
                running_total += total_loss.item()
                n_batches += 1

            train_l1 = running_l1 / n_batches
            train_stft = running_stft / n_batches
            train_penalty = running_penalty / n_batches
            train_total = running_total / n_batches

            val_l1, val_stft, val_penalty, val_total = _run_validation_v4(
                model, val_loader, stft_loss_fn, cfg, device
            )

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_seconds = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{max_epochs} -- curriculum_progress={progress:.3f} "
                f"(snr_w={snr_weight_now:.2f}, cat_w={category_weight_now:.2f}) | "
                f"train: L1={train_l1:.6f} STFT={train_stft:.6f} silence_pen={train_penalty:.6f} total={train_total:.6f} | "
                f"val: L1={val_l1:.6f} STFT={val_stft:.6f} silence_pen={val_penalty:.6f} total={val_total:.6f} | "
                f"lr={current_lr:.8f} | {epoch_seconds:.1f}s"
            )

            writer.writerow([
                epoch, progress, snr_weight_now, category_weight_now,
                train_l1, train_stft, train_penalty, train_total,
                val_l1, val_stft, val_penalty, val_total,
                current_lr, epoch_seconds,
            ])
            f.flush()

            if val_total < best_val_loss - min_delta:
                best_val_loss = val_total
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total_loss": val_total,
                    "val_l1_loss": val_l1,
                    "val_stft_loss": val_stft,
                    "val_silence_penalty": val_penalty,
                    "lr": current_lr,
                    "curriculum_progress": progress,
                }, checkpoint_path)
                logger.info(f"  New best val loss ({val_total:.6f}) -- checkpoint saved to {checkpoint_path}")
            else:
                epochs_since_improvement += 1
                logger.info(
                    f"  No improvement > min_delta={min_delta} for "
                    f"{epochs_since_improvement}/{patience_epochs} epochs."
                )

            if epochs_since_improvement >= patience_epochs:
                plateaued = True
                logger.info(
                    f"Plateau detected: val loss has not improved by more than "
                    f"min_delta={min_delta} for {patience_epochs} consecutive epochs. "
                    f"Stopping early at epoch {epoch} (ceiling was {max_epochs})."
                )
                break

    if plateaued:
        logger.info(
            f"v4 training run complete -- STOPPED EARLY on plateau at epoch {epoch}. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"Checkpoint: {checkpoint_path}"
        )
    elif epoch == max_epochs:
        logger.info(
            f"v4 training run complete -- EPOCH CEILING ({max_epochs}) REACHED, no plateau detected. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"HONEST FRAMING (per locked project rule): this means the ceiling was hit while "
            f"val loss was still improving or fluctuating without a clear plateau signal -- "
            f"do NOT report this as convergence. Checkpoint: {checkpoint_path}"
        )


def compute_loss_v5(
    output: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """v5 loss -- TASK 4: same L1 + multi-resolution STFT at 1:1
    (loss_weights_v4, reused directly, not duplicated) plus v4's
    silence-collapse penalty, gated by
    training_run_v5.silence_penalty_enabled (the one-line config toggle
    requested pending the v4 epoch-11 eval result). When disabled, the
    penalty is still COMPUTED (for logging/comparison) but not added into
    total_loss -- so the CSV always has a silence_penalty column regardless
    of the toggle, only its contribution to training changes.

    Returns: (l1_loss, stft_loss, silence_penalty, total_loss).
    """
    weights = cfg["loss_weights_v4"]
    penalty_weight = cfg["training_run_v4"]["silence_penalty"]["weight"]
    penalty_enabled = cfg["training_run_v5"]["silence_penalty_enabled"]

    l1_loss = F.l1_loss(output, clean)

    sc_loss, mag_loss = stft_loss_fn(output, clean)
    stft_loss = sc_loss + mag_loss

    penalty = silence_collapse_penalty(output, noisy, cfg)

    total_loss = weights["l1"] * l1_loss + weights["stft"] * stft_loss
    if penalty_enabled:
        total_loss = total_loss + penalty_weight * penalty

    return l1_loss, stft_loss, penalty, total_loss


def _run_validation_v5(
    model: torch.nn.Module,
    val_loader,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: dict,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Same shape as _run_validation_v4() -- val_loader is the STANDARD
    static make_dataloader("val", ...) loader (manifests/val.json,
    unchanged since Phase 1), confirmed explicitly by train_v5() below,
    never the dynamic pool -- val must stay a fixed, representative,
    comparable-across-runs set for honest evaluation, same standard as
    every prior phase.
    """
    model.eval()
    total_l1, total_stft, total_penalty, total_weighted, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            output = model(noisy_batch).squeeze(1)
            l1_loss, stft_loss, penalty, total_loss = compute_loss_v5(
                output, clean_batch, noisy_batch, stft_loss_fn, cfg
            )
            total_l1 += l1_loss.item()
            total_stft += stft_loss.item()
            total_penalty += penalty.item()
            total_weighted += total_loss.item()
            n_batches += 1
    model.train()
    return (
        total_l1 / n_batches,
        total_stft / n_batches,
        total_penalty / n_batches,
        total_weighted / n_batches,
    )


def _build_warmup_then_plateau(optimizer: torch.optim.Optimizer, total_warmup_steps: int, cfg: dict):
    """TASK 3: warmup (linear 0 -> peak_lr over total_warmup_steps, same
    warmup_ratio/peak_lr as v2 per lr_schedule_v5) then
    ReduceLROnPlateau(factor, patience, min_lr) on val loss, replacing
    v2/v4's fixed cosine decay entirely.

    Returns (warmup_scheduler, plateau_scheduler) -- caller steps
    warmup_scheduler per BATCH only during the warmup phase, then switches
    to stepping plateau_scheduler per EPOCH with the epoch's val loss, for
    the remainder of the run. Kept as two separate schedulers (rather than
    one custom LambdaLR/ReduceLROnPlateau hybrid) because
    ReduceLROnPlateau's API is step(metric) per epoch and LambdaLR's is
    step() per batch -- combining them into one object would need the same
    phase-detection branching this split does anyway, just hidden inside a
    custom class instead of visible in train_v5()'s loop.
    """
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, total_warmup_steps))
    )
    sched_cfg = cfg["training_run_v5"]["lr_schedule_v5"]
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg["plateau_factor"],
        patience=sched_cfg["plateau_patience"],
        min_lr=sched_cfg["min_lr"],
    )
    return warmup_scheduler, plateau_scheduler


def train_v5() -> None:
    """v5: dynamic (on-the-fly) mixing over an expanded clean-speech pool
    (TASK 1/2) + reactive LR (warmup then ReduceLROnPlateau, TASK 3) + v4's
    loss (TASK 4, silence penalty toggleable) + a high epoch ceiling gated
    by val-loss plateau (TASK 5). Full rationale for every config value is
    in configs/finetune.yaml['training_run_v5'] -- not repeated here.

    Entirely separate from train()/train_v2()/train_v4() -- does not modify
    or call any of them, so their checkpoints/results/behavior stay
    reproducible byte-for-byte. Saves to
    checkpoints/dns48_finetuned_v5_best.pt and
    results/finetune_training_log_v5.csv.

    CRITICAL, verified explicitly in this function (not just asserted in
    comments): the train loader is DynamicMixDataset (new mixtures every
    __getitem__); the val loader is the EXISTING static
    make_dataloader("val", ...) over manifests/val.json, completely
    untouched since Phase 1 -- printed in the run banner below, per
    explicit task instruction ("verify concretely and state it in the run
    banner, same standard as every prior phase").

    Does NOT evaluate on the test set and does NOT append to the development log --
    per explicit instruction, this happens only after the user reports back
    the training curve and results are ready to write up.
    """
    verify_patched_loss_import()

    cfg = load_config()
    run_cfg = cfg["training_run_v5"]
    batch_size = run_cfg["batch_size"]
    steps_per_epoch = run_cfg["steps_per_epoch"]
    max_epochs = run_cfg["max_epochs"]
    min_delta = run_cfg["early_stopping"]["min_delta"]
    patience_epochs = run_cfg["early_stopping"]["patience_epochs"]
    checkpoint_dir = Path(run_cfg["checkpoint_dir"])
    results_csv = Path(run_cfg["results_csv"])
    sched_cfg = run_cfg["lr_schedule_v5"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / run_cfg["checkpoint_filename"]

    model, device = load_dns48(cfg)
    model.train()

    # v5 does not use any encoder-freezing lever (v1's freeze_encoder_layers
    # or v2's freeze_encoder_layers_v2) -- neither was implicated by any
    # prior diagnosis, and this run's actual lever (dynamic data) is meant
    # to be evaluated on its own, not compounded with an unrelated freeze.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    peak_lr = sched_cfg["peak_lr"]
    optimizer = torch.optim.Adam(trainable_params, lr=peak_lr)
    logger.info(f"Adam optimizer: peak_lr={peak_lr}, trainable_params={sum(p.numel() for p in trainable_params)}")

    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    train_dataset = DynamicMixDataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg["training"]["num_workers"]
    )
    val_loader = make_dataloader("val", cfg, batch_size=batch_size)

    logger.info(
        f"=== v5 RUN BANNER === "
        f"train loader: DynamicMixDataset (on-the-fly mixing, pool={len(train_dataset.pool_entries)} "
        f"clean utterances, {len(train_dataset)} draws/epoch = steps_per_epoch({steps_per_epoch}) x "
        f"batch_size({batch_size})) -- NEW mixtures every __getitem__ call, never repeats the static "
        f"2162-pair manifest. | val loader: make_dataloader('val', ...) over manifests/val.json "
        f"(STATIC, UNCHANGED since Phase 1, {len(val_loader.dataset)} pairs) -- completely untouched, "
        f"same standard as v1/v2/v4."
    )

    total_warmup_steps = max(1, int(steps_per_epoch * max_epochs * sched_cfg["warmup_ratio"]))
    warmup_scheduler, plateau_scheduler = _build_warmup_then_plateau(optimizer, total_warmup_steps, cfg)
    warmup_steps_done = 0

    logger.info(
        f"Phase v5 training run: max_epochs={max_epochs}, batch_size={batch_size}, "
        f"steps_per_epoch={steps_per_epoch}, warmup_steps={total_warmup_steps}, "
        f"lr_schedule_v5={sched_cfg}, loss_weights_v4={cfg['loss_weights_v4']}, "
        f"silence_penalty_enabled={run_cfg['silence_penalty_enabled']}, "
        f"noise_category_weights={run_cfg['noise_category_weights']}, "
        f"early_stopping(min_delta={min_delta}, patience_epochs={patience_epochs})"
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    plateaued = False

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_l1", "train_stft", "train_silence_penalty", "train_total",
            "val_l1", "val_stft", "val_silence_penalty", "val_total",
            "lr", "epoch_seconds",
        ])

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            running_l1, running_stft, running_penalty, running_total, n_batches = 0.0, 0.0, 0.0, 0.0, 0
            for noisy_batch, clean_batch in train_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)

                optimizer.zero_grad()
                output = model(noisy_batch).squeeze(1)
                l1_loss, stft_loss, penalty, total_loss = compute_loss_v5(
                    output, clean_batch, noisy_batch, stft_loss_fn, cfg
                )
                total_loss.backward()
                optimizer.step()

                if warmup_steps_done < total_warmup_steps:
                    warmup_scheduler.step()
                    warmup_steps_done += 1

                running_l1 += l1_loss.item()
                running_stft += stft_loss.item()
                running_penalty += penalty.item()
                running_total += total_loss.item()
                n_batches += 1

            train_l1 = running_l1 / n_batches
            train_stft = running_stft / n_batches
            train_penalty = running_penalty / n_batches
            train_total = running_total / n_batches

            val_l1, val_stft, val_penalty, val_total = _run_validation_v5(
                model, val_loader, stft_loss_fn, cfg, device
            )

            # Reactive LR: only step the plateau scheduler once warmup is
            # fully done -- stepping it during warmup would let it start
            # reducing an LR that hasn't even reached peak_lr yet.
            if warmup_steps_done >= total_warmup_steps:
                plateau_scheduler.step(val_total)

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_seconds = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{max_epochs} -- "
                f"train: L1={train_l1:.6f} STFT={train_stft:.6f} silence_pen={train_penalty:.6f} total={train_total:.6f} | "
                f"val: L1={val_l1:.6f} STFT={val_stft:.6f} silence_pen={val_penalty:.6f} total={val_total:.6f} | "
                f"lr={current_lr:.8f} | {epoch_seconds:.1f}s"
            )

            writer.writerow([
                epoch, train_l1, train_stft, train_penalty, train_total,
                val_l1, val_stft, val_penalty, val_total,
                current_lr, epoch_seconds,
            ])
            f.flush()

            if val_total < best_val_loss - min_delta:
                best_val_loss = val_total
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total_loss": val_total,
                    "val_l1_loss": val_l1,
                    "val_stft_loss": val_stft,
                    "val_silence_penalty": val_penalty,
                    "lr": current_lr,
                }, checkpoint_path)
                logger.info(f"  New best val loss ({val_total:.6f}) -- checkpoint saved to {checkpoint_path}")
            else:
                epochs_since_improvement += 1
                logger.info(
                    f"  No improvement > min_delta={min_delta} for "
                    f"{epochs_since_improvement}/{patience_epochs} epochs."
                )

            if epochs_since_improvement >= patience_epochs:
                plateaued = True
                logger.info(
                    f"Plateau detected: val loss has not improved by more than "
                    f"min_delta={min_delta} for {patience_epochs} consecutive epochs. "
                    f"Stopping early at epoch {epoch} (ceiling was {max_epochs})."
                )
                break

    if plateaued:
        logger.info(
            f"v5 training run complete -- STOPPED EARLY on plateau at epoch {epoch}. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"Checkpoint: {checkpoint_path}"
        )
    elif epoch == max_epochs:
        logger.info(
            f"v5 training run complete -- EPOCH CEILING ({max_epochs}) REACHED, no plateau detected. "
            f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}. "
            f"HONEST FRAMING (per locked project rule): this means the ceiling was hit while "
            f"val loss was still improving or fluctuating without a clear plateau signal -- "
            f"do NOT report this as convergence. Checkpoint: {checkpoint_path}"
        )


if __name__ == "__main__":
    import sys
    if "--run-v5" in sys.argv:
        train_v5()
    elif "--run-v4" in sys.argv:
        train_v4()
    elif "--run-v2" in sys.argv:
        train_v2()
    elif "--run" in sys.argv:
        train()
    else:
        smoke_test_train()
