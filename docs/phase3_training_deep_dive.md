# Phase 3 Training Deep Dive — From Transfer Learning to v5

This document explains, from first principles up to expert depth, how fine-tuning
actually worked in this project — not as a generic ML tutorial, but grounded
entirely in this repo's real files, real config values, and real numbers from
the development log. Every number here is either quoted directly from
those files or from `configs/finetune.yaml` / `src/model/finetune.py`. Nothing
is invented or approximated for narrative convenience.

---

## 1. Why fine-tune dns48 instead of training from scratch?

### 🧠 The intuition

Imagine you're teaching someone who has already spent years learning to
distinguish "voices" from "background noise" in general — they've heard
thousands of hours of ordinary noisy audio (traffic, music, chatter) and
gotten quite good at picking speech out of it. Now you want them to get
specifically good at one narrow, unusual case: picking a soldier's voice out
of gunfire and battlefield noise. You have two choices:

1. Find a brand-new person who has never heard *any* audio before, and teach
   them everything — general speech, general noise-filtering, *and* the
   defence-specific case — from absolute zero.
2. Take the person who already knows the general skill, and give them focused
   additional practice on just the defence-specific noise types.

Option 2 is obviously faster and more reliable — the person already has all
the "general auditory scene analysis" wiring in place; you're not rebuilding
that from nothing, you're specializing it. This is **transfer learning**, and
the specific act of continuing training on a pretrained model for a narrower
task is called **fine-tuning**.

`dns48` (Facebook Research's Denoiser, dns48 checkpoint) is the "already
experienced" model: 18,867,937 parameters (the exact locked count from
`configs/finetune.yaml['model']['expected']`, verified in Phase 3a — not the
rounded "~18.9M" figure some documents use casually), pretrained on general
noisy speech. Project rule 6 makes this a hard constraint for the project:
**no training from scratch, ever** — only transfer learning on top of dns48.

### 📐 The math: why 3e-5 and not 3e-4?

When you fine-tune, you are not resetting the 18.9M parameters — you are
nudging them, in small steps, from wherever dns48's pretraining left them.
The "nudge size" per step is governed by the **learning rate (LR)**, the
scalar that multiplies the gradient in a weight update:

```
θ_new = θ_old − η · ∇L(θ_old)
```

Where:
- `θ` is a single parameter (one of the 18.9M numbers).
- `η` (eta) is the learning rate.
- `∇L(θ_old)` is the gradient of the loss with respect to that parameter —
  "which direction, and how strongly, does moving this weight reduce the
  loss?"

If `η` is too large, each step moves `θ` a long way from where pretraining
put it. Early in fine-tuning, gradients computed on a small, narrow dataset
(this project's 2,162 train pairs, vastly smaller and less diverse than
whatever dns48 originally trained on) can be noisy and unrepresentative. A
large step taken on a noisy gradient signal can wreck weights that encoded
genuinely useful, general "voice vs. noise" structure — the model gets
worse at the general task while barely improving at the specific one. This
failure mode has a name: **catastrophic forgetting**.

Mathematically, catastrophic forgetting is a statement about **gradient
magnitude vs. distance from the pretrained weights**. Let `θ₀` be dns48's
pretrained weights and `θ_t` the weights after `t` fine-tuning steps. The
drift is:

```
‖θ_t − θ₀‖ ≈ η · Σ_{i=1}^{t} ‖∇L(θ_i)‖   (informally, ignoring direction cancellation)
```

A large `η` inflates this drift per step. If the drift grows faster than the
new, narrow dataset can "teach" useful specialization, the model moves away
from its good general prior faster than it learns anything to replace it
with — net result: worse at everything. The standard fine-tuning heuristic,
used directly in this project, is to set `η` to roughly **1/10th of a
typical from-scratch learning rate** for the task family — this project used
`training.learning_rate: 3.0e-5` (`configs/finetune.yaml`), against a typical
from-scratch Adam LR for speech enhancement of ~3e-4. Comment in the config
states this explicitly: "1/10th of a typical from-scratch rate, standard
fine-tuning heuristic to avoid catastrophically forgetting dns48's pretrained
weights." A small `η` keeps `‖θ_t − θ₀‖` small per step, so the model only
drifts as far as the new data actually justifies, rather than lurching.

### 🔗 The connection

This same drift-control logic reappears later in Phase 3c/4's LR *scheduling*
decisions (Section 4) — v2 raised the *peak* LR to 5x v1's flat rate but only
reached it gradually via warmup, for exactly this reason: a pretrained model
tolerates a higher LR once it has already taken a few small, safe steps and
the optimizer's momentum estimates have stabilized, but not from step zero.

### ❓ Check yourself

If you fine-tuned dns48 with `η = 3e-3` (100x this project's actual value)
for even a handful of steps, what would you expect to see happen to the
model's *general* denoising ability (on noise types close to what dns48 was
originally trained on), even if the *new* gunshot-specific loss looked like
it was dropping fast?

---

## 2. The training loop mechanics

### 🧠 The intuition

Every single training step in this project (and in almost all of supervised
deep learning) is the same four-beat cycle, repeated over and over:

1. **Guess.** Show the model a noisy clip. It produces its best attempt at
   the clean speech (the **forward pass**).
2. **Grade the guess.** Compare the guess to the real clean speech using a
   loss function — a single number where lower means "closer to correct."
3. **Figure out who's to blame.** Trace back through every one of the 18.9M
   parameters and compute exactly how much each one contributed to the
   error (the **backward pass**, via backpropagation).
4. **Adjust.** Nudge each parameter a small amount in the direction that
   would have reduced the error (the **optimizer step**).

Repeat this thousands of times and the guesses get progressively better.

### 📐 The math

**Forward pass:** `ŷ = f(x; θ)` — the model `f`, parameterized by the current
weights `θ`, maps noisy input `x` to an enhanced output `ŷ`. In this project,
concretely: `Demucs.forward()` takes a noisy waveform batch and returns an
enhanced waveform batch of identical shape (Phase 3a's verified architecture
guarantee — `x[..., :length]` crops the internally-padded output back to
exactly the input length).

**Loss computation:** `L = ℓ(ŷ, y)` compares the guess `ŷ` to the true clean
signal `y`. Section 3 below derives this project's exact `ℓ` in full.

**Backward pass (backpropagation):** the chain rule of calculus, applied
layer by layer, backward through the network. If `L` depends on `ŷ`, and `ŷ`
depends on some intermediate activation `h`, and `h` depends on a weight
`w`, then:

```
∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂h) · (∂h/∂w)
```

PyTorch's `backward()` call computes this automatically for all 18.9M
weights in one pass, by walking the computation graph built during the
forward pass in reverse. Each weight ends up with a `.grad` attribute — the
value of `∂L/∂w` for that specific weight, given this specific batch.

**Optimizer step:** `θ_new = θ_old − η · update(∇L)`, where for Adam
(this project's optimizer, `training.learning_rate: 3.0e-5`) `update()` also
folds in running estimates of the gradient's mean and variance (per-weight
adaptive step sizing) — not literally the vanilla SGD formula from Section 1,
but the same "move opposite the gradient, scaled by η" principle underneath.

### 💻 The code — this project's real cycle

From `src/model/finetune.py`'s `compute_loss()` and the training loop it
feeds (paraphrased/annotated, not altered from the real function signatures
already verified in Phase 3b/3c):

```python
# --- 1. forward pass ---
model.train()                          # switch from eval() (used at inference) to train()
output = model(noisy_batch)            # (B, 1, samples) -- Demucs adds a channel dim
output_2d = output.squeeze(1)          # (B, samples) -- MultiResolutionSTFTLoss expects 2D (B, T)

# --- 2. loss computation ---
l1_loss, stft_loss, total_loss = compute_loss(output_2d, clean_batch, stft_loss_fn, cfg)
# compute_loss() internally does:
#   l1_loss = F.l1_loss(output, clean)
#   sc_loss, mag_loss = stft_loss_fn(output, clean)
#   stft_loss = sc_loss + mag_loss
#   total_loss = weights["l1"] * l1_loss + weights["stft"] * stft_loss

# --- 3. backward pass ---
optimizer.zero_grad()                  # clear old .grad values from the previous step
total_loss.backward()                  # populate .grad on all 18.9M parameters via the chain rule

# --- 4. optimizer step ---
optimizer.step()                       # Adam updates every weight using its .grad
```

The Phase 3b smoke test ran exactly this cycle once, on real data, and
printed real numbers (not synthetic sine waves): **L1 loss 0.022368, STFT
loss 0.175597, total 0.197965**. A gradient spot-check on 5 real parameters
after `backward()` — `encoder.0.0.weight` (0.00453883), `encoder.0.0.bias`
(0.00641049), `encoder.0.2.weight` (0.00068696), `encoder.0.2.bias`
(0.00381913), `encoder.1.0.weight` (0.00037252) — confirmed every one was
non-zero, meaning the loss was genuinely connected all the way back to real
model weights, not silently detached from the computation graph (a common
and hard-to-notice failure mode where `.backward()` runs without error but
produces all-zero or `None` gradients because of a broken tensor operation
somewhere in the forward pass).

### 🔗 The connection

This exact four-beat cycle is what runs, unchanged in its fundamentals,
across all 12 epochs of v1's real training run, all 42 of v5's, and every
version in between — what changes between versions (Section 4/5) is never
this mechanic, only the *inputs to* and *scheduling around* it: which data
gets sampled, what the learning rate is at a given step, and which loss
terms are weighted how.

### ❓ Check yourself

If `optimizer.zero_grad()` were accidentally omitted from the loop (so
gradients accumulate across steps instead of resetting), what would you
expect to observe in the training loss curve as training progressed, and
why?

---

## 3. The loss function, derived in full

### 🧠 The intuition

The model needs a way to be told "how wrong" its guess was, in a form it can
use to improve. This project uses **two different, complementary ways** of
measuring wrongness, added together:

1. **L1 waveform loss** — "how different are the two audio signals, sample
   by sample, in the raw time domain?" Like laying two audio waveforms on
   top of each other on a graph and measuring the total vertical gap between
   them at every single point in time.
2. **Multi-resolution STFT loss** — "how different do the two signals sound
   in terms of their frequency content, at multiple time-scales?" Like
   comparing two songs not sample-by-sample but by their sheet music —
   which notes/frequencies were present, and how loud, at each moment —
   checked at several different zoom levels simultaneously (some passes look
   at short, precise time slices; others look at longer, blurrier slices
   that see frequency more precisely).

Why both? L1 alone is a purely time-domain measure and can be insensitive to
perceptually important spectral structure (two waveforms can have a similar
sample-by-sample L1 distance while sounding very differently "textured" —
e.g. one crisp, one muffled). STFT loss alone, on the other hand, can miss
fine time-domain alignment. Combining both pushes the model to match on
*both* axes at once — this project locked this choice explicitly in
the development log: "**L1 + multi-resolution STFT loss**."

### 📐 The math — L1 loss

```
L1(ŷ, y) = mean( |ŷ_i − y_i| )   for all samples i
```

Where `ŷ_i` is the model's predicted sample at time index `i`, and `y_i` is
the true clean sample at the same index. It's simply the mean absolute
difference across every sample in the batch.

**Why L1 (mean absolute error) instead of L2 (mean squared error) for
audio?** This is a gradient-behavior question. Consider a single sample's
error `e = ŷ_i − y_i`:

- L2's loss contribution is `e²`, and its gradient with respect to `e` is
  `2e` — the gradient grows **linearly** with the size of the error. A
  single sample with a very large error (an outlier — e.g. a transient
  click, or a moment the model got badly wrong) produces a disproportionately
  huge gradient, which can dominate the update and destabilize training
  around that one outlier at the expense of everything else.
- L1's loss contribution is `|e|`, and its gradient is `sign(e)` — a
  constant magnitude of 1 (in the direction of the error), **regardless of
  how large the error is**. A single bad outlier sample contributes exactly
  as much gradient "pull" as a slightly-wrong sample. This makes L1 far more
  robust to outliers, which matters directly for this project: audio
  waveforms with transient noise (gunshots specifically — a sudden,
  high-amplitude burst) are exactly the kind of outlier-heavy signal where
  L2's outsized gradient response to a few loud samples would risk
  destabilizing training around the transient rather than learning a
  balanced fit across the whole clip.

**Extremes check:** if `ŷ = y` exactly (perfect reconstruction), `L1 = 0` —
no gradient, nothing more to learn from that sample. If `ŷ` is off by a
constant amount `c` for every sample, `L1 = |c|` — grows linearly, not
quadratically, with the size of the mistake.

### 📐 The math — multi-resolution STFT loss

First, the **STFT** (Short-Time Fourier Transform) itself: instead of
looking at the whole waveform's frequency content at once, it slides a
window across the signal and computes the frequency content (via FFT)
within each windowed chunk, producing a 2D **spectrogram** — one axis time,
one axis frequency, values are magnitude. In `src/vendor/denoiser_patched/
stft_loss.py`'s `stft()` function:

```python
x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
magnitude = sqrt(clamp(real² + imag², min=1e-7))
```

`real` and `imag` are the real and imaginary parts of the complex-valued
STFT output at each time-frequency bin; magnitude is their Euclidean norm
(`√(real² + imag²)`) — literally the length of the complex vector at that
bin, i.e. "how much energy is present" at that time/frequency, discarding
phase. The `clamp(..., min=1e-7)` exists purely to avoid `sqrt(0)` or
negative-due-to-floating-point-error producing `NaN`/`Inf` downstream (a
numerical safety floor, not a modeling choice).

Given magnitude spectrograms `x_mag` (predicted) and `y_mag` (true clean),
this project's patched loss computes **two** distinct terms per resolution:

**1. Spectral convergence loss** (`SpectralConvergengeLoss`):

```
L_sc(x_mag, y_mag) = ‖y_mag − x_mag‖_F / ‖y_mag‖_F
```

Where `‖·‖_F` is the Frobenius norm (treat the whole 2D spectrogram as one
long vector and take its Euclidean length). This is a **relative** error
measure — the numerator is the magnitude of the difference, the denominator
normalizes by the magnitude of the true signal itself. This makes the loss
scale-invariant: a loud clip and a quiet clip with proportionally the same
spectral mismatch get the same `L_sc` value, rather than the loud clip
automatically producing a larger raw-magnitude loss just because it's loud.

**2. Log STFT magnitude loss** (`LogSTFTMagnitudeLoss`):

```
L_mag(x_mag, y_mag) = mean( |log(y_mag) − log(x_mag)| )
```

Taking the log before comparing matters because human hearing (and speech
energy distributions generally) is much more sensitive to relative,
multiplicative differences than absolute ones — the difference between
"very quiet" and "quiet" energy levels matters perceptually far more than
the same raw magnitude gap would at "loud" vs "very loud" levels. Working in
log-magnitude space makes the loss weight small-energy (quiet) regions
proportionally rather than being swamped entirely by loud regions.

**3. Multi-resolution combination.** A single STFT resolution (one choice of
FFT size / hop / window length) is a specific time-frequency tradeoff — a
short window gives good time resolution but poor frequency resolution, and
vice versa (an unavoidable consequence of the time-frequency uncertainty
principle). This project's patched loss (inherited unmodified from the
original `denoiser` implementation, only its `torch.stft()` call itself was
patched for compatibility) uses **three resolutions simultaneously**:

```
fft_sizes  = [1024, 2048, 512]
hop_sizes  = [120, 240, 50]
win_lengths= [600, 1200, 240]
```

For each of the 3 resolutions `k`, compute `(L_sc^(k), L_mag^(k))`, then sum
and average across the 3 resolutions:

```
L_STFT = (1/3) · Σ_k [ factor_sc · L_sc^(k) + factor_mag · L_mag^(k) ]
```

with `factor_sc = factor_mag = 0.1` (the loss module's own internal scaling
constants, unrelated to this project's `training.loss_weights` config — the
two scales operate at different levels: `factor_sc`/`factor_mag` balance the
two STFT sub-terms against each other *inside* the STFT loss, while
`training.loss_weights.{l1,stft}` balance the combined L1 loss against the
combined STFT loss at the top level). Summing across 3 window sizes forces
the model to match spectral structure at short, medium, and long time-scales
simultaneously — catching both fine transient detail (short windows, good
for something like a gunshot's sharp attack) and broader tonal/harmonic
structure (long windows, better frequency resolution for sustained speech
formants).

**Final combined loss, as implemented in `compute_loss()`:**

```
L_total = w_l1 · L1(ŷ, y) + w_stft · [ Σ_k (L_sc^(k) + L_mag^(k)) ]
```

v1 used `w_l1 = w_stft = 1.0` (implicit 1:1, `training.loss_weights`). v2
tried `w_stft = 2.0` (Section 5); v4/v5 reverted to 1:1 after v2's reweight
showed no measurable benefit.

### 💻 The code

```python
def compute_loss(output, clean, stft_loss_fn, cfg):
    weights = cfg["training"]["loss_weights"]          # {"l1": 1.0, "stft": 1.0} for v1
    l1_loss = F.l1_loss(output, clean)                  # mean(|output - clean|)
    sc_loss, mag_loss = stft_loss_fn(output, clean)     # summed across 3 resolutions internally
    stft_loss = sc_loss + mag_loss
    total_loss = weights["l1"] * l1_loss + weights["stft"] * stft_loss
    return l1_loss, stft_loss, total_loss
```

### 🔗 The connection

L1's outlier-robustness argument directly foreshadows the v1→v2 diagnosis in
Section 5: the model's dominant failure mode turned out to be
**over-suppression** (erasing signal), not blowing up on transients — i.e.
the L1/STFT loss combination was already doing its outlier-robustness job
correctly; the bottleneck lay elsewhere (Section 5's data-exhaustion
diagnosis), which is exactly why v2's loss-reweighting lever (STFT weight
1.0→2.0) produced no measurable change.

### ❓ Check yourself

If you saw `L_sc` staying stubbornly high while `L_mag` dropped steadily
over training, what would that combination suggest about *what kind* of
spectral mismatch remained — an amplitude-scale mismatch, or a genuine
structural mismatch in *where* energy is located across time-frequency
bins? (Hint: think about what dividing by `‖y_mag‖_F` does and doesn't
normalize away.)

---

## 4. Learning rate scheduling: warmup+cosine (v1–v4) vs. ReduceLROnPlateau (v5)

### 🧠 The intuition

A fixed learning rate is a blunt instrument: early in training, when the
model is far from a good solution, you might want a *bigger* step to make
fast progress; late in training, when you're close to a good solution, a big
step risks overshooting and bouncing around instead of settling. A **schedule**
changes the learning rate over time to get the best of both.

**Warmup + cosine decay** (v1's flat rate replaced starting at v2) is a
*fixed, pre-planned* schedule: ramp the LR up gently at the start (warmup),
then smoothly ride it back down along a cosine curve to a low floor by a
*predetermined* end point.

**ReduceLROnPlateau** (v5) is *reactive* instead of pre-planned: keep the LR
constant as long as validation loss keeps improving, and only cut it (by a
fixed factor) once validation loss stops improving for a set number of
epochs — repeatedly, as many times as needed, with no fixed end-point baked
in ahead of time.

### 📐 The math — warmup + cosine (v1→v4)

**Warmup phase** (first `warmup_ratio` fraction of total training steps):
linear ramp from 0 to `peak_lr`:

```
η(t) = peak_lr · (t / T_warmup)      for t ≤ T_warmup
```

where `t` is the current step and `T_warmup = warmup_ratio · total_steps`.

**Cosine decay phase** (remaining steps): smooth decay from `peak_lr` down
to `min_lr` following a cosine curve:

```
η(t) = min_lr + 0.5 · (peak_lr − min_lr) · [1 + cos(π · (t − T_warmup) / (T_total − T_warmup))]
```

At `t = T_warmup` (right after warmup ends), the cosine term is `cos(0) = 1`,
so `η = min_lr + (peak_lr − min_lr) = peak_lr` — continuous with the end of
warmup, no jump. At `t = T_total` (end of training), the cosine term is
`cos(π) = −1`, so `η = min_lr + 0 = min_lr` — lands exactly at the floor.
The cosine shape (versus, say, linear decay) spends more time near both the
peak and the floor and moves fastest through the middle — a smooth,
accelerating-then-decelerating taper.

This project's actual v2 values (`configs/finetune.yaml`,
`training_run.lr_schedule`): `peak_lr: 1.5e-4` (5x v1's flat `3e-5`, per the
config's own comment — justified because after even a few warmup steps on
pretrained weights, gradients are more stable/informative than at step
zero, so a higher rate is safer than it would've been from a cold start),
`min_lr: 3.0e-5` (kept at v1's original flat rate as a floor, never driven
to zero — a genuine floor learning rate is retained throughout so the model
never fully stops adapting even at the tail of training), `warmup_ratio:
0.05` (5% of steps spent ramping up — short and deliberate, per config
comment: long enough to not jump straight to `peak_lr` on pretrained
weights, short enough not to waste the training budget).

**Why cosine decay needs a fixed end-point:** the formula above literally
requires knowing `T_total` in advance (it's baked into the denominator) — the
curve is *shaped around reaching `min_lr` exactly at the last step of a
committed `max_epochs`*. This becomes the exact reason v5 abandoned it
(below).

### 📐 The math — ReduceLROnPlateau (v5)

No fixed decay curve. Instead, monitor validation loss each epoch and apply
a rule:

```
if val_loss has not improved by > threshold for `plateau_patience` consecutive epochs:
    η ← η × plateau_factor
    (repeat every time patience is exhausted again, down to a floor)
η ← max(η × plateau_factor, min_lr)   # never go below the floor
```

This project's real v5 values (`configs/finetune.yaml`,
`training_run_v5.lr_schedule`):
- **Warmup:** same shape as v2 — `peak_lr: 1.5e-4`, `warmup_ratio: 0.05` (per
  config comment: "keep ~same warmup as v2", reused directly rather than
  duplicated into a new key).
- **`plateau_factor: 0.5`** — each time a plateau is detected, halve the LR.
  The config's own comment calls this "a standard, moderate
  ReduceLROnPlateau default (halving is gentle enough...)".
- **`plateau_patience: 5`** — wait 5 epochs of no meaningful val-loss
  improvement before cutting; shorter than the overall early-stopping
  patience of 15 by design, so the LR gets a chance to react and potentially
  *rescue* a plateau before early stopping gives up on the run entirely.
- **`min_lr: 1.0e-5`** — the floor below which the LR is never cut further;
  set below v2/v4's `min_lr: 3.0e-5` floor since v5 has no fixed decay
  curve that needs a matching endpoint — it can keep cutting as long as
  plateaus keep recurring, so a lower absolute floor is safe.

**Why the switch happened, mathematically:** v5 raised `max_epochs` to 120 —
described in the config as "a loose ceiling, not a tight budget a cosine
curve needs to reach zero by." A cosine schedule *pre-commits* its entire
decay shape to hitting `min_lr` exactly at whatever `max_epochs` value you
chose ahead of time. If the run actually plateaus and early-stops well
before `max_epochs` (as v1 did at epoch 12 of a 40-epoch ceiling, and as
most versions did well under their ceilings), the cosine curve is still only
partway through its planned decay at the moment training actually stops —
it never reaches its intended floor, and the LR trajectory that was
*supposed* to happen during those actually-unneeded remaining epochs never
gets a chance to matter. `ReduceLROnPlateau` sidesteps this mismatch
entirely: it responds to what training is *actually doing*, epoch by epoch,
regardless of how many epochs are still nominally available.

### 🔗 The connection

This is a direct instance of a general principle worth carrying forward:
**a schedule that has to guess the future (cosine, requiring `max_epochs`
upfront) is strictly less robust than one that reacts to what's actually
observed (plateau-triggered)**, whenever your stopping point is itself
uncertain — which describes almost every real training run, since you
rarely know in advance exactly which epoch will turn out to be "enough."

### ❓ Check yourself

Suppose a training run's validation loss plateaus at epoch 20 out of a
cosine schedule planned for `max_epochs=120`. At epoch 20, is the learning
rate still much higher than `min_lr`, or already close to it? What does that
imply about whether the model got the *chance* to take smaller, more
careful steps around the point where it was actually struggling?

---

## 5. The full diagnostic journey: v1 through v5

This is the centerpiece of Phase 3 — not a list of hyperparameter tweaks,
but a real, methodologically disciplined debugging investigation, carried
out over four fine-tuning iterations. The lesson here is as much about *how*
to diagnose a stuck model as it is about *what* eventually fixed it.

### v1 → v2: diagnose before touching anything

v1 (the first real training run, Phase 3c) plateau-stopped at epoch 12 of
its 40-epoch ceiling, best checkpoint at epoch 7 (`val_total_loss=0.143079`).
Real evaluation numbers (test split, n=299): **SNR 8.671dB, STOI 0.6620,
PESQ 1.7882** — a large, real improvement over the spectral-subtraction
baseline (SNR 1.84dB, STOI 0.592, PESQ 1.292), but nowhere near the PS
targets (SNR>15dB, STOI>0.85, PESQ>2.5).

Rather than guessing at hyperparameters, `src/eval/diagnose_failures.py` was
run against v1's own worst-scoring test pairs: **18 out of 18** of the worst
pairs (by PESQ/STOI) were labeled "likely over-suppression" by an
STFT-based residual-noise-estimate vs. speech-distortion-estimate heuristic
— i.e. the model was *erasing signal it shouldn't have*, not *failing to
remove noise it should have*. This is a meaningfully different failure mode
than "the model just isn't powerful/trained enough" — it points at specific,
targetable levers: the loss function's relative weighting of
speech-preservation vs. noise-removal, and how much exposure the model gets
to the hardest (most easily over-suppressed) examples.

**v2's three concurrent levers, directly motivated by that diagnosis:**
1. LR schedule: v1's flat `3e-5` → warmup+cosine, peak `1.5e-4` (Section 4).
2. STFT loss weight 1.0 → 2.0 (targets STOI/PESQ, which are more spectrally
   sensitive metrics, more directly than raw L1).
3. 3x oversampling of hard (low-SNR, <5dB) mixtures via `WeightedRandomSampler`
   (train-only; val/test kept at natural distribution).

**Real result (n=299, overall):** SNR 8.564dB, STOI 0.6656, PESQ 1.7753 —
**near-identical to v1** (8.671/0.6620/1.7882) despite three simultaneous
changes to the training recipe. This is itself an important, real finding:
none of these three levers moved the needle. The bottleneck was *not*
optimizer/loss configuration in the way v2 hypothesized.

A follow-up, `diagnose_data_gaps.py` and `v1_vs_v2_worst_pairs_comparison.csv`,
ruled out one more candidate explanation: cross-tabulating failure rates by
noise category confirmed that gunshot's poor performance was **not**
gunshot-specific data scarcity — every category showed similarly poor
performance at low SNR. The gap was a *general* low-SNR capability
shortfall, not a category-specific one. This explicitly ruled out "just add
more gunshot data" as the next lever, before that lever was ever tried —
saving a wasted iteration.

### v4: curriculum learning + silence-collapse penalty, and why it also regressed

Since v2's three simultaneous changes produced no measurable improvement, v4
reverted the two *unproven* levers (STFT reweight back to 1:1, static 3x
oversampling replaced rather than just removed) and kept only the one v2
change *not implicated by any diagnosis* (the warmup+cosine LR schedule
itself). Two new, diagnosis-targeted levers were added instead:

1. **Curriculum learning:** an epoch-aware sampling weight, linearly ramped
   over the *first 40% of epochs* then held constant: low-SNR pairs (<5dB)
   ramp up to 3x sampling weight, gunshot-category pairs ramp up to 2x
   weight, **multiplicative when both apply** (i.e. exactly the diagnosed
   worst cell: low-SNR gunshot pairs get up to 6x weight during the ramp).
   Critically, this design ends training at the *true, unskewed*
   distribution rather than a permanently harder one — the idea being to
   front-load difficulty as a kind of targeted practice, then let the model
   consolidate on the real-world distribution it will actually be evaluated
   against.
2. **Silence-collapse penalty:** an additional loss term (weight 0.05) that
   directly penalizes the enhanced output going near-silent in frames where
   the *noisy input itself* had substantial energy (RMS > 0.01) and the
   output/input RMS ratio dropped below 0.1. This targets the diagnosed
   over-suppression failure mode head-on, and deliberately keys off the
   *input's own energy* rather than a blanket minimum-energy floor compared
   against clean speech — a floor compared against clean would incorrectly
   penalize genuinely-correct silence (e.g. a pause between words in the
   true clean speech), whereas keying off "was there energy in what the
   model actually received" avoids punishing correct suppression.

Because the curriculum's back half faces a genuinely different (harder,
front-loaded) training distribution than the front half, the epoch ceiling
was raised to 100 (patience 20) — a real second learning phase, post-ramp,
was expected, not just noise around an already-found optimum.

**What actually happened:** the run plateau-stopped at epoch 31, but epoch
11 was the *last new-best checkpoint* — meaning the curriculum's back-half
(post-ramp, harder-to-easier transition) never produced a new best model in
the remaining 20 epochs it ran for. **Real result (n=299, overall): SNR
8.586dB, STOI 0.6631, PESQ 1.7718** — again near-identical to v1/v2. The
signature here — best checkpoint arriving early, then 20 further epochs of
no improvement despite an explicitly harder-then-real curriculum designed to
give the model more to learn from — matches classic **overfitting to a
fixed, static dataset**: the ramp changed *which* of the same 2,162 fixed
pairs got sampled more often, but it never introduced genuinely new
information the model hadn't already seen by epoch 11. Once the model had
extracted what it could from the fixed pool at the sampling weights it saw
early on, continuing to reshuffle *emphasis* over the *same* underlying
2,162 pairs had nothing further to teach it.

### The over-suppression diagnosis: methodology in detail

It's worth stating explicitly *how* `diagnose_failures.py`'s heuristic
worked, since "likely over-suppression" isn't a metric PESQ/STOI report
directly — it had to be derived:

1. **Automated diagnostic method:** for each of the worst-scoring test
   pairs, compute an STFT-based *residual-noise estimate* (how much
   noise-like energy remains in the enhanced output relative to the clean
   reference) versus a *speech-distortion estimate* (how much speech-like
   energy present in the clean reference is *missing* from the enhanced
   output). A pair where speech-distortion dominates over residual-noise is
   labeled "likely over-suppression" — the model removed too much, not too
   little.
2. **Manual listening confirmation:** the automated labels were not taken
   purely on faith — a manual listening pass over flagged pairs confirmed
   the labels matched what a human could actually hear (audibly missing
   speech content, not audible leftover noise).
3. **The v1-vs-v2-on-same-pairs isolation test:** to rule out the
   possibility that v2's specific changes (LR schedule, STFT reweight,
   oversampling) had somehow *caused* or *worsened* over-suppression rather
   than v1's original training recipe being the root cause, the same worst
   pairs were compared v1-vs-v2 directly (`v1_vs_v2_worst_pairs_comparison.csv`).
   Because v2 showed the *same* failure signature (near-identical metrics,
   same category of worst pairs) despite genuinely different training
   levers, this isolated the failure mode as something *upstream of* the
   specific v1/v2 training configuration — i.e. not an artifact introduced
   by any one version's specific choices.
4. **Data-gap cross-tabulation:** as described above, this ruled out
   category-specific data scarcity — the gap cross-tabulated as a general
   low-SNR problem, present in gunshot, stationary, *and* general noise
   categories alike, pinpointing the worst *cell* specifically as
   (gunshot × −5 to 0dB SNR) without attributing the general pattern to
   gunshot data volume.

This four-step discipline — automated heuristic, manual confirmation,
cross-version isolation, cross-tabulated data analysis — is what separates a
genuine diagnosis from a guess dressed up as one.

### The root-cause pivot: static dataset exhaustion

By the time v4 also landed at v1's plateau, a pattern became undeniable:
**every version, regardless of optimizer configuration, loss weighting, or
sampling curriculum, converged to the same performance ceiling.** The common
factor across v1, v2, and v4 was not any of the things that had been varied
— it was the one thing that had *not* been varied: **the training data
itself.** All three versions trained on the identical, fixed 2,162-pair
manifest (`manifests/train.json`, produced once in Phase 1) — the exact same
2,162 noisy/clean pairs, seen repeatedly, epoch after epoch, regardless of
how the *sampling weights* over that fixed pool were reshuffled.

This connects directly to a basic capacity argument: an 18.9M-parameter
model, given only 2,162 fixed examples to see repeatedly, will eventually
extract essentially everything learnable from that finite, static set — at
that point, further epochs (or reweighted epochs) are not providing new
information, only more repetitions of information already seen, which is
precisely the setup for overfitting/memorization rather than continued
generalization improvement. No amount of optimizer path, loss reweighting,
or sampling curriculum can manufacture information that isn't present in a
fixed 2,162-pair pool. This reasoning is what explains, in one unified way,
why v1, v2, and v4 all independently converged to the same plateau despite
substantially different training recipes: **the bottleneck was never the
optimization path — it was the fixed size of the dataset itself.**

### v5's fix: dynamic mixing

v5's central change was to stop training on a *static* manifest entirely.
Instead of a fixed 2,162-pair pool sampled (with various weightings) epoch
after epoch, v5 draws **fresh noise/speech mixtures on-the-fly, every step**
— `src/data/dynamic_dataset.py` / `src/data/build_dynamic_pool.py`, with
`manifests/dynamic_train_pool.json` describing the expanded pool.
Critically, the speech pool itself was also expanded to **134 speakers /
44.71 hours** (compared to the original 32-speaker, 2,162-utterance static
train split) — meaning v5's model genuinely sees new speaker/noise
combinations it has never encountered before, in every epoch, rather than
re-sampling the same fixed set of pairings with different emphasis.

Design details that kept the comparison fair against v1–v4:
- `noise_category_weights` were set explicitly uniform (gunshot=stationary=
  general=1.0) to match the natural ~1/3-per-category balance v1–v4's random
  choice produced — v5's improvement, if any, would need to come from
  genuine data diversity, not from a stealth rebalancing of category
  exposure.
- `steps_per_epoch=270` was deliberately kept identical to v1/v2/v4's
  static-manifest epoch size (`2162 // 8`), so wall-clock and epoch-count
  comparisons across all four versions remain apples-to-apples — v5 isn't
  "winning" by simply training on more total steps.
- The dynamic dataset was seeded once at construction (making the run
  reproducible) but *not* re-seeded per epoch, so successive epochs within
  one run genuinely see different mixtures rather than silently repeating.

**Real result — v5 broke past the plateau where nothing else did:**

| version | overall SNR(dB) | overall STOI | overall PESQ | best-val epoch | total epochs run |
|---------|------------------|--------------|---------------|-----------------|-------------------|
| v1      | 8.671            | 0.6620       | 1.7882        | 7               | 12 (plateau-stopped) |
| v2      | 8.564            | 0.6656       | 1.7753        | 7               | ~17 (patience 10) |
| v4      | 8.586            | 0.6631       | 1.7718        | 11              | 31 (plateau-stopped) |
| **v5**  | **8.750**        | **0.6744**   | 1.7863        | **27**          | **42**            |

v5's best-val checkpoint arrived at epoch 27 — nearly 4x later than v1's
epoch 7 and more than 2x later than v4's epoch 11 — and its final val loss
(`0.13689`) is the best of all four versions. This is exactly the signature
you'd expect if the earlier versions' early plateaus were genuinely a data
ceiling, not an optimization ceiling: once given genuinely new information
per epoch (rather than the same 2,162 pairs reshuffled), the model kept
finding real improvements for far longer before its own validation curve
told it to stop.

Per-SNR-bucket, the hardest bucket (−5 to 0dB, the diagnosed weak spot)
improved the most in relative terms from v1→v5 (STOI +0.015, the largest
gain of any bucket: v1 0.591 → v5 0.606) — consistent with the low-SNR
capability gap that v2's diagnosis identified being the thing v5's dynamic
mixing (far more exposure to varied hard mixtures) most directly addressed.

**v5 also switched LR scheduling** (fixed cosine → `ReduceLROnPlateau`, see
Section 4) for a reason directly tied to this same insight: once training
data is no longer a small, fixed, exhaustible pool, there's no principled
reason to *pre-commit* to a specific epoch count the way cosine decay
requires — `max_epochs=120` became a loose ceiling rather than a tight
budget, and the reactive schedule let training run as long as genuine
improvement continued to show up.

### The methodology, not just the outcome

The transferable engineering discipline this project's real history
demonstrates, worth internalizing independent of this specific model or
dataset:

1. **Diagnose before you fix.** v1→v2 didn't start from "let's try a
   different LR" — it started from `diagnose_failures.py` establishing,
   concretely, *what kind* of mistake the model was making (over-suppression,
   18/18 worst pairs), before any lever was pulled.
2. **Isolate one variable at a time — and when you can't avoid changing
   several at once, verify the *unchanged* thing didn't secretly cause the
   result.** v2 changed three things simultaneously (a compromise given the
   deadline), but the v1-vs-v2-on-same-worst-pairs isolation test then
   checked whether the *shared* failure mode was caused by v1's original
   recipe or by v2's new one — confirming it predated v2's changes entirely.
3. **Verify a hypothesis against real data before trusting it.** "Maybe it's
   gunshot-specific data scarcity" was a plausible-sounding hypothesis that
   `diagnose_data_gaps.py`'s real cross-tabulation disproved *before* any
   gunshot-data-collection effort was spent chasing it.
4. **When several independently-varied attempts converge to the identical
   ceiling, suspect the one thing that was never varied.** v1, v2, and v4
   all plateaued at essentially the same performance despite substantially
   different training recipes — the shared, unvaried factor (a fixed
   2,162-pair dataset) was the actual root cause, and only became visible
   *because* three different optimization approaches had already been ruled
   out as an explanation.
5. **A fix should follow directly from the diagnosis, not from "let's also
   try this."** v5's dynamic mixing directly targets "the dataset is too
   small and fixed" — the diagnosed root cause — rather than being one more
   entry in a general grab-bag of plausible-sounding levers.

---

## 6. Reporting Δ-SNR instead of absolute SNR>15dB

### 🧠 The intuition

The PS's absolute SNR target (>15dB) sounds like a clean, simple bar to
clear. But it silently assumes something about the *starting point* of the
audio being cleaned — and this project's real test-set construction makes
that assumption false for most of the data.

### 📐 The math

Recall the test split's noisy-input SNR range is **−5dB to +15dB**
(`configs/finetune.yaml`'s locked mixing range, confirmed in Phase 1's
manifest `snr_db` values). SNR itself is defined (in dB) as:

```
SNR(dB) = 10 · log10( P_signal / P_noise )
```

where `P_signal`/`P_noise` are the average power of the clean speech and the
noise component, respectively.

Now consider a test pair mixed at, say, **−5dB input SNR** — meaning the
noise's power is roughly `10^(5/10) ≈ 3.16x` the speech's power at the
input. For the model's *output* to reach the PS's absolute target of
**>15dB**, it would need to improve the ratio by:

```
ΔSNR_required = 15 − (−5) = 20 dB
```

A 20dB improvement in SNR corresponds to reducing the noise-to-signal power
ratio by a factor of `10^(20/10) = 100x`. That is an extremely aggressive
bar for a *single* input clip drawn from the hardest end of this project's
own test distribution — and it's not a hypothetical edge case: by
construction, roughly a quarter of the test set sits in the −5 to 0dB bucket
(83 of 299 test pairs, per the per-SNR-bucket table in the development log's v1-vs-v5
comparison). For those pairs specifically, reaching an *absolute* 15dB
output would require recovering speech from a signal where noise started
out over 3x louder than speech, well past what "impossible to fully
recover" looks like for any denoising system, learned or classical — some
information genuinely is destroyed and can't be reconstructed, a real,
physical ceiling `how_it_works.md` describes explicitly.

Compare this to reporting **Δ-SNR (SNR improvement)** instead — the
difference between the model's output SNR and the *input's own* SNR for that
same clip:

```
ΔSNR = SNR_output − SNR_input
```

This measure doesn't reward or penalize a model based on how hard a
*specific* input clip happened to be — a model that takes a −5dB clip to
+1dB (a genuinely large, meaningful improvement, `ΔSNR = +6dB`) isn't
penalized relative to a model that takes an already-easy +12dB clip to
+15dB (`ΔSNR = +3dB`, a smaller real improvement despite clearing the
absolute 15dB bar) just because the second clip started closer to the
finish line. Reporting Δ-SNR is standard practice in the speech-enhancement
literature for exactly this reason: it separates "how hard was this
specific input" from "how much did the model actually help," which an
absolute output-SNR threshold conflates.

**This project's real numbers support the honest reading, not a flattering
one:** v5's overall SNR improvement over the unprocessed floor is `8.75 −
0.47 = +8.28dB` (using the noisy-floor's own overall SNR of 0.47dB from
the development log's evaluation table) — a large, real, consistently-observed
improvement across every category and every SNR bucket, reported honestly
alongside the fact that the absolute >15dB target was never reached by any
of the four fine-tuned versions.

### 🔗 The connection

This same absolute-vs-relative framing issue is why the development log consistently reports finetuned-vs-baseline deltas (SNR: 1.84→8.67dB,
**+6.83dB**; STOI: 0.592→0.662, **+0.070**; PESQ: 1.292→1.788, **+0.496**)
side-by-side with the honest absolute PS-target assessment, rather than
letting one framing stand in for the other.

### ❓ Check yourself

If a hypothetical "v6" achieved the exact same absolute SNR/STOI/PESQ
numbers as v5 but was trained and evaluated on a test set drawn entirely
from the +10 to +15dB range (dropping the −5 to +5dB pairs), would its
Δ-SNR over its own (easier) noisy floor look better or worse than v5's real
number — and would that comparison be telling you anything genuine about
which model is "better" at speech enhancement?

---

## 7. Final v5 results, in full, with honest assessment

**v5 is the FINAL, locked model** (user decision, 2026-09-11 — no further
training iterations planned). Checkpoint: `checkpoints/dns48_finetuned_v5_best.pt`,
best-val at epoch 27 of 42 epochs run, `val_total_loss=0.13689` — the best
validation loss of all four versions.

**Overall test-split results (n=299), all four versions, source
`results/baseline_results.csv`:**

| version | SNR(dB) | STOI   | PESQ   |
|---------|---------|--------|--------|
| v1      | 8.671   | 0.6620 | 1.7882 |
| v2      | 8.564   | 0.6656 | 1.7753 |
| v4      | 8.586   | 0.6631 | 1.7718 |
| **v5**  | **8.750** | **0.6744** | 1.7863 |

v5 is best or near-best on every metric — narrowly below v1 specifically on
PESQ (1.7863 vs. 1.7882, a 0.0019 gap).

**v5 per-category (n=299):**

| category   | n   | SNR(dB) | STOI  | PESQ  |
|-----------|-----|---------|-------|-------|
| gunshot   | 97  | 8.901   | 0.654 | 1.870 |
| stationary| 98  | 9.430   | 0.702 | 1.833 |
| general   | 104 | 7.969   | 0.667 | 1.664 |
| overall   | 299 | 8.750   | 0.674 | 1.786 |

**v5 per-SNR-bucket, vs. v1 (n=299):**

| bucket      | n  | v1 SNR | v5 SNR | v1 STOI | v5 STOI | v1 PESQ | v5 PESQ |
|-------------|----|--------|--------|---------|---------|---------|---------|
| −5 to 0 dB  | 83 | 5.929  | 6.147  | 0.591   | 0.606   | 1.478   | 1.495   |
| 0 to 5 dB   | 78 | 7.767  | 7.823  | 0.663   | 0.673   | 1.720   | 1.745   |
| 5 to 10 dB  | 60 | 10.560 | 10.571 | 0.714   | 0.725   | 1.978   | 1.940   |
| 10 to 15 dB | 78 | 11.040 | 11.047 | 0.697   | 0.710   | 2.040   | 2.019   |

**Honest PS-target assessment, v5 overall (n=299) — targets NOT cleared:**

| metric | v5 result | PS target | cleared? |
|--------|-----------|-----------|----------|
| SNR    | 8.75dB    | >15dB     | ❌ No |
| STOI   | 0.674     | >0.85     | ❌ No |
| PESQ   | 1.786     | >2.5      | ❌ No |

**What the project's real contribution is, stated honestly:** across four
methodologically-motivated fine-tuning iterations, this project achieved a
large, real, and *consistent* improvement over the classical
spectral-subtraction baseline on every metric, in every noise category, at
every SNR bucket — SNR +6.91dB (1.84→8.75), STOI +0.082 (0.592→0.674), PESQ
+0.494 (1.292→1.786), overall. It did **not** clear the PS's absolute
numeric targets on any of the three metrics, on any version, and this is
reported as-is rather than reframed. The real, load-bearing engineering
achievement here is not "we hit the number" — it's the diagnostic discipline
that identified *why* three different optimization strategies (v1's flat LR,
v2's reweighted/oversampled recipe, v4's curriculum + silence-penalty) all
converged to the identical ceiling, correctly attributed that ceiling to
static-dataset exhaustion rather than any optimizer/loss misconfiguration,
and then fixed the actual bottleneck (v5's dynamic mixing + expanded
speaker/speech pool) rather than continuing to search the wrong space.
