# Phase 2 — Spectral Subtraction Baseline + Evaluation Metrics

A deep dive into what was built, why, and the theory/math underneath.

---

## 🧠 1. The Intuition

### What is spectral subtraction?

Imagine you're on a phone call in a room with a fan running. The fan makes a
steady hiss that's *always there* — while you're talking and in the silence
before you talk. If you could somehow "measure" what the fan's hiss looks
like during the quiet moments, you could later subtract that exact amount
of hiss from the mixed sound whenever you talk, leaving something closer to
just your voice.

That's spectral subtraction: measure the noise's "fingerprint" during a
noise-only period, then subtract that fingerprint from every subsequent
frame of noisy audio, working in the frequency domain rather than on the
raw waveform (a waveform is just amplitude vs. time — you can't say "this
part is fan noise" and cleanly remove it there, because at any instant the
fan and voice samples are just added together into one number).

### Why the frequency domain?

A fan hiss is spread thin and evenly across many frequencies (like white
noise). A human voice concentrates its energy into specific, moving bands
(formants). If you decompose the mixed sound into "how much energy is at
each frequency, at each moment in time" (that's exactly what a spectrogram
is), the hiss and the voice look different and separable, even though in
raw amplitude terms they were tangled together.

### Why is this the "classical failure mode" the project argues against?

Spectral subtraction assumes the noise's fingerprint stays constant and
that you can measure it cleanly from a noise-only chunk. Real noise
(gunshots, engines, chatter) isn't constant, and clean noise-only segments
aren't always available. When the subtraction is imperfect, you get
"musical noise" — random frequency bins popping in and out — which is the
telltale artifact you hear from old noise-cancellation software. A learned
model (like the dns48 fine-tuning this project is building toward) doesn't
rely on this fixed-fingerprint assumption; it learns what speech and noise
*look like* from data. Building this baseline honestly, warts included, is
what gives the pitch ("our model beats this") something real to point at.

---

## 📐 2. The Math

### First, what is a Fourier Transform? (if this is new to you)

Any sound wave, no matter how complicated, can be described as a mix of
simple pure tones (sine waves) of different frequencies, each at some
volume. A piano chord is really three or four separate sine waves added
together; a voice is dozens of them layered at once. The **Fourier
Transform** is the math operation that takes a waveform (amplitude over
time) and tells you "how much of each frequency is present" — it doesn't
change the sound, it just re-describes the exact same information from a
different angle: instead of "loudness at each moment," you get "how much
of each pitch, total."

**Why do that at all here?** Because the earlier "fan hiss vs. voice"
intuition only becomes *usable* once you've made this switch. In raw
amplitude-over-time form, a hiss sample and a voice sample at the same
instant are just added into one number — there's no way to peel them
apart. In "how much of each frequency" form, hiss and voice occupy
different, separable regions, so you can act on one without touching the
other.

One single Fourier Transform over an entire recording would only tell you
the *average* frequency content for the whole clip — it would lose all
sense of *when* things happened (a gunshot 2 seconds in would get
smeared together with silence at second 0). So instead we use the
**Short-Time** Fourier Transform (STFT): chop the waveform into many
short overlapping chunks (called "frames" or "windows"), and run a
separate Fourier Transform on each chunk. Now you get frequency content
*per moment in time*, not just one number for the whole clip — this
is exactly what a spectrogram (a picture with time on one axis and
frequency on the other) visualizes.

### Step 1 — STFT (Short-Time Fourier Transform)

The waveform is chopped into overlapping short windows (here, 512 samples
= 32ms at 16kHz, sliding forward 128 samples = 8ms each time — a 75%
overlap, chosen because it satisfies the "constant overlap-add" (COLA)
condition for a Hann window, which is what guarantees the inverse
transform can reconstruct the signal without seams). Each window gets a
Fourier Transform, turning "amplitude over time" into "how much energy at
each frequency, for this instant."

The result is a complex-valued matrix `X[f, t]`, one complex number per
frequency bin `f` and time frame `t`. Don't worry if "complex number"
sounds intimidating — you don't need complex-number algebra to follow
this, just what the two pieces it carries *mean*: a complex number here is
a compact way to store two real numbers at once (magnitude and phase, both
plain real numbers), instead of using two separate arrays. The code always
immediately splits it back into those two ordinary numbers
(`spec.abs()`, `torch.angle(spec)`), so in practice you can just think of
`X[f, t]` as a pair `(magnitude, phase)`:

```
X[f, t] = magnitude(f, t) × e^(i × phase(f, t))
```

You can safely treat this line as notation, not something to compute by
hand — it's just the standard mathematical way of writing "a magnitude and
a phase glued into one complex number," the way `(x, y)` is a standard way
of writing a 2D point. The two pieces that actually matter:

- **Magnitude** — *how much* energy is at that frequency at that instant.
  This is what largely determines what you perceive as loudness/timbre.
  Think of it as the *height* of that frequency's sine wave.
- **Phase** — *where in the wave cycle* that frequency component is at
  this instant (a sine wave repeats every cycle — phase says "we're 30%
  of the way through one repeat" versus "70% of the way through"). This
  matters for reconstructing the exact waveform shape but is much less
  perceptually important than magnitude — two sounds with identical
  magnitude but scrambled phase tend to sound very similar to a human
  ear, which is precisely why step 5 below gets away with reusing the
  noisy signal's original phase unchanged.

### Step 2 — Estimate the noise magnitude spectrum

```
N_hat[f] = mean over t in [0, K) of |X[f, t]|
```

- `N_hat[f]` — the estimated noise magnitude at frequency `f` (one number
  per frequency bin, not per time frame — the core assumption is the noise
  doesn't change over time, so one snapshot represents it for the whole
  clip).
- `K` — `noise_estimate_frames` in the config (6 frames ≈ 6 × 8ms ≈ 48ms of
  audio). The classic Boll (1979) approach: assume the *first* K frames of
  the recording are noise-only (before speech starts) and average their
  magnitude.
- This project's honest caveat: Phase 1's data-mixing overlays the noise
  clip under the *entire* utterance, not just a leading silence, so "first
  K frames = noise-only" isn't strictly guaranteed here. That's
  acknowledged directly in the code rather than hidden — this is precisely
  the kind of assumption-breaking that hurts classical methods in the real
  world.

### Step 3 — Subtract, with oversubtraction

Plain subtraction (`|X| - N_hat`) tends to under-suppress noise because
`N_hat` is only an average, not the exact per-frame noise. Berouti et al.
(1979) introduced an oversubtraction factor `α` to subtract more
aggressively:

```
M[f, t] = |X[f, t]| - α × N_hat[f]
```

- `α` (`oversubtraction_factor` = 1.5 in config) — subtract 1.5× the
  estimated noise instead of 1×. At `α = 1`: plain subtraction. Larger `α`:
  more aggressive noise removal, but more speech gets damaged too (bigger
  trade-off toward under-noise, over-distortion).

### Step 4 — Spectral floor (avoiding musical noise)

If `M[f, t]` goes negative (noise estimate overshoots the actual noise in
that frame), naively clamping to zero causes "musical noise" — bins
flicker between zero and non-zero from frame to frame, which the ear
perceives as random chirps/tones. Instead, floor at a small fraction of
the *original* noisy magnitude:

```
M_floor[f, t] = max(M[f, t], β × |X[f, t]|)
```

- `β` (`spectral_floor` = 0.02) — never suppress more than 98% of a bin's
  energy, leaving a small noise-like residue rather than a hard zero. At
  `β = 0`: hard clamp, worst musical noise. At `β = 1`: no suppression at
  all (output ≈ input).

### Step 5 — Recombine with original phase, inverse STFT

```
X_hat[f, t] = M_floor[f, t] × e^(i × phase(X[f, t]))
```

The enhanced magnitude is paired back up with the **original noisy
phase** (not re-estimated — phase errors are known to matter less
perceptually than magnitude errors, a long-standing empirical finding in
speech processing, and estimating phase well is a much harder problem).
Then the inverse STFT (overlap-add the windows back together) turns this
back into a waveform, same length as the input.

---

## 💻 3. The Code

`src/baseline/spectral_subtraction.py`, the core function. (`torch` is the
PyTorch library — a numerical computing library that, among other things,
ships built-in, well-tested implementations of `stft`/`istft` so nobody
has to hand-roll the Fourier Transform math itself; `.abs()`/`torch.angle()`
below are simply "give me the magnitude" / "give me the phase" of a
complex tensor.)

```python
def spectral_subtract(noisy: torch.Tensor, cfg: dict) -> torch.Tensor:
    window = torch.hann_window(win_length)

    # Step 1: STFT -> complex spectrogram [freq_bins, frames]
    spec = torch.stft(noisy, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window,
                       return_complex=True)

    magnitude = spec.abs()      # Step 1b: split into magnitude...
    phase = torch.angle(spec)   # ...and phase

    # Step 2: noise fingerprint = average magnitude over first K frames
    noise_mag_estimate = magnitude[:, :frames_for_estimate].mean(dim=1, keepdim=True)

    # Step 3: oversubtract
    subtracted = magnitude - alpha * noise_mag_estimate

    # Step 4: floor to avoid hard zeros -> musical noise
    floor = beta * magnitude
    enhanced_magnitude = torch.maximum(subtracted, floor)

    # Step 5: recombine with ORIGINAL phase, inverse STFT
    enhanced_spec = torch.polar(enhanced_magnitude, phase)
    enhanced = torch.istft(enhanced_spec, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, length=noisy.shape[-1])
    return enhanced
```

`torch.polar(mag, phase)` builds a complex number from magnitude+phase in
one call — the direct inverse of `spec.abs()`/`torch.angle(spec)`.

Every number that could have been a "magic number" (`n_fft`, `hop_length`,
`noise_estimate_frames`, `alpha`, `beta`) is read from
`configs/finetune.yaml` (`baseline.*` keys), per the project's rules
rule 1.

---

## 📐+💻 The Three Evaluation Metrics

### SNR — Signal-to-Noise Ratio

**Intuition:** if you subtract the clean speech from the enhanced output,
whatever's left over is "error" (leftover noise + distortion introduced by
processing). SNR asks: how much bigger is the real signal than that
leftover error, in decibels?

**Quick decibel refresher, if you need it:** a decibel (dB) is not a raw
amount of energy — it's a *ratio*, expressed on a logarithmic scale, so
"twice the ratio" doesn't mean "twice the dB number." The key landmarks to
keep in your head: 0 dB means the two things being compared are exactly
equal; positive dB means the first thing (here, the clean signal) is
bigger; negative dB means it's smaller. Every +10 dB roughly means "10×
more energy," which is why the scale is logarithmic in the first place —
raw energy ratios in audio can span from 1 to a million or more, and a
logarithmic scale compresses that huge range into small, readable numbers
like "+1.84" instead of "6.9×". (`dataset.md` section 5 has the full
derivation if you want the formula worked through in detail — this doc
reuses the same idea for a different ratio: signal energy vs. error energy,
instead of signal energy vs. noise-clip energy.)

**Math:**
```
SNR(dB) = 10 × log10( sum(clean²) / sum((enhanced - clean)²) )
```
- `sum(clean²)` — total energy of the true clean signal (the "signal
  power").
- `enhanced - clean` — the error signal, and `sum((enhanced-clean)²)` its
  power.
- The `10 × log10(...)` converts a power *ratio* into decibels — a
  logarithmic scale because human perception of loudness is roughly
  logarithmic, and because raw power ratios span many orders of magnitude.
- At the extreme: if `enhanced == clean` exactly, the denominator → 0 and
  SNR → +∞ (perfect). If the "enhanced" signal is pure noise (uncorrelated
  with clean), SNR is low or negative.

```python
def snr(clean, enhanced, eps=1e-8):
    noise = enhanced - clean
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10((signal_power + eps) / (noise_power + eps))
```
(`eps` avoids a divide-by-zero / log(0) if a signal is silence.)

### STOI — Short-Time Objective Intelligibility

**Intuition:** SNR treats every error equally, but a listener cares about
whether they can still understand the *words*, not just whether the
waveform matches numerically. STOI breaks the signal into short time-
frequency segments (similar to the STFT idea above) and measures how well
the *pattern* of energy in the enhanced signal correlates with the clean
signal's pattern in each segment, then averages that correlation into a
single 0–1 score. It's been validated against real human intelligibility
listening tests, which is why it's an industry-standard proxy rather than
inventing an ad-hoc metric.

```python
def stoi_score(clean, enhanced, sample_rate=16000):
    return _stoi(clean, enhanced, sample_rate, extended=False)
```
This project uses `pystoi`'s implementation directly rather than
reimplementing it — STOI's exact banding/correlation procedure is a
published, standardized algorithm (Taal et al., 2011), not something to
casually reinvent.

### PESQ — Perceptual Evaluation of Speech Quality

**Intuition:** an ITU telecom standard (P.862) originally built to let
phone companies automatically score call quality the way a human panel
would, without paying humans to listen to every test call. It models
aspects of human auditory perception (frequency masking, loudness
scaling) and outputs a Mean Opinion Score-like number, roughly 1 (bad) to
4.5 (excellent).

```python
def pesq_score(clean, enhanced, sample_rate=16000, mode="wb"):
    return _pesq(sample_rate, clean, enhanced, mode)
```
`mode="wb"` = wideband PESQ (ITU-T P.862.2), the correct variant for
16kHz audio (narrowband PESQ is for 8kHz telephony-quality audio; using
narrowband here would silently degrade toward that lower bandwidth
assumption and give misleading scores).

### Why all three together?

They disagree on purpose:
- **SNR** — purely numerical, easy to compute, but doesn't correlate
  perfectly with what a human actually hears.
- **STOI** — "can I understand the words."
- **PESQ** — "does it sound good," closer to overall perceived quality.

A method can score well on one and poorly on another — e.g. spectral
subtraction's musical noise can look fine on raw SNR (the average energy
of the error may be small) while sounding noticeably artificial (worse
PESQ) or actually reducing intelligibility (worse STOI) than the numbers
alone suggest. Reporting all three, per noise category, is what lets the
per-noise-type breakdown ("does spectral subtraction fail specifically on
gunshots, a sharp transient, versus stationary hiss?") tell a real story
instead of a single misleading average.

---

## 📊 The Real Results — What the Numbers Actually Mean

This is the table produced by running both scripts on the real 299-pair
test split (2026-09-11):

| column   | category   | n   | SNR(dB) | STOI  | PESQ  |
|----------|-----------|-----|---------|-------|-------|
| noisy    | gunshot   | 97  | -1.09   | 0.566 | 1.296 |
| noisy    | stationary| 98  | 2.01    | 0.631 | 1.368 |
| noisy    | general   | 104 | 0.47    | 0.596 | 1.260 |
| noisy    | overall   | 299 | 0.47    | 0.598 | 1.307 |
| baseline | gunshot   | 97  | -0.12   | 0.561 | 1.297 |
| baseline | stationary| 98  | 3.22    | 0.626 | 1.309 |
| baseline | general   | 104 | 2.36    | 0.588 | 1.271 |
| baseline | overall   | 299 | 1.84    | 0.592 | 1.292 |

Read this row by row, column by column — each number is answering a
different question.

### Reading the `noisy` row first — the floor

Before judging the baseline at all, look at what "doing absolutely
nothing" scores. This matters because every number below only means
something *relative to this floor*. A "baseline SNR of 1.84 dB" sounds
unimpressive in isolation, but SNR is not an absolute-quality scale — it's
only informative as a *difference* from the floor.

- **noisy overall SNR = 0.47 dB.** Recall SNR compares clean-signal power
  to the power of `(processed − clean)`. For the *noisy* column,
  "processed" is literally the untouched noisy recording, so this number
  is really "how loud is the speech relative to how loud the mixed-in
  noise is," on average across the test set. 0.47 dB is close to 0, which
  means on average the noise power is roughly *comparable to* the speech
  power in these mixtures — a genuinely hard test set, not an easy one.
  This checks out against Phase 1's config: SNR mixing range was −5 dB to
  +15 dB, so the test split spans from noise-dominated to speech-dominated
  pairs, and 0.47 dB overall is a plausible middle-of-the-distribution
  average.
- **noisy gunshot SNR = −1.09 dB (worst of the three).** Negative SNR
  here means, on average, gunshot noise clips carry *more* power than the
  speech they're mixed with. That tracks with what a gunshot physically
  is — a short, very high-amplitude transient — versus MUSAN's steadier
  background noise, which tends to be mixed in at more moderate,
  consistent levels.
- **noisy stationary SNR = 2.01 dB (best of the three).** Stationary noise
  (steady hiss/hum-type MUSAN clips) is the "easiest" category even with
  zero processing — consistent with stationary noise typically being
  mixed at levels that don't overwhelm speech, since it doesn't have sharp
  loud peaks the way a gunshot does.

### Reading the `baseline` row — did spectral subtraction help?

Now compare each baseline number to its noisy counterpart directly above
it — that comparison *is* the result.

- **Overall SNR: 0.47 → 1.84 dB (+1.37 dB gain).** The subtraction step
  genuinely removed some noise energy, on average, across the whole test
  set. This is spectral subtraction doing its one job correctly: the
  noise magnitude estimate correlated enough with the real noise that
  subtracting it left less residual energy than leaving the mixture
  untouched.
- **Overall STOI: 0.598 → 0.592 (a small decrease).** This is the
  important, non-obvious result. STOI measures whether the *pattern* of
  time-frequency energy still resembles clean speech's pattern closely
  enough for a listener to parse the words. Spectral subtraction doesn't
  just remove noise energy — the oversubtraction (α=1.5) and flooring
  (β=0.02) steps also *distort* the speech's own spectral pattern in
  frames where speech and noise energy overlap in frequency (this is
  unavoidable: the algorithm doesn't know which energy in a bin came from
  speech versus noise, it just subtracts a fixed estimate from the total).
  So intelligibility went down slightly even while raw energy-ratio went
  up. This is direct numeric evidence that **SNR alone would have
  reported a false win** — reading only the SNR column, you'd conclude
  the baseline clearly helped; STOI says it's roughly a wash on
  intelligibility, maybe very slightly worse.
- **Overall PESQ: 1.307 → 1.292 (a small decrease).** PESQ is closest to
  "would this sound acceptable to a person," and it also went down
  slightly. This is the numeric fingerprint of *musical noise* described
  in the theory section above: the flooring/oversubtraction process
  leaves behind the characteristic warbling artifact, and PESQ's
  perceptual model penalizes that artifact even though it technically
  reduced average error energy. Both PESQ and STOI moving down while SNR
  moves up, together, is exactly the signature the project's pitch
  predicted this classical method would show — it's not a coincidence or
  a bug in the eval code, it's the textbook trade-off spectral subtraction
  is known for.

### Reading the per-category breakdown — where it fails hardest

- **Gunshot stays SNR-negative even after processing (−1.09 → −0.12 dB).**
  It improved, but nowhere near enough to cross into positive territory,
  and it's still the worst-performing category by a wide margin on every
  metric. The reason ties directly back to the noise-estimate assumption
  documented in the code: the algorithm estimates one *constant* noise
  fingerprint from the first ~48ms of the signal and subtracts that same
  fingerprint from every later frame. A gunshot is the opposite of
  constant — it's a short, sharp burst, so a fingerprint measured from an
  early quiet-ish moment tells you almost nothing about the loud transient
  spike that may occur later in the clip. This is the single clearest
  piece of evidence in this table for *why* a learned model (Phase 3) is
  needed: it doesn't have to assume noise is constant over time.
- **Stationary is the best-performing baseline category on SNR (2.01 →
  3.22 dB, the largest gain of the three) but still loses a touch of STOI
  (0.631 → 0.626).** This is spectral subtraction closest to its "ideal"
  use case — MUSAN's steady background noise really does look like a
  fixed fingerprint over time, so the core assumption holds reasonably
  well here, and the SNR gain is real and meaningfully sized. Even here
  though, the STOI/PESQ dip is still present, just smaller — confirming
  the musical-noise trade-off isn't specific to hard cases like gunshots,
  it's a structural property of the algorithm itself, just less severe
  when the underlying assumption is closer to true.
- **General sits in between on every metric** (SNR +1.89 dB, small STOI/
  PESQ dips) — consistent with MUSAN's "general" (non-stationary,
  non-gunshot) noise being neither as steady as the stationary subset nor
  as transient/extreme as gunshots.

### The one-sentence takeaway for the PPT/demo

*Spectral subtraction measurably reduces noise energy (SNR +1.37 dB
overall) but slightly degrades both intelligibility (STOI) and perceived
quality (PESQ) versus doing nothing at all, and this trade-off is worst
specifically on transient noise (gunshots) — this is the exact classical
failure mode the project's fine-tuned model needs to beat on all three
metrics, not just SNR, to be a genuine improvement rather than a
one-metric illusion.*

---

## 🔗 4. The Connection

- This baseline reuses the exact `manifests/test.json` schema from Phase 1
  (noise_category tagging), so the per-category breakdown code written now
  in `evaluate.py` is the *same code path* Phase 3 will reuse once the
  fine-tuned dns48 model produces its own `results/finetuned/` output —
  only a `finetuned` column needs real data, nothing about `evaluate.py`
  itself needs to change.
- The "noisy" column (metrics computed with zero processing at all) is the
  floor every method must beat — if spectral subtraction doesn't clearly
  outperform doing nothing, that's itself a finding worth reporting, not a
  bug to hide.
- The STFT machinery here (`torch.stft`/`torch.istft`) is conceptually the
  same operation the multi-resolution STFT loss (planned for fine-tuning,
  see the development log) uses internally — understanding spectral subtraction's
  magnitude/phase split now makes that loss function's mechanics much less
  mysterious later.

---

## ❓ Check Your Understanding

If you doubled `oversubtraction_factor` (α) from 1.5 to 3.0 and left
everything else the same, what would you expect to happen to (a) the SNR
number, and (b) how "natural" the enhanced audio sounds — and why might
those two move in *different* directions?
