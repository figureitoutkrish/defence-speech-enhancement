# Export & Quantization Deep Dive — PyTorch to On-Device Core ML

This document explains, from first principles to expert depth, how the
locked v5 checkpoint (`checkpoints/dns48_finetuned_v5_best.pt`) became three
shippable on-device artifacts (fp32, fp16, int8 `.mlpackage` files) — the
real bugs hit along the way, the real verification numbers at every step,
and the real engineering reasoning behind each precision decision. All
numbers are quoted directly from the development log's Phase 5a/5b entries; nothing is
approximated.

---

## 1. What is ONNX, and why is it the bridge format?

### 🧠 The intuition

Different deep learning frameworks (PyTorch, TensorFlow, Core ML) each have
their own internal way of representing "what operations does this model do,
in what order, on what shapes." A model trained in PyTorch is, internally,
a PyTorch-specific object — you can't hand it directly to an iOS device's
Core ML runtime, because Core ML doesn't know how to read PyTorch's internal
representation.

**ONNX (Open Neural Network Exchange)** solves this by defining a single,
framework-independent way to describe a model: a **computation graph** — a
list of mathematical operations (convolution, matrix multiply, add,
activation functions, etc.) and how they connect, with no dependency on
PyTorch, TensorFlow, or any specific runtime. Think of it like a universal
sheet-music notation: a symphony written on a page of standard musical
notation can be played by any orchestra that reads musical notation, even
though the composer might have written the first draft on a specific piece
of software. ONNX is that shared notation for neural network graphs — you
export *once*, from whichever framework trained the model, into this common
format, and then any ONNX-compatible tool downstream can read it without
needing to know or care that the model started life in PyTorch.

### 📐 The mechanism, from first principles

An ONNX file is, structurally, a serialized graph: a list of **nodes** (each
one an operator like `Conv`, `LSTM`, `Add`, `Relu`), each with named input
and output **tensors**, plus the model's learned **weights** stored as
constant tensors attached to specific nodes. There is no Python code in an
ONNX file — it is a pure, static description of "given input of this shape,
apply these operations in this order, using these fixed numbers, to produce
output of that shape."

This project's export (`src/export/to_onnx.py`) targets **opset 17** — ONNX
defines its operator set in versioned "opsets," and pinning a specific
version (rather than "whatever the exporter defaults to") is exactly the
same no-magic-numbers discipline this project applies to hyperparameters:
the config (`configs/finetune.yaml`'s `export:` block) locks
`opset_version: 17` explicitly rather than leaving it implicit.

### 🔗 The connection

ONNX being a static graph (not live Python) is precisely what makes the
`torch.jit.trace` mechanism in Section 2 both necessary and hazardous — the
export step has to convert *dynamic Python code* into a *static graph*, and
anything in the original Python that isn't a pure tensor operation has to be
"frozen" into the graph somehow, which is exactly where this project's real
bug (Section 2) came from.

### ❓ Check yourself

If ONNX has no Python interpreter behind it — just a fixed list of
operations — what does that imply about a PyTorch model that contains an
`if` statement whose condition depends on the *actual numeric content* of
an input tensor at runtime (as opposed to just its shape)?

---

## 2. `torch.jit.trace`: what it actually does, and the real bug it caused here

### 🧠 The intuition

`torch.jit.trace` converts a PyTorch model into a static graph (the same
kind of representation ONNX needs) by a specific, somewhat blunt method: it
runs the model **once**, on one concrete example input, and **records every
tensor operation that actually executed** during that single run — literally
watching what happened and writing it down as a fixed recipe. It does *not*
read or understand your Python source code, your `if` statements, or your
loop logic — it only sees the sequence of tensor operations that happened to
fire for that one specific input.

This is the crux of tracing's central hazard: **anything in your model that
isn't itself a tensor operation — plain Python arithmetic on Python `int`s
or `float`s, `if` branches based on those plain values, Python-level loop
counts — gets silently "baked in" as whatever value it happened to have
during that one traced run**, because tracing has no way to represent "this
was a piece of Python logic that could have gone differently" in a graph
format that only understands tensor ops.

### 📐 The real bug hit in this project

This project hit exactly this hazard, twice, in `denoiser`'s own model code
during Core ML conversion (`src/export/to_coreml.py`, Phase 5b).

**Where it came from:** `Demucs.forward()` internally calls
`valid_length(length)` — a function that computes, from the model's static
architecture hyperparameters (`resample=4, depth=5, kernel_size=8, stride=4`)
and the input's length, exactly how much zero-padding is needed before the
encoder/LSTM/decoder stack so the shapes work out cleanly through every
layer. This computation is pure **Python integer arithmetic** — plain `int`
math, not `torch` tensor operations. Similarly, `denoiser/resample.py`'s
`downsample2`/`upsample2` helpers contain lines like:

```python
*other, time = x.shape        # reads a tensor's shape into plain Python ints
```

and a parity check:

```python
if x.shape[-1] % 2 != 0:      # a Python-level branch on a shape-derived int
    ...
```

When you run this code *eagerly* (normal PyTorch execution, no tracing),
`x.shape[-1]` is read as an ordinary Python `int`, the `%` and `!=` are
ordinary Python operators, and the `if` is an ordinary Python branch — all
of this happens once per call, correctly, no matter what the actual input
length is, because it's re-evaluated fresh every single time the function
runs.

But under `torch.jit.trace`, the tracer doesn't see "Python code that reads
a shape and branches on it" — it sees a `Tensor.shape` access, which the
tracer machinery sometimes converts into a traced *tensor* operation (a
0-dimensional tensor holding that shape value) rather than leaving it as a
plain Python int, specifically so the resulting graph *could* in principle
handle a different shape at inference time. The tracer then tried to treat
that traced 0-d tensor as if it were still usable in ordinary Python
integer arithmetic and comparisons — which is where the visible symptom
appeared: a **`TracerWarning`** flagging that a Python value was derived
from a tensor in a way that might not generalize to different inputs, and,
downstream in `coremltools`' MIL conversion frontend, an outright crash:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

`coremltools`'s op-lowering code (`ops.py::_int`/`_cast`) hit a traced 0-d
array where it expected an ordinary Python `int` it could directly cast —
because tracing had converted what used to be plain Python arithmetic into
a tensor-valued graph node instead.

### 💻 The `TraceableDemucs` fix, and why it's valid *here specifically*

The fix (`src/export/traceable_demucs.py::TraceableDemucs`) is to **wrap**
the trained `Demucs` instance (same weights, same layers — `denoiser/
demucs.py` and `denoiser/resample.py` are left completely untouched) and
replace every piece of length-dependent glue arithmetic with **hardcoded
Python constants**, precomputed once, outside the traced function, for
the one fixed input length this export targets: **64,000 samples (4.0s at
16kHz)** — exactly `training.segment_seconds` / `export.fixed_length_seconds`
from config.

The precomputed constants, verified by direct arithmetic against the real
model hyperparameters:
- `valid_length(64000) = 64085` (pad amount: 85 samples).
- Both `downsample2` calls in the `resample=4` path see **even-length**
  input at this specific length (256,340, then 128,170) — meaning the
  odd-length parity branch is *never* taken for this one fixed shape, so it
  can be safely hardcoded away rather than reproduced.
- `upsample2`'s two input `time` values (64085, 128170), `downsample2`'s two
  `odd_time` values (128170, 64085), and the decoder's 5 skip-connection
  slice lengths (`_DECODER_LENGTHS = [249, 1000, 4004, 16020, 64084, 256340]`)
  — all replaced from `skip[..., :x.shape[-1]]`-style shape-dependent slicing
  to plain fixed-index slicing with these precomputed ints.

**Why hardcoding is valid here, and would NOT be valid in general:** this
model's on-device deployment target is a *single, fixed input length* — the
input contract is locked at exactly 64,000 samples (verified against
`ModelRunner.swift`'s own documented contract in the iOS app: "The
traced/converted graph is ONLY valid at this exact length"). Because the
shape is fixed by design at export time, every one of these Python-level
length computations has exactly one possible answer, forever, for this
artifact — hardcoding them isn't cutting a corner, it's making explicit
something that was already, structurally, a constant for this specific
export. This would be **invalid** for a model meant to handle variable-length
input dynamically (e.g. the dynamic-length ONNX export from Phase 5a, which
deliberately kept `dynamic_axes` and did *not* hardcode anything) — hardcoding
there would silently produce wrong output for any input length other than
whatever happened to be traced, which is precisely the bug class this whole
section is about.

Two self-inflicted instances of this same hazard were found and fixed
*within* the wrapper itself, during debugging:
1. The wrapper's own `_upsample2_traceable`/`_downsample2_traceable` helper
   functions had been copied near-verbatim from `denoiser/resample.py` and
   still contained the exact same `*other, time = x.shape` pattern the
   wrapper was built to eliminate — reintroducing the hazard by copying the
   original code too literally instead of also replacing this line.
2. An added runtime guard, `if mix.shape[-1] != FIXED_INPUT_LENGTH`, placed
   *inside* `forward()` itself, also traced into a graph op (the same
   shape-to-Python-branch hazard). Fix: moved this check to `to_coreml.py`,
   executed once *before* tracing begins, rather than inside the traced
   function.

### The verification method — not "it converted without error"

Per this project's standing discipline (verify before trusting, applied
consistently across every phase), a clean conversion was explicitly **not**
treated as sufficient proof of correctness. Two behavioral checks were run,
in order, before Core ML conversion was attempted again:

1. **Eager wrapper vs. original `Demucs.forward()`** — same random input, at
   length 64,000, run through both the unmodified original model and the new
   `TraceableDemucs` wrapper (both still in ordinary eager PyTorch, no
   tracing involved yet). Result: **exact 0.0 max absolute difference** —
   proof the wrapper's hardcoded-constant rewrite computes *identically* to
   the original dynamic arithmetic, for this one fixed length.
2. **Traced wrapper vs. eager wrapper** — the same wrapper, once actually
   put through `torch.jit.trace`, compared against its own eager (untraced)
   output on the same input. Result: **exact 0.0 max absolute difference**,
   with no shape-cast `TracerWarning` remaining — proof tracing itself
   introduced no distortion once the Python-arithmetic hazard had been
   removed.

Only after both checks passed cleanly was Core ML conversion re-attempted.

### 🔗 The connection

This exact discipline — "does it run without error" is not "is it correct"
— is the same standard applied throughout Phase 5's verification chain
(Section 3): a successful `ct.convert()` call was never treated as proof of
anything beyond "the conversion process didn't crash."

### ❓ Check yourself

If this model instead needed to support *multiple* fixed input lengths (say,
2-second and 4-second variants, each shipped as its own `.mlpackage`), would
`TraceableDemucs`'s current hardcoded-constant approach need one wrapper
instance per length, or could a single wrapper somehow handle both — and why?

---

## 3. The full verification chain: isolating drift, one variable at a time

### 🧠 The intuition

Every conversion step (PyTorch→ONNX, ONNX→Core ML, fp32→fp16, fp32→int8)
introduces the *possibility* of numerical drift. The engineering challenge
is that if you only measure the drift at the very end (compare the original
PyTorch model to the final shipped artifact), and you find a gap, **you
cannot tell which step caused it** — was it the ONNX export? The Core ML
conversion? The precision reduction? A confound in how you measured?

This project's verification strategy is the standard scientific-method
discipline of **changing exactly one variable at a time**, and confirming at
each step that the *previous* step's already-established cleanliness still
holds, before introducing the next variable.

### 📐 The chain, step by step, with real numbers

**Step 0 — baseline, established first:** v5's own PyTorch evaluation on the
full 299-pair test split (Phase 4/final): **SNR 8.75dB, STOI 0.674, PESQ
1.786** (variable-length inputs — clips are *not* cropped to a fixed length
for this number; it's the reference point everything downstream is measured
against).

**Step 1 — PyTorch → ONNX (`verify_onnx.py`).** First raw comparison: ONNX
overall SNR 8.4508dB, STOI 0.6700, PESQ 1.7505 — a delta of **SNR −0.2992dB,
STOI −0.0040, PESQ −0.0355** against Step 0. This delta was immediately
flagged, in the script's own logged warning, as **confounded**: ONNX's fixed
export requires a fixed input length, so test clips were center-cropped/
zero-padded to 4.0s (64,000 samples) before being run through the ONNX
graph — and **184 of 299 test clips (61.5%) are longer than 4 seconds**
(confirmed by direct manifest inspection: min 1.445s, max 32.485s, mean
6.47s). This means most of the test set had *real audio content physically
discarded* by the crop, entirely independent of whether the ONNX export
itself was faithful.

**The isolation fix — `verify_onnx_isolated.py` (written specifically to
resolve this ambiguity):** feed the **identical already-cropped waveform**
to both the ONNX graph and the original PyTorch model — holding the crop
variable perfectly constant, so any *remaining* difference between the two
outputs can only be attributable to the export/runtime itself, not to what
content each one saw.

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| **export drift** (onnx vs. pytorch, identical cropped input) | **−0.0000** | **−0.0000** | **−0.0000** |
| **crop effect** (pytorch-on-cropped-input vs. v5's full-length PyTorch eval) | −0.2992 | −0.0040 | −0.0355 |

**Conclusion: the entire Step-1 gap is exactly, exclusively, the crop
effect.** Export drift is precisely zero to displayed precision — the ONNX
graph computes *identically* to the PyTorch model when given the same input.
This satisfied the project's own explicit stopping condition ("don't
proceed to Core ML/TFLite on a broken export") — proceeding was supported by
evidence, not assumed.

**Step 2 — ONNX → Core ML fp32 (`verify_coreml.py`).** Same fixed-length-crop
methodology (n=299), compared this time against the *ONNX-isolated* baseline
(already known to be crop-confound-free relative to that specific
comparison):

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| Core ML (fp32 mlprogram) | 8.4508 | 0.6700 | 1.7505 |
| ONNX isolated baseline | 8.4508 | 0.6700 | 1.7505 |
| **delta (coreml − onnx)** | **+0.0000** | **−0.0000** | **+0.0000** |

**Conclusion: zero measurable conversion-format drift.** Note this specific
comparison isolates *conversion format* (ONNX graph representation vs. Core
ML `mlprogram` representation) from *precision* (both are fp32 at this
step, deliberately — `compute_precision=ct.precision.FLOAT32` was
explicitly forced during this conversion, overriding `mlprogram`'s default
fp16 compute precision on Apple targets, specifically so this comparison
wouldn't conflate format drift with precision drift).

**Step 3 — fp32 → fp16 (`verify_coreml.py --fp16`).** Same methodology,
fp16 `.mlpackage` compared against the already-verified fp32 Core ML
artifact (not against ONNX — isolating pure precision effect from the
already-settled format question):

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| Core ML fp16 | 8.2036 | 0.6693 | 1.6881 |
| Core ML fp32 (baseline) | 8.4508 | 0.6700 | 1.7505 |
| **delta (fp16 − fp32)** | **−0.2472** | **−0.0007** | **−0.0624** |

**Step 4 — fp32 → int8, weights-only (`verify_coreml.py --int8`).** Same
methodology, int8 `.mlpackage` (quantized directly from the already-verified
fp32 artifact, not re-derived from ONNX or re-traced from PyTorch) compared
against fp32 Core ML:

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| Core ML int8 (weights-only) | 8.4443 | 0.6698 | 1.7481 |
| Core ML fp32 (baseline) | 8.4508 | 0.6700 | 1.7505 |
| **delta (int8 − fp32)** | **−0.0065** | **−0.0002** | **−0.0024** |

### 🔗 Why this "change one variable" discipline matters

Notice what each step's isolation makes possible: because Step 1 proved
export drift is exactly zero, Step 2's comparison against the ONNX baseline
cleanly isolates conversion-*format* drift alone. Because Step 2 proved
conversion-format drift is exactly zero, Steps 3 and 4's comparisons against
the fp32 Core ML baseline cleanly isolate *precision* effects alone, with no
possibility that some of the observed delta is secretly leftover
format-conversion noise from an earlier, unverified step. If any earlier
step in this chain had been skipped or left unverified, a downstream
"the delta looks small" or "the delta looks large" conclusion would rest on
an untested assumption about where in the chain that delta actually
originated. This is the same "isolate one variable, verify a hypothesis
against real data" discipline that ran through the entire Phase 3 training
diagnostic journey (see `docs/phase3_training_deep_dive.md`, Section 5).

### ❓ Check yourself

Suppose Step 2 (ONNX → Core ML fp32) had shown a non-zero delta instead of
exactly zero. Would that delta necessarily mean the Core ML conversion was
"worse" than the ONNX export, or could there be an alternative,
equally-valid explanation given everything you now know about how
`mlprogram`'s default `compute_precision` behaves?

---

## 4. Precision formats from first principles: fp32, fp16, int8

### 🧠 The intuition

A number stored in a computer isn't stored with infinite precision — it's
stored using a fixed number of bits, and how those bits are laid out
determines two things simultaneously: **how large or small a number can be
represented at all** (dynamic range), and **how finely nearby numbers can
be distinguished** (precision). There's an unavoidable tradeoff: fewer bits
means less storage and (often) faster computation, but coarser distinctions
between nearby values.

### 📐 The math — fp32 and fp16 (floating point)

Both fp32 and fp16 use the same *kind* of layout — a sign bit, an exponent
field, and a mantissa (fraction) field — following the general floating-point
formula:

```
value = (−1)^sign × 1.mantissa × 2^(exponent − bias)
```

- **fp32 (32-bit, "single precision"):** 1 sign bit, 8 exponent bits, 23
  mantissa bits. The 8 exponent bits give a huge dynamic range (roughly
  `10⁻³⁸` to `10³⁸`); the 23 mantissa bits give roughly 7 decimal digits of
  precision at any given magnitude.
- **fp16 (16-bit, "half precision"):** 1 sign bit, 5 exponent bits, 10
  mantissa bits. Both dynamic range (roughly `10⁻⁵` to `65504`) and
  precision (roughly 3 decimal digits) are substantially reduced — half the
  total bits doesn't just halve storage, it specifically squeezes both the
  range of representable magnitudes *and* how finely values are
  distinguished within that range.

Because floating point's exponent lets the *spacing* between representable
values scale with magnitude (values near 1.0 are spaced closely; values
near 1000.0 are spaced much further apart, in absolute terms, for the same
number of mantissa bits), fp16's precision loss isn't uniform — it's worse,
in absolute terms, for larger-magnitude intermediate values, which matters
directly for Section 5's LSTM discussion.

### 📐 The math — int8 (fixed point, via quantization)

int8 has no exponent at all — it's a plain 8-bit signed integer, representing
only the 256 values from −128 to 127. To represent a *continuous* range of
real-valued weights (which is what a trained neural network's weights
actually are) using only these 256 discrete integer buckets, you need an
explicit mapping: **quantization**.

The standard **affine (scale + zero-point) quantization** formula, which is
what `coremltools.optimize.coreml.linear_quantize_weights` implements
(mode `"linear_symmetric"` in this project's case — meaning the zero-point
is fixed at 0, a simplification valid when the value range is roughly
symmetric around zero, which weight distributions in a trained network
typically are):

```
q = round(w / scale)                    # quantize: real weight -> int8 value
ŵ = q × scale                           # dequantize: int8 value -> approximate real weight
```

Where:
- `w` is the original real-valued (fp32) weight.
- `scale` is a per-channel constant (this project used **per-channel
  granularity**, coremltools 9.0's documented default for this call) chosen
  so the full range of that channel's real weight values maps onto the
  available int8 range (−128 to 127) as tightly as possible without
  clipping.
- `q` is the stored int8 integer.
- `ŵ` is the value you get back after dequantizing — necessarily an
  *approximation* of the original `w`, since `round()` discards whatever
  fractional information didn't survive the division-then-rounding step.

**The error introduced by one quantize/dequantize round-trip** is bounded by
half the step size:

```
|w − ŵ| ≤ scale / 2
```

A smaller `scale` (achieved by choosing it per-channel, so each channel's
own actual value range is used, rather than one global scale for the whole
model) means less error per weight — this is exactly why per-channel
granularity, rather than one single scale for the entire model, was the
right default choice here.

**What actually happened in this project, confirmed against the
coremltools 9.0 API directly rather than assumed:** `linear_quantize_weights`
quantizes **only the stored weight tensors** (via `constexpr_affine_
dequantize`/`constexpr_blockwise_shift_scale` graph ops) — at inference
time, weights are *dequantized back to float* (`ŵ`) before any actual
matrix multiply/convolution happens. `coremltools.optimize.coreml.
linear_quantize_activations` is a separate, uncalled function in this
project — meaning **activations (the intermediate values flowing between
layers during inference) remain full float throughout**, never touched by
int8 at all. This is why this project's int8 artifact is described
precisely as "weights-only" quantization, not full int8 inference — a real,
verified distinction, not a formality.

**Extremes check:** if `scale` were huge relative to the actual weight
values, most weights would quantize to the same handful of int8 buckets —
massive information loss, large `ŵ` error. If `scale` were near zero
(weights spanning a very narrow real range), quantization error per weight
would be tiny — this is exactly why per-channel scale (letting each
channel use its own tightly-fit scale, rather than being stretched to
accommodate the model's single largest-magnitude weight anywhere) keeps
error small in practice.

### 🔗 Why LSTMs are historically riskier to quantize than conv layers

**The intuition:** a convolutional layer processes its input once, in one
pass — any precision error introduced there affects that one layer's output,
which then moves forward through the rest of the network exactly once. An
LSTM, by contrast, is **recurrent**: it processes a sequence step by step,
and at every single time step, it feeds its *own previous output* (the
hidden state) back in as part of the *next* step's input. This creates a
mechanism for error to **compound**: a small precision error introduced at
time step 1 doesn't just affect that step's output — it becomes part of the
input to time step 2, whose own (now doubly-affected) output becomes part
of the input to time step 3, and so on, for however many time steps the
sequence has.

This is a fundamentally different error-propagation structure than a
feedforward conv stack, where error from layer `k` reaches layer `k+1` once
and never gets fed back through the same computation again. Formally, if
`ε_t` represents the error introduced at recurrent step `t`, and the LSTM's
own recurrence has some sensitivity `J` (a Jacobian-like factor describing
how much a hidden-state error at step `t` influences the hidden state at
step `t+1`), the accumulated error after `T` steps grows roughly like a sum
of terms involving powers of `J` — if `J` isn't well below 1, this can
compound rather than dampen out, unlike single-pass feedforward error which
never gets this opportunity to reinforce itself across many repeated
applications of the same operation.

**How this project's real dns48 architecture is affected:** the model
contains a 2-layer LSTM (hidden size 768) sitting at the compressed
bottleneck between the 5-layer encoder and 5-layer decoder — exactly the
component where recurrent error-compounding risk applies, on top of the
5+5 feedforward conv layers and 2 downsample stages surrounding it. The
fp16 delta result (Section 3, Step 3: SNR −0.2472dB, PESQ −0.0624, "the
largest proportional hit of the three metrics," per the development log) is
**plausible given** exactly this mechanism: fp16 changes *runtime compute
precision*, meaning every one of the LSTM's many sequential recurrent
steps computes at reduced precision and feeds its own (now-degraded)
hidden state forward — precisely the compounding pathway described above.

**This project's real, concrete check for this specific failure mode:**
`verify_coreml.py --int8`'s per-SNR-bucket breakdown was added specifically
to look for an LSTM-quantization-error signature (the hypothesis being: if
recurrent error compounds over a longer effective sequence, harder/noisier
inputs — which force the LSTM to work harder across the sequence — might
show a *disproportionately* larger int8 penalty than easier inputs):

| snr_bucket | n | SNR(dB) | STOI | PESQ |
|---|---|---|---|---|
| −5 to 0 dB | 83 | 5.70 | 0.599 | 1.483 |
| 0 to 5 dB | 78 | 7.73 | 0.671 | 1.716 |
| 5 to 10 dB | 60 | 10.18 | 0.718 | 1.855 |
| 10 to 15 dB | 78 | 10.75 | 0.706 | 1.980 |

**What this found, stated honestly:** these are int8's own *absolute*
per-bucket scores, not a per-bucket delta against fp32 (fp32's own
per-bucket numbers were never separately recorded with this bucket
breakdown, so there's no bucket-level fp32 baseline to diff against
directly). The pattern shown — monotonically increasing SNR/STOI/PESQ with
higher input SNR — is simply the expected general trend for *any* precision
level, not evidence specific to quantization error. Given the *overall*
int8-vs-fp32 delta was already extremely small (−0.0065 SNR, −0.0024 PESQ —
an order of magnitude inside the verification script's own "small" threshold),
attributing any of that already-tiny remainder to a specific bucket, let
alone specifically to the LSTM, was explicitly judged **not supportable
from this data** — and a proper fp32-vs-int8 per-bucket delta comparison
(which would require re-running fp32 through the same bucket-reporting
code) was identified as the correct next step *if* this hypothesis were
worth pursuing further, but was not run, since the overall delta was
already far inside the acceptable range and unlikely to change the ship
decision. This is a real, disciplined instance of **checking a hypothesis
and honestly reporting an inconclusive result**, rather than either
overclaiming a finding or silently dropping the question.

### ❓ Check yourself

Given that int8 here is *weights-only* (activations stay float, weights are
dequantized back to float before each operation actually runs), why might
this explain why int8's overall delta was so much smaller than fp16's,
despite int8 being the nominally more aggressive bit-width reduction
(8 bits vs. 16)?

---

## 5. The fp32-vs-fp16 and fp32-vs-int8 decisions: measure, then decide

### 🧠 The intuition

Two different precision-reduction options were tested against the exact
same fp32 baseline, and they led to **two different real decisions** — not
because one number was arbitrarily judged "good enough" and the other
wasn't, but because each decision weighed a *measured* quality cost against
a *context-specific* benefit, and the two contexts (this project's actual
deployment target and threat model) pointed in different directions. This
is the throughline worth internalizing: **"measure then decide" beats
"assume then hope."** Nothing in this project's precision decisions was
accepted or rejected based on an assumption about what "should" be fine —
every one was backed by an isolated, quantified delta first.

### 📐 fp16: measured acceptable in isolation, rejected anyway

The verification script's own explicit thresholds for "small enough to
ignore" were: `SNR < 0.05dB, STOI < 0.005, PESQ < 0.02`. fp16's real
measured delta — **SNR −0.2472dB (≈5x over threshold), STOI −0.0007 (within
threshold), PESQ −0.0624 (≈3x over threshold)** — explicitly **failed**
this threshold on two of the three metrics. This was reported plainly as
"NOT small," not smoothed over.

**The size benefit measured alongside it:** fp32 `.mlpackage` 72MB → fp16
`.mlpackage` 54MB, a **~25% reduction** — notably less than the ~50% a naive
"half the bits, half the size" assumption would predict, because
`compute_precision=FLOAT16` changes *compute and intermediate-activation*
precision, not necessarily the width of every single stored weight/buffer
in the package — the size reduction is real but partial, reported as
measured, not assumed.

**The actual rejection reasoning (user decision, prior to the int8 step):**
FLOAT32 was locked as the shipping precision. fp16 was *not* rejected
because its quality cost looked catastrophic in isolation (a 0.25dB SNR and
0.06 PESQ hit is a real but moderate cost, not a broken model) — it was
rejected because of a **cost-benefit mismatch specific to this project's
deployment context**: the model's real, measured inference throughput on
the M4 Pro target gives roughly **~111x real-time headroom** (i.e. it
processes audio dramatically faster than the rate at which that audio
plays back) — meaning fp16's speed/size benefit is not something this
project's deployment actually *needs* to fix any real constraint. Given
that the benefit is unnecessary, there is no offsetting reason to accept
fp16's non-trivial, above-threshold quality cost — **in a defence context,
where a false or degraded transcription/enhancement carries real
consequences, "we didn't need the speed anyway" is reason enough to decline
a measured, non-negligible quality tradeoff.** This is explicitly a safety-
margin decision, not a claim that fp16 is unusable in general — a
resource-constrained deployment with genuine latency/size pressure might
reasonably make the opposite call given the identical measured numbers.

### 📐 int8: measured clean, accepted despite being nominally more aggressive

int8's real measured delta — **SNR −0.0065dB, STOI −0.0002, PESQ −0.0024**
— is comfortably **within** the same verification thresholds
(`SNR<0.05, STOI<0.005, PESQ<0.02`) on all three metrics, by roughly an
order of magnitude of margin. This is despite int8 being, on paper, a *more*
aggressive bit-width reduction than fp16 (8 bits vs. 16) — the explanation
being the weights-only mechanism discussed in Section 4 (dequantized back
to float before each operation; activations never touched).

**The size benefit:** fp32 72MB → int8 18MB, an **exactly 4.00x reduction**
(72/18 = 4.00) — a cleaner, fully-predictable result than fp16's partial
~25%, because weights-only int8 quantization directly targets the dominant
contributor to `.mlpackage` size (the stored weight tensors themselves)
rather than a `compute_precision` flag that leaves some storage widths
unaffected.

**The acceptance reasoning, and an important nuance:** int8 was *verified*
as a clean, low-cost option — but the same ~111x real-time-headroom argument
that argued against adopting fp16 *for speed/size reasons* applies equally
to int8: if there's no actual speed or size problem this project's
deployment needs solved, then int8's clean numbers don't automatically
override the fp32-locked shipping decision either. int8 is recorded as a
**measured, available, low-cost option** — should app bundle size become a
real constraint independent of inference speed (a genuinely different
concern than the runtime headroom argument), int8's near-zero quality cost
makes it a strong candidate to revisit. This is an explicitly open decision
for the user, not resolved by this analysis — measurement informs the
decision without dictating a single "correct" answer independent of what
the deployment actually needs.

### 🔗 The throughline

Both decisions used the *identical* verification methodology (same fixed-
length-crop scoring, same n=299 test split, same script-level thresholds)
and arrived at *different* real-world calls — fp16 rejected despite a
"only moderately bad" measured cost, int8 left open despite an
excellent measured cost — precisely because the decision in both cases
was made from the actual numbers plus the actual deployment context, not
from an a-priori assumption in either direction ("smaller bit-width is
always fine" or "any precision reduction is risky, don't bother measuring").
Measuring first is what made it possible to reject fp16 confidently (not
guessing it might be risky) and to leave int8 open confidently (not
guessing it might be too aggressive because "8 bits sounds more extreme
than 16").

### ❓ Check yourself

If this project's target device changed from an M4 Pro Mac (111x real-time
headroom) to a much lower-power embedded chip with, say, only 2x real-time
headroom, which of the two precision decisions (fp16 rejection, int8 left
open) would you expect to be reconsidered first, and why — based purely on
the *measured* numbers already established here, without needing to
re-measure anything?
