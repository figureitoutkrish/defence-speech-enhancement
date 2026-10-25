![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Core ML](https://img.shields.io/badge/Core%20ML-000000?logo=apple&logoColor=white)
![Swift](https://img.shields.io/badge/Swift-F05138?logo=swift&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)

![SNR improvement](https://img.shields.io/badge/SNR%20improvement-%2B8.28%20dB-blue)
![STOI](https://img.shields.io/badge/STOI-0.674-blue)
![PESQ](https://img.shields.io/badge/PESQ-1.786-blue)
![On-device warm](https://img.shields.io/badge/on--device%20warm-~54ms-blue)
![Params](https://img.shields.io/badge/params-18.9M-blue)

# SIH26052 — Defence-Noise Speech Enhancement

Fine-tuned real-time speech enhancement for defence communications, deployed
on-device (iOS / Core ML). Given speech recorded near gunfire, engine noise,
or other battlefield sound, the model reconstructs an estimate of the clean
voice — trained via transfer learning on top of Facebook Research's
Denoiser (`dns48`).

## The problem

Classical noise suppression (spectral subtraction, adaptive filtering)
assumes the noise floor is roughly stationary — it estimates "what the
background sounds like" from a few leading frames and subtracts that same
estimate throughout. Gunfire breaks this assumption outright: it's a sudden,
extremely loud, non-stationary burst, not a steady hum. Classical methods
built on a stationary-noise assumption degrade badly on exactly this kind of
transient noise, which is the headline noise type this problem statement
cares about.

## What this is (and isn't)

This is **speech enhancement via computational source separation**, not
acoustic ANC. Noise-cancelling headphones physically cancel sound waves in
the air, in real time, before they reach your ear (phase cancellation on a
live acoustic signal). This system does not do that, and structurally can't:
by the time a microphone captures a soldier's voice and nearby gunfire, they
are already mixed into a single waveform. Instead, the model takes that
already-mixed recording and reconstructs an estimate of the clean speech
component — a learned, data-driven reconstruction problem, not a physical
wave-cancellation trick. A full plain-language explanation of this
distinction and how the model works is in [how_it_works.md](how_it_works.md).

## Results

All numbers below are from the final locked model (**v5**), evaluated on a
299-pair, speaker-disjoint held-out test split. Per-category and
per-SNR-bucket tables are in [`results/`](results/), and the full methodology
is in the team's [development log](https://github.com/Arnav-3012/ai-ml-based-active-noise-cancellation/blob/main/logs.md).

### Overall (test split, n=299)

| | SNR (dB) | STOI | PESQ |
|---|---|---|---|
| Noisy (no processing) | 0.47 | 0.598 | 1.307 |
| Classical baseline (spectral subtraction) | 1.84 | 0.592 | 1.292 |
| **Fine-tuned v5** | **8.75** | **0.674** | **1.786** |

SNR is reported as an **improvement over the unprocessed noisy signal**
(Δ dB), not an absolute value, because "how clean is loud enough" only
makes sense relative to how noisy the input was to begin with. v5 improves
SNR by **+8.28 dB** over the noisy floor and **+6.91 dB** over the classical
baseline, while also reversing the baseline's regressions on STOI and PESQ
(the classical method actually made intelligibility and perceptual quality
*worse* than doing nothing — a known spectral-subtraction failure mode from
musical noise — the fine-tuned model does not have this problem).

### Gunshot-specific (the PS's own headline noise type, n=97)

| | SNR (dB) | STOI | PESQ |
|---|---|---|---|
| Noisy | -1.09 | 0.566 | 1.296 |
| Classical baseline | -0.12 | 0.561 | 1.297 |
| **Fine-tuned v5** | **8.90** | **0.654** | **1.870** |

The classical baseline stays SNR-negative on gunshots even after processing
— it cannot handle this noise type by design. The fine-tuned model swings
gunshot-corrupted audio from -1.09 dB to +8.90 dB, a **~10 dB improvement**,
and is the best-performing category for v5 on PESQ.

### Honest caveat: PS absolute targets not yet met

The problem statement's absolute numeric targets (SNR > 15 dB, STOI > 0.85,
PESQ > 2.5) are **not cleared** by v5, or by any of the four fine-tuning
iterations run (v1/v2/v4/v5). This is reported plainly, not smoothed over.
What v5 does demonstrate is a consistent, substantial improvement over both
the unprocessed signal and a classical baseline, across every metric and
every noise category, achieved through four methodologically-motivated
iterations (each driven by a concrete diagnosis of the previous version's
failure mode, not blind hyperparameter search — see
[docs/phase3_training_deep_dive.md](docs/phase3_training_deep_dive.md)). Closing the remaining gap to the PS's absolute targets
was explicitly scoped out as further work once v5 was locked, not attempted
or claimed here.

### On-device (verified, iPhone hardware, int8 Core ML build)

| | Time |
|---|---|
| Cold start (first inference after launch) | ~274 ms |
| **Warm inference** (subsequent calls, same model instance) | **~53–55 ms** |

The app performs a silent warmup inference at launch specifically so the
first real recording a user makes measures the warm number, not the cold
one. At ~54 ms to process a 4.0 s audio window, that's roughly a
**74× real-time factor** — inference finishes far faster than the audio it's
processing, leaving substantial headroom for a future streaming
implementation (see [On-device demo](#on-device-demo) below).

## Architecture

**dns48** (Facebook Research `denoiser`), fine-tuned via transfer learning —
never trained from scratch, by design.

- **18,867,937 parameters** (exact count, not rounded).
- Encoder (5 layers) → 2-layer LSTM bottleneck (hidden size 768) → decoder
  (5 layers), with skip connections between matching encoder/decoder stages.
- **Causal by construction**: every output sample depends only on past
  input, never future input — a real-time-capable design constraint, not
  an afterthought, since a radio operator can't wait for audio that hasn't
  been spoken yet.
- The model predicts a soft mask over its internal representation rather
  than generating new audio outright, which bounds failure modes to
  muffling/over-suppression rather than hallucinated content.

Fine-tuning used an L1 + multi-resolution STFT loss, Adam at a low
(1/10th-of-typical) learning rate to avoid catastrophically forgetting
dns48's pretrained weights, and MPS (Apple Silicon) acceleration on an
M4 Pro, with an explicit, non-silent CPU fallback if MPS is unavailable.

## How it was built

- **Dataset construction**: clean speech (LibriSpeech dev-clean, 40
  speakers) mixed with real noise (MUSAN stationary/general noise,
  UrbanSound8K + Kabealo gunshot recordings — 2,522 gunshot clips combined)
  at SNR-targeted ratios spanning **-5 dB to 15 dB**, plus reverb/clipping
  augmentation. Split **speaker-disjoint**, 80/10/10 — verified
  programmatically to have zero speaker-ID overlap across train/val/test,
  so the model is never evaluated on a voice it trained on.
- **Classical baseline**: a hand-built STFT spectral-subtraction
  implementation (Boll 1979 noise estimate + Berouti et al. 1979
  oversubtraction/spectral floor), not a third-party library. It exists to
  demonstrate the stationary-noise assumption's real failure mode on
  transient noise like gunfire, which it does.
- **Training iteration (v1 → v5)**: four fine-tuning runs, each motivated by
  a concrete failure diagnosis of the previous version (over-suppression →
  curriculum learning + silence-collapse penalty → dynamic per-step mixing
  + reactive LR), not a blind hyperparameter sweep. v5 is the final, locked
  model. Full iteration-by-iteration story: [docs/phase3_training_deep_dive.md](docs/phase3_training_deep_dive.md)
  and the team's [development log](https://github.com/Arnav-3012/ai-ml-based-active-noise-cancellation/blob/main/logs.md).
- **Export/quantization pipeline**: PyTorch → ONNX (verified zero export
  drift, isolated from a fixed-length-crop confound) → Core ML
  (`mlprogram`, iOS15 target, verified zero drift vs. ONNX at fp32) → int8
  weights-only quantization (4.00× size reduction, 72MB → 18MB, rounding-
  level quality delta). fp32 is the locked shipping precision for the
  underlying model; fp16 was measured and rejected for a non-trivial
  quality cost given ample real-time headroom; the deployed iOS app uses
  the **int8** build for its favorable size/quality tradeoff. Full details,
  including two real conversion failures that were diagnosed and fixed
  (ONNX-to-Core-ML has no direct converter path in this coremltools
  version; an LSTM shape-tracing crash traced to Python-scalar shape reads
  inside a traced graph) are in
  [docs/export_quantization_deep_dive.md](docs/export_quantization_deep_dive.md).

## On-device demo

`ANCDemo/` is a standalone iOS app (SwiftUI) proving the exported int8
Core ML model runs real inference on real device hardware, end to end:
record audio → run on-device inference → play back original vs. enhanced
for A/B comparison → **Clean & Share** via iOS's native Share Sheet
(surfacing AirDrop automatically). This is a **Tier 1 / Tier 1.5** demo:
record-then-process, not continuous streaming. **Live, continuous
streaming inference (Tier 2) was not built.** The model's causal
architecture and ~54 ms warm-inference time (well under the 250 ms window
of a 4 s chunk) make streaming architecturally feasible as a next step, but
implementing a real-time audio pipeline (chunked buffering, low-latency
I/O, continuous Core ML invocation) was out of scope for this build.

## Repo structure

```
├── src/                    # Pipeline source of truth
│   ├── data/                #   dataset download + SNR mixing + augmentation
│   ├── baseline/             #   hand-built spectral subtraction
│   ├── model/                #   dns48 loading, dataset, fine-tuning loop, inference
│   ├── eval/                 #   SNR/STOI/PESQ metrics + evaluation scripts
│   ├── export/                #   ONNX / Core ML export + quantization + verification
│   └── vendor/denoiser_patched/  # minimal local patches to the denoiser pip package
├── configs/finetune.yaml   # single source of truth for every tunable
├── manifests/               # train/val/test pair manifests (JSON)
├── checkpoints/              # .pt / onnx / coreml model artifacts (gitignored .pt/.onnx)
├── results/                  # metrics CSVs/JSONs, training logs, verification reports
├── dashboard/app.py          # standalone Streamlit results viewer (outside src/, reads-only)
├── ANCDemo/                  # iOS SwiftUI demo app (Xcode project)
├── docs/phase3_training_deep_dive.md  # full v1→v5 training iteration story
├── notebooks/                # scratch/exploration only, never imported by src/
├── how_it_works.md           # plain-language explainer of the whole system
├── dataset.md                # every training setting in finetune.yaml, explained
├── phase2_baseline_and_eval.md  # the classical baseline and the metrics, explained
└── requirements.txt
```

## Setup / reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data pipeline (download the Kabealo gunshot dataset by hand first, see below)
python -m src.data.download
python -m src.data.build_dataset

# Classical baseline + evaluation
python -m src.baseline.spectral_subtraction
python -m src.eval.evaluate

# Fine-tuning (uses configs/finetune.yaml's training_run_v5 section for the final model)
python -m src.model.finetune --run

# Inference + evaluation of the fine-tuned model
python -m src.model.inference
python -m src.eval.evaluate

# Export
python -m src.export.to_onnx
python -m src.export.to_coreml
python -m src.export.quantize_coreml
```

Datasets: LibriSpeech dev-clean, MUSAN and UrbanSound8K are fetched by
`src.data.download`. The Kabealo gunshot dataset
([Zenodo record 7004819](https://zenodo.org/records/7004819)) has to be
downloaded by hand into `data/raw/gunshots/`.

The Streamlit results dashboard (reads only already-computed files under
`results/`, does not run training/inference itself):

```bash
streamlit run dashboard/app.py
```

The iOS demo app is a standard Xcode project at `ANCDemo/ANCDemo.xcodeproj`
— open and run on a physical device (Core ML on-device inference does not
run meaningfully in the Simulator).

## References

- Defossez, A., Synnaeve, G., Adi, Y. — *Real Time Speech Enhancement in the
  Waveform Domain* (Denoiser), Interspeech 2020.
- Luo, Y., Mesgarani, N. — *Conv-TasNet: Surpassing Ideal Time-Frequency
  Magnitude Masking for Speech Separation*, IEEE/ACM TASLP 2019.
- Panayotov, V. et al. — *LibriSpeech: An ASR corpus based on public domain
  audio books*, ICASSP 2015.
- Snyder, D., Chen, G., Povey, D. — *MUSAN: A Music, Speech, and Noise
  Corpus*, 2015.
- Piczak, K. J. — *ESC: Dataset for Environmental Sound Classification*,
  ACM Multimedia 2015.
- Kabealo, W. et al. — *A multi-firearm, multi-orientation audio dataset of
  gunshots*, Data in Brief, 2023 (Zenodo record 7004819).
- Taal, C. H. et al. — *An Algorithm for Intelligibility Prediction of
  Time-Frequency Weighted Noisy Speech* (STOI), IEEE TASLP 2011.
- Rix, A. W. et al. — *Perceptual Evaluation of Speech Quality (PESQ)*,
  ICASSP 2001.
- Boll, S. — *Suppression of Acoustic Noise in Speech Using Spectral
  Subtraction*, IEEE TASSP 1979.
- Berouti, M., Schwartz, R., Makhoul, J. — *Enhancement of Speech
  Corrupted by Acoustic Noise*, ICASSP 1979.

## Team / acknowledgments

Built by **Team Gradient Descent** for **Smart India Hackathon 2026**,
Problem Statement **SIH26052**:

- [Arnav Chaturvedi](https://github.com/Arnav-3012)
- [Krish Viramgama](https://github.com/figureitoutkrish)
- [Rushil](https://github.com/Rushil0700)

The team's original repository, with the full development log, is
[Arnav-3012/ai-ml-based-active-noise-cancellation](https://github.com/Arnav-3012/ai-ml-based-active-noise-cancellation).
