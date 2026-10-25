# iOS App — Brief Reference

A concise reference for the ANCDemo iOS app (`ANCDemo/`). This is a
reference doc, not a tutorial — the deep technical content behind the model
itself lives in `docs/export_quantization_deep_dive.md`; this covers only
what's specific to getting that model running inside an iOS app.

---

## What Core ML / `.mlpackage` is, briefly

Core ML is Apple's on-device machine learning runtime — it takes a
converted model (a `.mlpackage`, the modern Core ML container format,
containing the model graph plus its weights) and runs inference locally on
the device's CPU/GPU/Neural Engine, with no network call and no server.
Xcode automatically generates a typed Swift interface from a `.mlpackage`
added to a project at build time — for this project's shipped model,
`dns48_finetuned_v5_int8.mlpackage`, Xcode's `coremlc` tool generates a
Swift class (named after the file, sanitized for Swift identifier rules)
exposing a `prediction(noisy_waveform:) throws -> ...Output` method. The app
calls this directly — no manual graph loading, no custom runtime code.

The model's exact input contract (from `ModelRunner.swift`, verified
against `src/export/traceable_demucs.py`, `to_coreml.py`, and
`verify_onnx.py`'s `_fit_to_fixed_length`, not assumed): 16kHz mono Float32
PCM, exactly 64,000 samples (4.0s), tensor name `noisy_waveform` in,
`enhanced_waveform` out, shape `(1, 1, 64000)`. The traced/converted graph
is **only valid at this exact length** — this is a direct consequence of
the fixed-length export decision covered in depth in the export doc; any
other length silently produces wrong output, since the graph's internal
padding math is hardcoded for this one shape.

## The Xcode/device pairing chain — quick reference

Getting a locally-built app running on a physical iPhone (rather than the
simulator) requires a chain of trust to be established correctly, and each
link in that chain is a real place things can silently fail:

1. **Developer Mode** must be enabled on the physical device itself
   (Settings → Privacy & Security → Developer Mode) — without it, iOS
   refuses to run locally-signed (non-App-Store) builds at all, regardless
   of whether Xcode successfully installs them.
2. **Code signing** requires a valid signing identity tied to an Apple
   Developer account. This project's `project.pbxproj` uses
   `CODE_SIGN_STYLE = Automatic` with a real `DEVELOPMENT_TEAM` ID
   (`MGWJ67MC23`) and bundle identifier `com.arnav.anc.ANCDemo` — automatic
   signing lets Xcode manage the provisioning profile itself rather than
   requiring a manually-created one, which is the right default for a
   single-developer demo app like this one.
3. **Keychain access** — Xcode needs access to the signing certificate's
   private key, stored in the local macOS Keychain, to actually sign the
   build. A Keychain access prompt denied or a missing/expired certificate
   here breaks the chain even when the team ID and bundle ID are both
   correct.
4. **Device trust** — the first time a locally-signed app from a given
   developer is run on a device, iOS requires the user to explicitly trust
   that developer's certificate (Settings → General → VPN & Device
   Management) before the app is allowed to launch, even after a
   successful install.

Each of these four links is independent — a failure at any one of them
produces a different, sometimes cryptic, Xcode error, and getting a fresh
device paired correctly typically means walking all four in order. This
cost real debugging time on this project and is recorded here so it doesn't
need to be re-derived from scratch on the next device.

## The four build stages

**Stage 1 — model load & verify.** Confirmed the shipped
`dns48_finetuned_v5_int8.mlpackage` actually loads on-device and produces
correct-shaped output, using a bundled test file (`test_gunshot_noisy.wav`)
rather than live recording — isolating "does the model work on this
device at all" from any recording/UI code. Ran 4 back-to-back prediction
calls on the same model instance and logged each one's timing
(`ModelRunner.runVerification()`), which is where this project's real
cold/warm latency numbers (below) were first established on a physical
device. Kept in the codebase as a manually-invokable diagnostic, not wired
to app launch, in case the cold/warm pattern ever needs re-checking after a
model or device change.

**Stage 2 — record → infer → play.** Built the real user-facing flow:
`AudioRecorder` (via `AVAudioRecorder`, not `AVAudioEngine` — deliberately
chosen because this app only needs "record to a file, then read it back
once," not live streaming access, and real-time streaming inference is
explicitly out of scope per the development log) records directly at 16kHz mono
Float32, matching the model's input format exactly with no conversion step
needed. `ModelRunner.process(rawSamples:)` reuses the *same* shared model
instance and the *same* `fitToFixedLength` fit-to-64000 logic Stage 1
verified, rather than a second implementation. This stage also introduced
the **silent launch warmup** (see below) and Raw/Enhanced A/B playback via
`AudioPlayer`, so a listener can hear the noisy original and the cleaned
output side by side.

**Stage 3 — Clean & Share / AirDrop.** Added a "Clean & Share" button that
presents iOS's native Share Sheet (`UIActivityViewController`, wrapped for
SwiftUI in `ShareSheet.swift`) with the enhanced audio file. This
automatically surfaces AirDrop as a share target — along with Messages,
Mail, Save to Files, etc. — with no AirDrop-specific code required; showing
the Share Sheet with a file URL is sufficient, iOS handles the rest. The
enhanced file is also given a human-readable, timestamped filename
(`cleaned_speech_<timestamp>.wav`) rather than a raw UUID, specifically so
it's recognizable once it lands on the receiving device.

**Stage 4 — UI polish.** Status-state UI (`Ready`/`Recording`/`Processing`/
`Done`/`Error`, each with distinct color), a pulsing ring animation while
actively recording, a live recording-duration readout, an inference-time
readout shown to the user after each recording, and per-button
play/stop state management in `AudioPlayer` that guarantees only one of
Raw/Enhanced can play at a time (stopping whichever is active before
starting the other) — preventing the two tracks from overlapping and
sounding "confusing," a real usability issue found and fixed during this
stage. This stage proved the app is coherent as an end-to-end demo
experience, not just functionally correct.

## Real on-device latency numbers, and the silent-warmup decision

**Measured directly on-device (Stage 1, 4 back-to-back calls, same model
instance, same input):** a clear cold/warm pattern — **~274ms for the first
call**, dropping to **~53–55ms for every call after** on the same shape.
This mirrors the identical cold/warm pattern already found on the Mac
during Phase 3a (per-shape MPS kernel compilation cost, paid once per
unique input length, not a global one-time cost) — the same underlying
mechanism (a first-touch compilation/dispatch cost for a given tensor
shape) shows up on both the Mac's MPS backend and the iPhone's Core ML
runtime.

**Design decision — silent launch warmup:** since the app's real input
length is fixed at exactly 64,000 samples (Section above), the cold-start
cost is entirely predictable and always attributable to the *same* shape
every time. `ModelRunner.warmup()` runs one throwaway prediction call on
the bundled Stage 1 test file the moment the app launches, discards the
output, and does so invisibly — `ContentView` calls it from a background
`Task.detached` in `.task {}`, with no UI tied to it. Because the warmup
call uses the exact same shared model instance the real recording flow
later reuses (a `lazy static`, initialized thread-safely exactly once by
Swift itself), the cold-start cost is absorbed *before* the user ever taps
record — meaning the first real recording's measured and displayed
inference time reflects the warm ~53–55ms number, not the ~274ms cold
number. This directly shapes what a live demo audience sees: without the
warmup, the very first demo recording would show a misleadingly slow
number that has nothing to do with the model's real steady-state speed.

The warmup deliberately reuses the Stage 1 bundled test file rather than a
synthetic buffer — it's already verified to load/fit/convert correctly end
to end, and the model only cares about input *shape* for warmup purposes,
not content, so a synthetic buffer would add one more untested code path
for no benefit.
