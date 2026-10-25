//
//  AudioRecorder.swift
//  ANCDemo
//
//  Stage 2 recording: AVAudioRecorder (not AVAudioEngine) -- chosen because
//  this app only needs "record to a file, then read it back once recording
//  stops," which is exactly AVAudioRecorder's use case. AVAudioEngine is
//  built for real-time streaming/tap access to live buffers, which this
//  project explicitly does NOT need yet (real-time streaming inference is
//  out of scope per the development log's "Explicitly out of scope" list) --
//  AVAudioEngine would be the right tool if/when a live-streaming Stage
//  existed, not for this record-then-process flow.
//
//  Records directly at 16kHz mono Float32 PCM (matching the model's
//  expected input format exactly) via AVAudioRecorder's settings dict, so
//  no post-recording sample-rate/channel conversion step is needed at all
//  -- Stage 1's AVAudioConverter path in ModelRunner.swift stays as
//  dead-but-correct code for the bundled test .wav (which is also already
//  16kHz mono, so that path isn't exercised in practice either) and would
//  only be exercised if a future input source arrives in a different
//  native format.

import Foundation
import AVFoundation
import Combine

enum AudioRecorderError: Error, CustomStringConvertible {
    case permissionDenied
    case recorderSetupFailed(String)
    case noRecordingAvailable
    case readFailed(String)

    var description: String {
        switch self {
        case .permissionDenied: return "microphone permission was denied"
        case .recorderSetupFailed(let msg): return "failed to set up AVAudioRecorder: \(msg)"
        case .noRecordingAvailable: return "no recording available to read"
        case .readFailed(let msg): return "failed to read recorded audio: \(msg)"
        }
    }
}

@MainActor
final class AudioRecorder: NSObject, ObservableObject {

    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0

    private var recorder: AVAudioRecorder?
    private var durationTimer: Timer?

    /// Kept internal-readable (not private) so ContentView can offer
    /// "Play Raw" against the exact file that was recorded, using the
    /// same URL stopAndReadSamples() reads from -- one source of truth,
    /// not a second copy.
    private(set) var recordingURL: URL?

    /// Same target format as ModelRunner's model input contract --
    /// 16kHz mono Float32 PCM -- recorded directly, no conversion needed.
    private static let recordSettings: [String: Any] = [
        AVFormatIDKey: Int(kAudioFormatLinearPCM),
        AVSampleRateKey: ModelRunner.expectedSampleRate,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
    ]

    /// Requests microphone permission if not already determined, then
    /// starts recording to a fresh temp file. Throws .permissionDenied if
    /// the user has denied access (caller should surface this in the UI).
    func requestPermissionAndStart() async throws {
        let granted = await requestPermission()
        guard granted else {
            throw AudioRecorderError.permissionDenied
        }
        try start()
    }

    private func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func start() throws {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
        } catch {
            throw AudioRecorderError.recorderSetupFailed("AVAudioSession activation failed: \(error)")
        }

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("recording_\(UUID().uuidString).wav")
        do {
            let newRecorder = try AVAudioRecorder(url: url, settings: Self.recordSettings)
            newRecorder.delegate = self
            guard newRecorder.record() else {
                throw AudioRecorderError.recorderSetupFailed("AVAudioRecorder.record() returned false")
            }
            recorder = newRecorder
            recordingURL = url
        } catch {
            throw AudioRecorderError.recorderSetupFailed("\(error)")
        }

        isRecording = true
        recordingDuration = 0
        durationTimer?.invalidate()
        durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let recorder = self.recorder else { return }
                self.recordingDuration = recorder.currentTime
            }
        }
    }

    /// Stops recording and returns the raw Float32 samples read back from
    /// the recorded file, at the recorder's native 16kHz mono format.
    /// Fitting to the model's exact 64000-sample contract happens in
    /// ModelRunner.process(rawSamples:), reusing the SAME
    /// fitToFixedLength() used by Stage 1 -- not reimplemented here.
    func stopAndReadSamples() throws -> [Float] {
        durationTimer?.invalidate()
        durationTimer = nil
        recorder?.stop()
        isRecording = false

        guard let url = recordingURL else {
            throw AudioRecorderError.noRecordingAvailable
        }

        do {
            let file = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(file.length)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
                throw AudioRecorderError.readFailed("could not allocate PCM buffer")
            }
            try file.read(into: buffer)
            guard let channelData = buffer.floatChannelData else {
                throw AudioRecorderError.readFailed("recorded file has no float channel data")
            }
            let frameLength = Int(buffer.frameLength)
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        } catch {
            throw AudioRecorderError.readFailed("\(error)")
        }
    }
}

extension AudioRecorder: @preconcurrency AVAudioRecorderDelegate {
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if !flag {
            print("AudioRecorder: recording finished unsuccessfully.")
        }
    }
}
