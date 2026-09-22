import AVFoundation
import Foundation

@MainActor
final class SpeechDirector: ObservableObject {
    private let synthesizer = AVSpeechSynthesizer()

    func speakShot(_ shot: Shot) {
        stop()

        var lines: [String] = ["第\(shot.id)镜，\(shot.title)。"]
        lines.append("摄影师：")
        lines.append(contentsOf: shot.photographer)
        lines.append("模特：")
        lines.append(contentsOf: shot.model)

        let utterance = AVSpeechUtterance(string: lines.joined(separator: " "))
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.92
        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}
