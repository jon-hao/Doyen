import Foundation
import Combine

@MainActor
final class ShootSession: ObservableObject {
    enum Phase: Equatable {
        case setup
        case shotList
        case directing
        case complete
    }

    @Published var phase: Phase = .setup
    @Published var selectedStyle: ShootStyle = .coupleTravel
    @Published var selectedScene: ShootScene = .railing
    @Published var plan: ShootPlan?
    @Published var currentIndex: Int = 0
    @Published var completedShotIDs: Set<Int> = []

    var currentShot: Shot? {
        guard let plan, plan.shots.indices.contains(currentIndex) else { return nil }
        return plan.shots[currentIndex]
    }

    var progressText: String {
        guard let plan else { return "0/0" }
        return "\(completedShotIDs.count)/\(plan.shots.count)"
    }

    func generatePlan() {
        plan = SampleShootPlans.make(style: selectedStyle, scene: selectedScene)
        currentIndex = 0
        completedShotIDs = []
        phase = .shotList
    }

    func startDirecting(from index: Int = 0) {
        currentIndex = index
        phase = .directing
    }

    func markCurrentShotDoneAndAdvance() {
        guard let shot = currentShot, let plan else { return }
        completedShotIDs.insert(shot.id)

        if currentIndex + 1 < plan.shots.count {
            currentIndex += 1
        } else {
            phase = .complete
        }
    }

    func reset() {
        phase = .setup
        plan = nil
        currentIndex = 0
        completedShotIDs = []
    }
}
