import SwiftUI

struct RootView: View {
    @EnvironmentObject private var session: ShootSession

    var body: some View {
        Group {
            switch session.phase {
            case .setup:
                SetupView()
            case .shotList:
                ShotListView()
            case .directing:
                DirectorView()
            case .complete:
                SessionCompleteView()
            }
        }
        .animation(.easeInOut(duration: 0.25), value: session.phase)
    }
}
