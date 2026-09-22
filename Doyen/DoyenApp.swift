import SwiftUI

@main
struct DoyenApp: App {
    @StateObject private var session = ShootSession()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
        }
    }
}
