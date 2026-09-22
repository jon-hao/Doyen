import SwiftUI

struct SessionCompleteView: View {
    @EnvironmentObject private var session: ShootSession

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 24) {
                Spacer()

                Text("本组完成")
                    .font(.system(size: 36, weight: .semibold, design: .serif))

                if let plan = session.plan {
                    Text("本次拍摄 \(plan.shots.count)/\(plan.shots.count) 镜完成")
                        .font(.title3)

                    VStack(alignment: .leading, spacing: 8) {
                        Text("已覆盖")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(plan.coveredTypes.joined(separator: "、"))
                            .font(.body)
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.secondary.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }

                Spacer()

                Button {
                    session.reset()
                } label: {
                    Text("再拍一组")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(24)
            .navigationTitle("收工")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
