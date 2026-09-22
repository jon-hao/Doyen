import SwiftUI

struct ShotListView: View {
    @EnvironmentObject private var session: ShootSession

    var body: some View {
        NavigationStack {
            List {
                if let plan = session.plan {
                    Section {
                        Text("\(plan.style.rawValue) · \(plan.scene.rawValue)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text("共 \(plan.shots.count) 镜，将覆盖：\(plan.coveredTypes.joined(separator: "、"))")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Section("Shot List") {
                        ForEach(Array(plan.shots.enumerated()), id: \.element.id) { index, shot in
                            Button {
                                session.startDirecting(from: index)
                            } label: {
                                HStack(alignment: .top, spacing: 12) {
                                    Text("\(shot.id)")
                                        .font(.headline.monospacedDigit())
                                        .frame(width: 28, alignment: .leading)
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(shot.title)
                                            .foregroundStyle(.primary)
                                        Text(shot.type)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.tertiary)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("拍摄方案")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("返回") { session.reset() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("开始第 1 镜") {
                        session.startDirecting(from: 0)
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}
