import SwiftUI

struct DirectorView: View {
    @EnvironmentObject private var session: ShootSession
    @StateObject private var speech = SpeechDirector()

    var body: some View {
        NavigationStack {
            Group {
                if let shot = session.currentShot {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 24) {
                            HStack {
                                Text("\(session.progressText) 镜")
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(shot.type)
                                    .font(.caption.weight(.semibold))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(Color.secondary.opacity(0.12))
                                    .clipShape(Capsule())
                            }

                            VStack(alignment: .leading, spacing: 6) {
                                Text("第 \(shot.id) 镜")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                Text(shot.title)
                                    .font(.system(size: 32, weight: .semibold, design: .serif))
                            }

                            instructionBlock(role: "摄影师", color: .blue, lines: shot.photographer)
                            instructionBlock(role: "模特", color: .orange, lines: shot.model)

                            VStack(alignment: .leading, spacing: 8) {
                                Text("合格标准")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Text(shot.successCriteria)
                                    .font(.body)
                            }
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.secondary.opacity(0.06))
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                            CameraPlaceholderView()

                            VStack(spacing: 12) {
                                Button {
                                    speech.speakShot(shot)
                                } label: {
                                    Label("朗读本镜指令", systemImage: "speaker.wave.2.fill")
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 14)
                                }
                                .buttonStyle(.bordered)

                                Button {
                                    speech.stop()
                                    session.markCurrentShotDoneAndAdvance()
                                } label: {
                                    Text(isLastShot ? "拍完，完成本组" : "拍了，下一镜")
                                        .font(.headline)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                }
                                .buttonStyle(.borderedProminent)
                            }
                            .padding(.bottom, 12)
                        }
                        .padding(20)
                    }
                } else {
                    ContentUnavailableView("没有当前镜头", systemImage: "camera")
                }
            }
            .navigationTitle("导演")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("方案") {
                        speech.stop()
                        session.phase = .shotList
                    }
                }
            }
            .onDisappear { speech.stop() }
        }
    }

    private var isLastShot: Bool {
        guard let plan = session.plan else { return true }
        return session.currentIndex >= plan.shots.count - 1
    }

    private func instructionBlock(role: String, color: Color, lines: [String]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(role)
                .font(.subheadline.weight(.bold))
                .foregroundStyle(color)

            ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                HStack(alignment: .top, spacing: 10) {
                    Text("\(index + 1).")
                        .font(.body.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text(line)
                        .font(.body)
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}
