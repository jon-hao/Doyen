import SwiftUI

struct SetupView: View {
    @EnvironmentObject private var session: ShootSession

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Doyen")
                            .font(.system(size: 40, weight: .semibold, design: .serif))
                        Text("第一次拍照，也能像导演一样出片")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 8)

                    section(title: "风格") {
                        ForEach(ShootStyle.allCases) { style in
                            selectRow(
                                title: style.rawValue,
                                subtitle: style.subtitle,
                                selected: session.selectedStyle == style
                            ) {
                                session.selectedStyle = style
                            }
                        }
                    }

                    section(title: "场景") {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                            ForEach(ShootScene.allCases) { scene in
                                Button {
                                    session.selectedScene = scene
                                } label: {
                                    Text(scene.rawValue)
                                        .font(.headline)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .background(
                                            session.selectedScene == scene
                                            ? Color.primary.opacity(0.12)
                                            : Color.secondary.opacity(0.08)
                                        )
                                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    Button {
                        session.generatePlan()
                    } label: {
                        Text("生成拍摄方案")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 8)
                }
                .padding(20)
            }
            .navigationTitle("开场")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func section(title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
    }

    private func selectRow(
        title: String,
        subtitle: String,
        selected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                }
                Spacer()
                if selected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.primary)
                }
            }
            .padding(14)
            .background(selected ? Color.primary.opacity(0.08) : Color.secondary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
    }
}
