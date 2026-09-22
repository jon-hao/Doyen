import SwiftUI

struct CameraPlaceholderView: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.black.opacity(0.85))
                .frame(height: 220)

            VStack(spacing: 10) {
                Image(systemName: "camera.fill")
                    .font(.system(size: 28))
                Text("相机取景（下一版接入）")
                    .font(.subheadline.weight(.medium))
                Text("现在先按指令摆位，用系统相机拍摄也可")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }
            .foregroundStyle(.white)
        }
    }
}
