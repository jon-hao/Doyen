import Foundation

enum ShootStyle: String, CaseIterable, Identifiable {
    case coupleTravel = "旅游情侣"
    case streetSoft = "日系扫街"
    case cafePortrait = "咖啡馆人像"

    var id: String { rawValue }

    var subtitle: String {
        switch self {
        case .coupleTravel: return "轻松、有故事感的一组景点人像"
        case .streetSoft: return "自然光、留白、步伐与回头"
        case .cafePortrait: return "窗光、特写、安静的半身情绪"
        }
    }
}

enum ShootScene: String, CaseIterable, Identifiable {
    case railing = "栏杆"
    case steps = "台阶"
    case trees = "树下"
    case window = "窗边"

    var id: String { rawValue }
}

struct Shot: Identifiable, Hashable {
    let id: Int
    let title: String
    let type: String
    let photographer: [String]
    let model: [String]
    let successCriteria: String
}

struct ShootPlan {
    let style: ShootStyle
    let scene: ShootScene
    let shots: [Shot]

    var coveredTypes: [String] {
        Array(Set(shots.map(\.type))).sorted()
    }
}
