import Foundation

enum SampleShootPlans {
    static func make(style: ShootStyle, scene: ShootScene) -> ShootPlan {
        let shots: [Shot]

        switch (style, scene) {
        case (.coupleTravel, .railing), (.streetSoft, .railing):
            shots = railingShots
        case (_, .steps):
            shots = stepsShots
        case (_, .trees):
            shots = treesShots
        case (_, .window), (.cafePortrait, _):
            shots = windowShots
        default:
            shots = railingShots
        }

        return ShootPlan(style: style, scene: scene, shots: shots)
    }

    private static let railingShots: [Shot] = [
        Shot(
            id: 1,
            title: "环境定场",
            type: "环境人像",
            photographer: ["让她站到栏杆右侧", "你后退约 1.2 米", "手机切到 2×", "把她放到画面右侧三分之一"],
            model: ["侧身靠近栏杆", "身体放松，先不要看镜头"],
            successCriteria: "能看清环境，人物约占画面 1/3，不裁脚"
        ),
        Shot(
            id: 2,
            title: "身体转向",
            type: "半身",
            photographer: ["保持 2×", "稍微靠近半步", "构图留出左侧负空间"],
            model: ["身体向左转约 30 度", "肩膀放松", "下巴微收"],
            successCriteria: "半身清晰，侧脸或 3/4 脸，背景不抢戏"
        ),
        Shot(
            id: 3,
            title: "看向远处",
            type: "留白",
            photographer: ["主体仍在右三分之一", "多留天空或远景", "等表情自然再拍"],
            model: ["看向远处", "眼神放软", "嘴角自然"],
            successCriteria: "有留白，眼神不僵，不居中"
        ),
        Shot(
            id: 4,
            title: "沿栏杆走",
            type: "动态",
            photographer: ["保持 2×", "跟着她向后移动", "预留行走空间", "等步伐顺了再按"],
            model: ["沿着栏杆慢慢走", "步伐自然，不要迈太大"],
            successCriteria: "有动态感，人物完整，不糊到脸"
        ),
        Shot(
            id: 5,
            title: "回头",
            type: "回头",
            photographer: ["保持跟随", "等她回头的瞬间", "按快门要果断"],
            model: ["走到第三步时回头看镜头", "只转头，身体继续朝前"],
            successCriteria: "回头自然，眼神接到镜头，半身以上清楚"
        ),
        Shot(
            id: 6,
            title: "侧脸特写",
            type: "特写",
            photographer: ["切回 1× 或靠近", "对焦眼睛", "注意不要逆光死黑"],
            model: ["侧脸", "视线看画面外一点", "头发拨顺"],
            successCriteria: "眼睛清晰，脸部曝光正常，不裁顶"
        ),
        Shot(
            id: 7,
            title: "低机位",
            type: "低机位",
            photographer: ["蹲下", "让天空约占画面 40%", "从下往上找线条"],
            model: ["站直", "稍微抬头看远", "手可轻扶栏杆"],
            successCriteria: "低机位明显，天空有比例，人物不变形过度"
        ),
        Shot(
            id: 8,
            title: "前景层次",
            type: "前景",
            photographer: ["用栏杆做前景虚化", "主体仍清晰", "检查边缘是否干净"],
            model: ["站到稍远一点", "身体侧 45 度", "看镜头微笑"],
            successCriteria: "有前景层次，主体清楚，画面不杂乱"
        ),
        Shot(
            id: 9,
            title: "对称站位",
            type: "对称",
            photographer: ["找栏杆重复线条", "人物可略偏，不要死盯正中", "水平线摆正"],
            model: ["双手轻搭栏杆", "身体正对或微侧", "表情平静"],
            successCriteria: "线条稳定，人物站位明确，地平线水平"
        ),
        Shot(
            id: 10,
            title: "收尾半身",
            type: "半身",
            photographer: ["回到舒适半身距离", "确认整组不重复上一张表情", "拍完说一声收工"],
            model: ["自然看镜头", "换一个和前面不同的手势", "放松肩膀"],
            successCriteria: "半身完整，表情与前面有区别"
        )
    ]

    private static let stepsShots: [Shot] = [
        Shot(
            id: 1,
            title: "台阶全景",
            type: "环境人像",
            photographer: ["退到能看见 3–4 级台阶", "人物放右三分之一", "保持水平"],
            model: ["站在台阶中段", "侧身，手轻扶或自然下垂"],
            successCriteria: "台阶线条清楚，人物完整"
        ),
        Shot(
            id: 2,
            title: "坐阶半身",
            type: "半身",
            photographer: ["略俯拍", "膝盖不要裁奇怪位置", "对焦脸"],
            model: ["坐在台阶上", "身体微侧", "看镜头"],
            successCriteria: "半身自然，坐姿不僵"
        ),
        Shot(
            id: 3,
            title: "向上看",
            type: "低机位",
            photographer: ["再低一点", "让台阶引导线指向人物", "天空留白"],
            model: ["站在较高一级", "轻轻抬头"],
            successCriteria: "有纵深，低机位明确"
        ),
        Shot(
            id: 4,
            title: "走下台阶",
            type: "动态",
            photographer: ["预对焦中段", "跟着移动", "步伐稳定时拍"],
            model: ["慢慢走下两三级", "不要看脚"],
            successCriteria: "有动态，脸清楚"
        ),
        Shot(
            id: 5,
            title: "回头阶上",
            type: "回头",
            photographer: ["等回头瞬间", "背景用台阶重复纹理"],
            model: ["走两步后回头", "微笑"],
            successCriteria: "回头自然，环境可读"
        ),
        Shot(
            id: 6,
            title: "侧脸近景",
            type: "侧脸",
            photographer: ["靠近", "虚化台阶背景", "注意下巴线条"],
            model: ["侧脸", "目光看斜下方"],
            successCriteria: "侧脸清晰，背景不抢"
        ),
        Shot(
            id: 7,
            title: "特写眼神",
            type: "特写",
            photographer: ["对眼睛", "切 2×", "曝光看脸"],
            model: ["看镜头", "表情收一点"],
            successCriteria: "眼神清楚，不晃"
        ),
        Shot(
            id: 8,
            title: "留白站姿",
            type: "留白",
            photographer: ["人物偏一侧", "多留天空或墙面", "别居中"],
            model: ["站直放松", "看远处"],
            successCriteria: "留白明显，情绪安静"
        )
    ]

    private static let treesShots: [Shot] = [
        Shot(
            id: 1,
            title: "树荫定场",
            type: "环境人像",
            photographer: ["寻找斑驳光", "人物放在光斑稳定处", "避免花屏高光"],
            model: ["站到树侧", "身体微侧"],
            successCriteria: "环境有树，脸部曝光可读"
        ),
        Shot(
            id: 2,
            title: "半身光影",
            type: "半身",
            photographer: ["让光线落在脸的 3/4", "背景虚化树叶"],
            model: ["转向有光的一侧", "下巴微收"],
            successCriteria: "半身，光在脸上"
        ),
        Shot(
            id: 3,
            title: "行走树下",
            type: "动态",
            photographer: ["横向跟随", "快门时机选步伐中间"],
            model: ["慢慢走过两棵树之间"],
            successCriteria: "有动态，不糊脸"
        ),
        Shot(
            id: 4,
            title: "回头树间",
            type: "回头",
            photographer: ["预留回头空间", "按得干脆"],
            model: ["走两步回头看你"],
            successCriteria: "回头自然"
        ),
        Shot(
            id: 5,
            title: "特写",
            type: "特写",
            photographer: ["靠近或 2×", "对焦眼睛", "避开乱枝穿过脸"],
            model: ["看镜头", "表情柔和"],
            successCriteria: "脸干净，眼睛清"
        ),
        Shot(
            id: 6,
            title: "低机位树冠",
            type: "低机位",
            photographer: ["蹲下", "树冠约占 40%", "人物抬头"],
            model: ["抬头看树叶间隙"],
            successCriteria: "低机位 + 树冠比例"
        ),
        Shot(
            id: 7,
            title: "前景树叶",
            type: "前景",
            photographer: ["用近处叶子挡一点角", "主体仍清楚"],
            model: ["站远半步", "看镜头"],
            successCriteria: "有前景层次"
        ),
        Shot(
            id: 8,
            title: "侧脸安静",
            type: "侧脸",
            photographer: ["侧光方向拍摄", "简洁背景"],
            model: ["侧脸", "目光离开镜头"],
            successCriteria: "侧脸情绪稳定"
        )
    ]

    private static let windowShots: [Shot] = [
        Shot(
            id: 1,
            title: "窗边环境",
            type: "环境人像",
            photographer: ["让窗户成为光源与背景", "人物不要完全贴窗死黑"],
            model: ["侧身靠窗", "手可扶窗台"],
            successCriteria: "窗光可读，人物完整"
        ),
        Shot(
            id: 2,
            title: "窗光半身",
            type: "半身",
            photographer: ["脸朝窗户", "曝光按脸计量"],
            model: ["身体微侧向窗", "看镜头"],
            successCriteria: "半身，脸亮背景可控"
        ),
        Shot(
            id: 3,
            title: "看向窗外",
            type: "留白",
            photographer: ["人物偏一侧", "窗外或墙面留白"],
            model: ["看向窗外", "表情安静"],
            successCriteria: "有留白，情绪在"
        ),
        Shot(
            id: 4,
            title: "特写",
            type: "特写",
            photographer: ["靠近", "对眼睛", "注意鼻影别太重"],
            model: ["微微低头再抬眼"],
            successCriteria: "特写清晰"
        ),
        Shot(
            id: 5,
            title: "侧脸窗光",
            type: "侧脸",
            photographer: ["让窗光勾脸轮", "背景简化"],
            model: ["纯侧脸", "肩放松"],
            successCriteria: "侧脸轮廓清楚"
        ),
        Shot(
            id: 6,
            title: "低机位",
            type: "低机位",
            photographer: ["稍蹲", "窗户线条做引导"],
            model: ["站直", "轻轻低头看镜头"],
            successCriteria: "低机位有变化"
        ),
        Shot(
            id: 7,
            title: "对称窗框",
            type: "对称",
            photographer: ["用窗框找结构", "水平垂直摆正"],
            model: ["站在窗中线附近", "姿势简单"],
            successCriteria: "结构稳定"
        ),
        Shot(
            id: 8,
            title: "收尾半身",
            type: "半身",
            photographer: ["回到舒服距离", "换一个表情结束"],
            model: ["微笑看镜头", "换手势"],
            successCriteria: "收束自然"
        )
    ]
}
