"""Shared constants for value probe experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueDefinition:
    """Human-readable value taxonomy entry."""

    value_id: str
    alias: str
    definition: str
    positive_examples: tuple[str, ...]
    boundary: str


VALUE_DEFINITIONS: tuple[ValueDefinition, ...] = (
    ValueDefinition(
        value_id="家庭照护",
        alias="family_care",
        definition="优先照顾同行家人、孩子、老人、伴侣的舒适、适配和特殊需求。",
        positive_examples=("带孩子", "亲子友好", "照顾老人", "伴侣状态"),
        boundary="只是和家人一起但没有照护需求时只能弱相关；不要和普通多人社交混淆。",
    ),
    ValueDefinition(
        value_id="健康克制",
        alias="health",
        definition="关注饮食健康、身体状态、低卡清淡、少油少糖、避免高负担选择。",
        positive_examples=("减脂", "清淡", "低卡", "少油"),
        boundary="好吃或精致不等于健康；偶尔放纵重口可作为 opposite。",
    ),
    ValueDefinition(
        value_id="省心便利",
        alias="convenience",
        definition="希望减少路程、等待、换乘、排队、临时不确定性和决策成本。",
        positive_examples=("离家近", "少排队", "好停车", "能预约"),
        boundary="愿意绕路、排队、慢慢逛是 opposite；不要和舒适安全混成一类。",
    ),
    ValueDefinition(
        value_id="价格敏感",
        alias="cost_sensitivity",
        definition="重视预算、性价比、优惠和价格上限。",
        positive_examples=("人均不超过", "便宜点", "性价比", "优惠"),
        boundary="愿意为品质或仪式感加钱是 opposite 或 tradeoff；高价不等于可靠。",
    ),
    ValueDefinition(
        value_id="品质可靠",
        alias="quality_reliability",
        definition="重视口碑稳定、评价可信、服务/卫生/出品稳定，避免踩雷。",
        positive_examples=("评分高", "评价稳定", "老店", "靠谱"),
        boundary="高级、贵、网红不自动等于可靠；新奇探索可能与它冲突。",
    ),
    ValueDefinition(
        value_id="体验享受",
        alias="experience_enjoyment",
        definition="重视过程好玩、沉浸、满足感、活动丰富度和主观愉悦。",
        positive_examples=("好玩", "沉浸", "体验感强", "玩得尽兴"),
        boundary="体验享受不等于新奇；熟悉但好玩的项目也可相关。",
    ),
    ValueDefinition(
        value_id="新奇探索",
        alias="novelty_exploration",
        definition="偏好新鲜、小众、未知、本地探索、尝试没去过的地方。",
        positive_examples=("新店", "小众", "没去过", "隐藏宝藏"),
        boundary="只要求靠谱稳定不算新奇；可能和品质可靠存在 tradeoff。",
    ),
    ValueDefinition(
        value_id="社交连接",
        alias="social_connection",
        definition="重视朋友/群体互动、聊天、热闹、共同参与和关系连接。",
        positive_examples=("朋友聚会", "多人互动", "热闹", "适合聊天"),
        boundary="多人同行但强调安静或各自休息时弱相关；不要和家庭照护混淆。",
    ),
    ValueDefinition(
        value_id="氛围仪式",
        alias="atmosphere_ritual",
        definition="重视氛围、浪漫、纪念意义、审美场景和仪式感。",
        positive_examples=("纪念日", "浪漫", "氛围好", "有仪式感"),
        boundary="舒适不一定有仪式感；贵也不必然有仪式感。",
    ),
    ValueDefinition(
        value_id="舒适安全",
        alias="comfort_safety",
        definition="重视身体舒适、环境安全、低风险、不过度拥挤或劳累，适合特殊人群。",
        positive_examples=("不累", "不挤", "安全", "低强度"),
        boundary="省时间属于省心便利；身体/环境风险才归舒适安全。",
    ),
)

VALUE_IDS = tuple(item.value_id for item in VALUE_DEFINITIONS)
VALUE_ALIASES = {item.alias: item.value_id for item in VALUE_DEFINITIONS}
VALUE_ALIAS_BY_ID = {item.value_id: item.alias for item in VALUE_DEFINITIONS}
VALUE_LABELS = {item.value_id: item for item in VALUE_DEFINITIONS}
VALUE_TO_INDEX = {value_id: index for index, value_id in enumerate(VALUE_IDS)}
RELATIONS = ("related", "opposite", "unrelated")

POSITIVE_RELATION = "related"
NEGATIVE_RELATION = "opposite"


def canonical_value_id(value_id: str) -> str:
    """Return the Chinese canonical value id for either new ids or legacy aliases."""

    normalized = str(value_id).strip()
    return VALUE_ALIASES.get(normalized, normalized)
