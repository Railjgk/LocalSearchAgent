"""B-stage itinerary blueprint extraction.

This module does not retrieve or rank POIs. It only turns a Chinese local-life
request into a small planning skeleton so B can distinguish the current legacy
half-day pair planner from richer multi-node itinerary work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict


@dataclass(frozen=True)
class RoleDefinition:
    role: str
    label: str
    supply_domain: str
    keywords: tuple[str, ...]
    default_duration_min: int
    current_support: str


ROLE_DEFINITIONS: tuple[RoleDefinition, ...] = (
    RoleDefinition(
        role="lodging",
        label="住宿",
        supply_domain="hotel",
        keywords=(
            "酒店",
            "住宿",
            "民宿",
            "住一晚",
            "住两天",
            "两天一晚",
            "住处",
            "家庭房",
            "套房",
            "有厨房",
            "过夜",
            "住酒店",
            "住海边",
            "海边住",
            "住在海边",
            "海景房",
            "海边酒店",
            "海景酒店",
        ),
        default_duration_min=720,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="dental_clinic",
        label="牙科/口腔诊所",
        supply_domain="local_service",
        keywords=("牙科", "口腔", "牙疼", "洗牙", "补牙", "种植牙", "牙齿矫正", "矫正", "牙医"),
        default_duration_min=60,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="sports_training",
        label="运动培训/足球培训",
        supply_domain="local_service",
        keywords=(
            "少儿足球",
            "足球体验",
            "足球试听",
            "足球课",
            "足球培训",
            "足球训练",
            "足球培训班",
            "青训",
            "体育培训",
            "教练",
            "培训班",
            "报个足球",
        ),
        default_duration_min=90,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="travel_agency",
        label="旅行社/签证出境游",
        supply_domain="local_service",
        keywords=("旅行社", "出境游", "办签证", "签证", "旅游团", "跟团游", "特价旅游", "境外游"),
        default_duration_min=45,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="exhibition",
        label="展览/博物馆",
        supply_domain="activity",
        keywords=(
            "展览",
            "看展",
            "博物馆",
            "美术馆",
            "画展",
            "展厅",
            "特展",
            "艺术展",
            "文物展",
            "摄影展",
            "珍品展",
            "漆器展",
            "大展",
            "专业讲解",
            "讲解服务",
            "讲解",
            "导览",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="family_activity",
        label="亲子活动",
        supply_domain="activity",
        keywords=(
            "亲子",
            "孩子",
            "小朋友",
            "儿童",
            "带娃",
            "亲子手作",
            "亲子手工",
            "陶艺",
            "手作",
            "手工",
            "DIY",
            "diy",
            "游乐园",
            "儿童乐园",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="family_indoor_play",
        label="室内亲子乐园",
        supply_domain="activity",
        keywords=("室内乐园", "儿童游乐", "室内游乐", "淘气堡", "蹦床乐园"),
        default_duration_min=150,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="citywalk_market",
        label="城市漫步/市集",
        supply_domain="activity",
        keywords=(
            "citywalk",
            "城市漫步",
            "步行街",
            "市集",
            "逛街",
            "本地市集",
            "历史文化",
            "历史建筑",
            "文化街区",
            "文化景区",
            "名街",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="park_scenic_walk",
        label="公园/夜景散步",
        supply_domain="activity",
        keywords=("公园", "绿地", "滨江", "江边", "河边", "夜景", "散步", "看夜景", "外滩夜景"),
        default_duration_min=75,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="river_cruise",
        label="游船/夜游",
        supply_domain="activity",
        keywords=("游船", "游轮", "邮轮", "浦江游览", "黄浦江夜游", "夜游黄浦江", "包厢", "自助餐", "游览船"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="restaurant_breakfast",
        label="早餐",
        supply_domain="restaurant",
        keywords=("早餐", "早饭", "早上吃", "包子", "馄饨", "豆浆", "粥店", "早点"),
        default_duration_min=45,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="restaurant_lunch",
        label="午餐",
        supply_domain="restaurant",
        keywords=("中午", "中饭", "午饭", "午餐", "吃个中饭", "吃午饭"),
        default_duration_min=70,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="restaurant_dinner",
        label="晚餐",
        supply_domain="restaurant",
        keywords=(
            "晚饭",
            "晚餐",
            "晚上吃",
            "晚上能去哪吃",
            "晚上去哪吃",
            "吃晚饭",
            "浪漫晚餐",
            "庆祝",
            "订座",
            "堂食",
            "先吃饭",
            "吃完饭",
            "吃顿饭",
            "吃饭后",
        ),
        default_duration_min=80,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="restaurant_specific",
        label="指定餐饮",
        supply_domain="restaurant",
        keywords=(
            "蟹黄面",
            "去哪吃",
            "宴请",
            "商务宴请",
            "重要客户",
            "本帮菜",
            "上海菜",
            "小笼包",
            "小笼",
            "生煎",
            "生煎包",
            "锅贴",
            "馄饨",
            "汤包",
            "海鲜",
            "海鲜餐厅",
            "胶东菜",
            "鲁菜",
            "火锅",
            "烤肉",
            "烧烤",
            "烧烤店",
            "日式烧肉",
            "炭火烤肉",
            "炭火",
            "炭烤",
            "韩式烤肉",
            "羊肉串",
            "扒房",
            "牛排",
            "西餐",
            "西餐厅",
            "日料",
            "咖喱",
            "小吃",
            "面馆",
            "轻食",
            "简餐",
            "餐厅",
            "聚餐",
            "宴请",
            "商务宴请",
            "夜宵",
            "吃饭",
            "吃个",
            "吃点",
            "吃点东西",
        ),
        default_duration_min=75,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="board_game_escape",
        label="桌游/密室/剧本杀",
        supply_domain="activity",
        keywords=(
            "桌游",
            "棋牌",
            "剧本杀",
            "狼人杀",
            "密室",
            "密室逃脱",
            "推理馆",
            "沉浸式游戏",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="karaoke",
        label="KTV/唱歌",
        supply_domain="activity",
        keywords=(
            "KTV",
            "ktv",
            "唱歌",
            "卡拉OK",
            "卡拉ok",
            "练歌房",
            "欢唱",
            "去唱",
            "唱一小时",
            "唱一会",
            "唱一会儿",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="internet_cafe",
        label="网吧/电竞",
        supply_domain="activity",
        keywords=("网吧", "网咖", "电竞馆", "电竞", "上网", "电玩", "游戏", "通宵玩游戏", "通宵"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="bar",
        label="酒吧/小酌",
        supply_domain="activity",
        keywords=("酒吧", "清吧", "喝一杯", "小酌", "鸡尾酒", "精酿", "夜店"),
        default_duration_min=75,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="talk_show",
        label="脱口秀/演出",
        supply_domain="activity",
        keywords=("脱口秀", "喜剧", "喜剧场", "相声", "曲艺", "评弹", "剧场", "演出", "livehouse", "Livehouse"),
        default_duration_min=100,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="theatre_performance",
        label="话剧/剧场演出",
        supply_domain="activity",
        keywords=("话剧", "戏剧", "舞台剧", "儿童剧", "剧院"),
        default_duration_min=110,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="cinema",
        label="电影/影院",
        supply_domain="activity",
        keywords=("电影", "影院", "电影院", "观影", "IMAX", "imax"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="cafe",
        label="咖啡/下午茶",
        supply_domain="restaurant",
        keywords=("咖啡", "咖啡厅", "咖啡馆", "下午茶", "坐坐", "小坐", "坐着聊", "坐下聊", "聊项目", "甜品"),
        default_duration_min=60,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="souvenir_shopping",
        label="特产/伴手礼",
        supply_domain="shopping",
        keywords=("特产", "伴手礼", "纪念品", "文创", "周边", "礼品", "礼物", "小礼物", "礼品店", "蛋糕", "生日蛋糕", "带回去", "送朋友", "送老师"),
        default_duration_min=45,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="beauty_cosmetics",
        label="美妆/日化",
        supply_domain="retail",
        keywords=("美妆", "日化", "化妆品", "护肤", "香水", "口红", "彩妆", "美妆店"),
        default_duration_min=35,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="nail_salon",
        label="美甲/美睫",
        supply_domain="beauty_service",
        keywords=("美甲", "美睫", "甲油胶", "做指甲"),
        default_duration_min=70,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="flower_shop",
        label="鲜花/花店",
        supply_domain="retail",
        keywords=("鲜花", "花店", "花束", "买花", "买束花", "小束花", "束花", "花艺"),
        default_duration_min=25,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="convenience_store",
        label="便利店/日用品",
        supply_domain="shopping",
        keywords=("便利店", "日用品", "超市", "买水", "买点水", "生活用品", "零食"),
        default_duration_min=20,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="cultural_photo",
        label="文化体验/拍照",
        supply_domain="activity",
        keywords=("汉服", "拍照", "写真", "摄影", "古装", "换装", "汉服拍照", "体验汉服"),
        default_duration_min=90,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="tea_house",
        label="茶艺/茶馆休息",
        supply_domain="restaurant",
        keywords=("茶艺", "茶馆", "茶室", "品茶", "喝茶", "茶空间"),
        default_duration_min=60,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="wellness_massage",
        label="足疗/按摩休息",
        supply_domain="activity",
        keywords=("SPA", "spa", "足疗", "按摩", "捏脚", "修脚", "洗脚", "推拿", "养生", "休息一下"),
        default_duration_min=60,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="pet_grooming",
        label="宠物美容/洗护",
        supply_domain="pet_service",
        keywords=("宠物美容", "宠物spa", "宠物SPA", "洗护", "宠物洗澡", "宠物护理", "毛发", "金毛"),
        default_duration_min=90,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="pet_cafe",
        label="宠物友好咖啡",
        supply_domain="restaurant",
        keywords=("宠物友好咖啡", "宠物友好", "可带宠物", "带狗咖啡"),
        default_duration_min=60,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="pet_hospital",
        label="宠物医院/体检",
        supply_domain="pet_service",
        keywords=("宠物医院", "宠物体检", "体检", "兽医", "动物医院"),
        default_duration_min=60,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="pet_store",
        label="宠物店/宠物用品",
        supply_domain="pet_service",
        keywords=("宠物店", "宠物用品", "营养品", "宠物营养", "狗粮", "猫粮"),
        default_duration_min=35,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="fitness",
        label="健身/瑜伽",
        supply_domain="activity",
        keywords=("健身", "健身房", "瑜伽", "普拉提", "运动"),
        default_duration_min=60,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="parking",
        label="停车",
        supply_domain="transport_service",
        keywords=("停车", "停车场", "好停车", "免费停车"),
        default_duration_min=15,
        current_support="unsupported",
    ),
)


SEQUENCE_WORDS = ("先", "然后", "再", "顺便", "最后", "接着", "之后")
OVERNIGHT_WORDS = ("住一晚", "住宿", "酒店", "民宿", "住处", "有厨房", "第二天", "过夜", "住海边", "海景房", "海边酒店")
EXPLICIT_FULL_DAY_WORDS = ("一整天", "全天", "一天")
FULL_DAY_SPAN_WORDS = ("上午", "晚上")
SHORT_WINDOW_WORDS = ("今晚", "夜宵", "晚上", "下班")
TWO_DAY_WORDS = ("两天", "2天", "二天", "两日", "2日", "周末两天", "明后天", "第二天", "次日")
CROSS_DAY_TIME_WINDOWS = {
    "saturday_afternoon_to_sunday_noon",
}
FULL_DAY_MIN_NODE_COUNT = 4
TWO_DAY_MIN_NODE_COUNT = 6
NAMED_EVENT_PATTERNS = (
    re.compile(r"“([^”]{4,80})”"),
    re.compile(r"\"([^\"]{4,80})\""),
)
NEGATED_ROLE_PREFIXES = (
    "不要",
    "不想",
    "不吃",
    "别吃",
    "别去",
    "避免",
    "避开",
    "排除",
    "不考虑",
    "不安排",
)
NEGATED_TERM_EQUIVALENTS = {
    "烧烤": ("烤肉", "炭火", "羊肉串"),
    "烤肉": ("烧烤", "炭火烤肉", "日式烧肉"),
    "火锅": ("火锅",),
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dedupe_keep_order(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _collect_text(state: PlanState, constraints: dict[str, Any] | None = None) -> str:
    constraints = constraints or state.get("constraints", {}) or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    raw_text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    negated_terms = _negated_role_terms(raw_text)

    def include_value(value: Any) -> bool:
        if value in (None, ""):
            return False
        return not _value_matches_negated_terms(value, negated_terms)

    values: list[Any] = [
        state.get("user_input"),
        constraints.get("raw_text"),
        state.get("scene_type"),
        constraints.get("scene"),
    ]
    for key in ("hard_tags", "soft_tags", "avoid", "scenario_activities"):
        values.extend(value for value in _as_list(constraints.get(key)) if include_value(value))
    for key in (
        "activity_type",
        "food_type",
        "restaurant_type",
        "experience_type",
        "facility_type",
    ):
        values.extend(value for value in _as_list(planning_preferences.get(key)) if include_value(value))
    values.extend(value for value in _as_list(state.get("scenario_activities")) if include_value(value))
    return " ".join(str(value) for value in values if value not in (None, ""))


def _named_entities(text: str) -> list[str]:
    entities: list[str] = []
    for pattern in NAMED_EVENT_PATTERNS:
        entities.extend(match.group(1).strip() for match in pattern.finditer(text))
    return _dedupe_keep_order(entities)


def _is_negated_role_occurrence(text: str, index: int) -> bool:
    window = text[max(0, index - 8): index + 1]
    return any(marker in window for marker in NEGATED_ROLE_PREFIXES)


def _negated_role_terms(text: str) -> set[str]:
    negated: set[str] = set()
    lowered_text = text.lower()
    for definition in ROLE_DEFINITIONS:
        for keyword in definition.keywords:
            lowered_keyword = keyword.lower()
            start = 0
            while True:
                index = lowered_text.find(lowered_keyword, start)
                if index < 0:
                    break
                if _is_negated_role_occurrence(text, index):
                    negated.add(keyword)
                    negated.update(NEGATED_TERM_EQUIVALENTS.get(keyword, ()))
                start = index + max(1, len(lowered_keyword))
    return negated


def _value_matches_negated_terms(value: Any, negated_terms: set[str]) -> bool:
    if not negated_terms:
        return False
    text = str(value)
    return any(term and (term in text or text in term) for term in negated_terms)


def _role_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    lowered_text = text.lower()
    for definition in ROLE_DEFINITIONS:
        matched_terms: list[tuple[int, str]] = []
        for keyword in definition.keywords:
            lowered_keyword = keyword.lower()
            if keyword.isascii() and keyword.isalpha():
                matches = re.finditer(
                    rf"(?<![a-z]){re.escape(lowered_keyword)}(?![a-z])",
                    lowered_text,
                    flags=re.IGNORECASE,
                )
                matched_terms.extend((match.start(), keyword) for match in matches)
            else:
                start = 0
                while True:
                    index = lowered_text.find(lowered_keyword, start)
                    if index < 0:
                        break
                    if not _is_negated_role_occurrence(text, index):
                        matched_terms.append((index, keyword))
                    start = index + max(1, len(lowered_keyword))
        if not matched_terms:
            continue
        matched_terms.sort(key=lambda item: item[0])
        hits.append(
            {
                "role": definition.role,
                "label": definition.label,
                "supply_domain": definition.supply_domain,
                "default_duration_min": definition.default_duration_min,
                "current_support": definition.current_support,
                "matched_terms": _dedupe_keep_order([term for _, term in matched_terms]),
                "matched_offsets": [
                    {"term": term, "position": index}
                    for index, term in matched_terms
                ],
                "first_position": matched_terms[0][0],
            }
        )
    hits.sort(key=lambda item: item["first_position"])
    return hits


def _has_child_companion_context(state: PlanState, constraints: dict[str, Any]) -> bool:
    raw_text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    if _explicitly_excludes_child_context(state, constraints, raw_text):
        return False
    strong_terms = ("孩子", "小孩", "小朋友", "带娃", "亲子", "一家", "家庭", "宝宝", "儿童友好")
    if any(term in raw_text for term in strong_terms):
        return True
    companions = constraints.get("companions") or []
    if isinstance(companions, list) and any(
        isinstance(item, dict) and str(item.get("role") or "").lower() == "child"
        for item in companions
    ):
        return True
    return constraints.get("child_age") not in (None, "")


def _explicitly_excludes_child_context(
    state: PlanState,
    constraints: dict[str, Any],
    raw_text: str,
) -> bool:
    companions = constraints.get("companions") or []
    has_current_child = (
        isinstance(companions, list)
        and any(
            isinstance(item, dict) and str(item.get("role") or "").lower() == "child"
            for item in companions
        )
    ) or constraints.get("child_age") not in (None, "")
    if has_current_child:
        return False

    exclusion_phrases = (
        "不带孩子",
        "不带小孩",
        "不带小朋友",
        "不带娃",
        "不带儿童",
        "孩子不带",
        "小孩不带",
        "娃不带",
        "不带宝宝",
        "没有孩子",
        "不是亲子",
        "不要亲子",
        "别按亲子",
        "别排亲子",
        "别套进来",
        "别套用",
        "不要套用",
        "不套用",
        "排除亲子",
    )
    if any(phrase in raw_text for phrase in exclusion_phrases) and any(
        term in raw_text for term in ("孩子", "小孩", "小朋友", "带娃", "宝宝", "亲子", "儿童")
    ):
        return True

    avoid_terms = {str(value).strip() for value in _as_list(constraints.get("avoid"))}
    return bool({"亲子", "儿童友好", "孩子", "儿童", "带娃"}.intersection(avoid_terms))


def _has_citywalk_context(state: PlanState, constraints: dict[str, Any]) -> bool:
    raw_text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    return any(
        term in raw_text
        for term in ("citywalk", "城市漫步", "漫步", "逛街", "市集", "街区", "历史文化", "历史建筑", "文化街区", "文化景区")
    )


def _has_explicit_restaurant_context(state: PlanState, constraints: dict[str, Any]) -> bool:
    raw_text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    return any(
        term in raw_text
        for term in (
            "餐厅",
            "饭店",
            "吃饭",
            "吃点",
            "吃个",
            "午饭",
            "午餐",
            "中饭",
            "晚饭",
            "晚餐",
            "早餐",
            "早饭",
            "正餐",
            "夜宵",
            "聚餐",
            "火锅",
            "宴请",
            "商务宴请",
            "重要客户",
            "烤肉",
            "烧烤",
            "健康餐",
            "轻食",
            "包子",
            "小吃",
            "上海菜",
            "小笼包",
            "小笼",
            "生煎",
            "生煎包",
            "锅贴",
            "馄饨",
            "汤包",
            "海鲜",
            "本地海鲜",
            "胶东菜",
            "鲁菜",
            "本地菜",
            "地方菜",
            "日料",
            "西餐",
            "蟹黄面",
            "去哪吃",
        )
    )


def _has_explicit_no_lodging_context(raw_text: str) -> bool:
    return any(
        term in raw_text
        for term in (
            "不订酒店",
            "不订住宿",
            "不用订酒店",
            "不用订住宿",
            "不需要酒店",
            "不需要住宿",
            "不要酒店",
            "不要住宿",
            "不住酒店",
            "不住民宿",
            "住我家",
            "住家里",
            "住在我家",
            "回家住",
            "住自己家",
        )
    )


def _has_explicit_lodging_booking_context(raw_text: str) -> bool:
    return any(
        term in raw_text
        for term in (
            "订酒店",
            "订住宿",
            "订民宿",
            "预订酒店",
            "预订住宿",
            "预订民宿",
            "找酒店",
            "找个酒店",
            "找住宿",
            "找民宿",
            "住酒店",
            "住一晚",
            "入住",
            "住宿",
            "民宿",
            "住处",
            "家庭房",
            "套房",
            "有厨房",
            "过夜",
            "两天一夜",
        )
    )


NEGATED_ROLE_PREFIXES = (
    "不想",
    "不要",
    "不去",
    "不再",
    "不用",
    "不能",
    "别",
    "避开",
    "排除",
    "拒绝",
)


def _sentence_containing(text: str, position: int) -> str:
    start = max(text.rfind(mark, 0, position) for mark in ("。", "；", ";", "\n"))
    end_candidates = [
        index
        for mark in ("。", "；", ";", "\n")
        if (index := text.find(mark, position)) >= 0
    ]
    start = 0 if start < 0 else start + 1
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end]


def _term_has_negated_prefix(text: str, position: int) -> bool:
    prefix = text[max(0, position - 16) : position]
    last_boundary = max(
        prefix.rfind(mark)
        for mark in ("。", "；", ";", "\n", "，", ",")
    )
    if last_boundary >= 0:
        prefix = prefix[last_boundary + 1 :]
    if any(term in prefix for term in ("能不能", "能否", "可不可以")):
        return False
    if any(term in prefix for term in ("不能下单", "不能预订", "不能预约", "不能购买")) and any(
        term in prefix for term in ("如果", "若", "电话确认", "到店确认", "写成")
    ):
        return False
    return any(term in prefix for term in NEGATED_ROLE_PREFIXES)


def _old_preference_rejected(text: str, offsets: list[Any]) -> bool:
    rejection_phrases = (
        "这次别按",
        "别再按",
        "别按那个来",
        "别按这个来",
        "别套进来",
        "不要按",
        "不要套用",
        "不套用",
    )
    old_preference_terms = (
        "亲子",
        "带娃",
        "儿童",
        "儿童友好",
        "低卡",
        "轻食",
        "减脂",
        "减肥",
        "健康餐",
        "火锅",
        "烤肉",
        "烧烤",
        "密室",
        "密室逃脱",
        "剧本杀",
        "桌游",
        "KTV",
        "ktv",
        "唱歌",
    )
    rejected_old_preference = False
    for item in offsets:
        try:
            position = int(item.get("position"))
        except (AttributeError, TypeError, ValueError):
            continue
        term = str(item.get("term") or "")
        if term and term not in old_preference_terms:
            return False
        sentence = _sentence_containing(text, position)
        if any(term in sentence for term in ("以前", "平时", "历史", "旧偏好", "上次", "那套", "那种")) and any(
            phrase in sentence for phrase in rejection_phrases
        ):
            rejected_old_preference = True
            continue
        return False
    return rejected_old_preference


def _has_positive_dental_service_intent(text: str) -> bool:
    return any(
        term in text
        for term in (
            "处理个口腔",
            "口腔问题",
            "找个牙科",
            "找个靠谱牙科",
            "找牙科",
            "找口腔",
            "牙科做",
            "预约牙科",
            "预约洗牙",
            "洗牙或补牙",
            "补牙咨询",
            "看牙",
        )
    )


def _anchor_only_service_role(role: str, text: str, matched_terms: list[str]) -> bool:
    if role == "cinema":
        anchor_patterns = (
            r"(看完|看过).{0,4}(电影|影院|观影)",
            r"(电影|影院|观影).{0,10}(结束|散场|出来|以后|之后|后)",
        )
        if any(re.search(pattern, text) for pattern in anchor_patterns):
            return not any(
                re.search(pattern, text)
                for pattern in (
                    r"(想|要|打算|准备|安排|帮我).{0,8}(看|订|买).{0,4}(电影|影院|影票)",
                    r"(再|然后|之后|饭后|吃完).{0,6}(去)?看.{0,4}电影",
                    r"(电影票|影票|订票)",
                )
            )
    if role == "dental_clinic":
        if _has_positive_dental_service_intent(text):
            return False
        anchor_patterns = (
            r"(洗牙|牙科|口腔|牙医).{0,6}(结束|做完|刚结束|刚做完)",
            r"(结束|做完|刚结束|刚做完).{0,8}(洗牙|牙科|口腔|牙医)",
            r"(洗牙|牙科|口腔|牙医).{0,6}后",
            r"(补牙|补完牙|刚补完牙).{0,12}(出来|结束|以后|之后|后)",
            r"(刚补完牙|刚补牙|补完牙)",
            r"(牙科|口腔).{0,4}医院.{0,8}(出来|离开|附近出来)",
        )
        if any(re.search(pattern, text) for pattern in anchor_patterns):
            if any(
                term in text
                for term in (
                    "牙科复诊",
                    "牙医复诊",
                    "口腔复诊",
                    "必须到",
                    "要到",
                    "已有预约",
                    "已经约",
                )
            ):
                return False
            return not any(
                term in text
                for term in ("找牙科", "找口腔", "预约牙科", "预约洗牙", "比较", "看牙")
            )
    if role == "sports_training":
        anchor_patterns = (
            r"(足球训练|足球培训|训练).{0,8}(下课|结束|刚结束|刚训练完)",
            r"(足球训练|足球培训|训练).{0,8}后",
            r"(少儿足球|足球试听|足球课|足球训练|足球培训|训练).{0,18}"
            r"(已约好|已经约好|约好了|不需要再|不用再|不需要|不用)",
        )
        has_anchor_pattern = any(re.search(pattern, text) for pattern in anchor_patterns)
        if has_anchor_pattern:
            if any(
                re.search(pattern, text)
                for pattern in (
                    r"(想|要|准备|打算|看看|找|预约|报名|报个)[^，。；;,.]{0,18}"
                    r"(少儿足球|足球体验|足球试听|足球课|足球培训|足球训练|青训)",
                    r"(少儿足球|足球体验|足球试听|足球课|足球培训|足球训练|青训)"
                    r"[^，。；;,.]{0,18}(有没有|可不可以|能不能|想|要|报名|预约)",
                )
            ):
                return False
            if any(term in text for term in ("已约好", "已经约好", "约好了", "不需要再", "不用再")):
                return True
            return not any(
                re.search(pattern, text)
                for pattern in (
                    r"(报个足球|找足球|预约足球)",
                    r"(?<!不需要)(?<!不用)(?<!别)(?<!不要)(报名|培训班|试听|体验课)",
                )
            )
    if role == "pet_hospital":
        if "体检" in matched_terms and not any(
            term in text
            for term in (
                "宠物",
                "猫",
                "狗",
                "金毛",
                "小狗",
                "小猫",
                "兽医",
                "动物医院",
                "宠物医院",
            )
        ):
            return True
    return False


def _medical_no_alcohol_overrides_bar(text: str, matched_terms: list[str]) -> bool:
    if not any(
        term in matched_terms for term in ("喝一杯", "小酌", "鸡尾酒", "精酿", "夜店")
    ):
        return False
    has_medical_context = any(
        term in text for term in ("医生", "麻药", "术后", "洗牙", "拔牙", "刚治疗")
    )
    has_no_alcohol = any(
        term in text
        for term in ("别喝酒", "不能喝酒", "不要喝酒", "不喝酒", "禁酒", "别饮酒")
    )
    return has_medical_context and has_no_alcohol


def _role_context_is_negative_or_anchor(
    hit: dict[str, Any],
    *,
    raw_text: str,
) -> bool:
    role = str(hit.get("role") or "")
    matched_terms = [str(term) for term in hit.get("matched_terms", [])]
    offsets = hit.get("matched_offsets") or []
    current_service_anchor = role == "dental_clinic" and any(
        term in raw_text
        for term in (
            "牙科复诊",
            "牙医复诊",
            "口腔复诊",
            "必须到",
            "已有预约",
            "已经约",
        )
    )
    has_negated_match = False
    has_positive_match = False
    for item in offsets:
        try:
            position = int(item.get("position"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not current_service_anchor and _term_has_negated_prefix(raw_text, position):
            has_negated_match = True
            continue
        has_positive_match = True
    if has_negated_match and not has_positive_match:
        return True
    if _old_preference_rejected(raw_text, offsets):
        return True
    if _anchor_only_service_role(role, raw_text, matched_terms):
        return True
    if role == "bar" and _medical_no_alcohol_overrides_bar(raw_text, matched_terms):
        return True
    return False


def _role_blocked_by_avoid(hit: dict[str, Any], constraints: dict[str, Any]) -> bool:
    role = str(hit.get("role") or "")
    avoid_terms = {str(value).strip() for value in _as_list(constraints.get("avoid"))}
    if not avoid_terms:
        return False

    role_avoid_terms = {
        "karaoke": {"KTV", "ktv", "KTV欢唱", "唱歌", "欢唱", "卡拉OK", "卡拉ok"},
        "bar": {"酒吧", "夜店", "喝酒", "小酌", "精酿", "鸡尾酒"},
        "board_game_escape": {"密室", "密室逃脱", "剧本杀", "桌游", "推理馆"},
        "exhibition": {"博物馆展览", "博物馆", "展览", "看展", "美术馆"},
        "family_activity": {"亲子", "儿童友好", "游乐", "亲子乐园", "儿童乐园", "孩子", "儿童", "带娃"},
        "family_indoor_play": {"亲子", "儿童友好", "游乐", "亲子乐园", "儿童乐园", "孩子", "儿童", "带娃"},
    }.get(role)
    if not role_avoid_terms:
        return False
    return bool(avoid_terms.intersection(role_avoid_terms))


def _family_activity_hit_is_companion_only(hit: dict[str, Any], raw_text: str) -> bool:
    if hit.get("role") not in {"family_activity", "family_indoor_play"}:
        return False
    matched_terms = {str(term) for term in hit.get("matched_terms", []) if term}
    if not matched_terms or not matched_terms.issubset({"孩子", "小朋友", "儿童", "带娃", "亲子"}):
        return False
    if any(
        term in raw_text
        for term in (
            "亲子活动",
            "儿童乐园",
            "室内乐园",
            "游乐",
            "手作",
            "看展",
            "展览",
            "博物馆",
            "美术馆",
            "电影",
            "公园",
            "散步",
            "出去玩",
            "玩一下",
            "玩一会",
            "透口气",
            "活动",
        )
    ):
        return False
    return bool(
        re.search(r"(孩子|小朋友|儿童).{0,64}(已约好|已经约好|约好了|不需要再|不用再)", raw_text)
        or re.search(r"(已约好|已经约好|约好了|不需要再|不用再).{0,64}(孩子|小朋友|儿童)", raw_text)
    )


def _keep_positive_matched_terms(
    hit: dict[str, Any],
    *,
    raw_text: str,
) -> dict[str, Any]:
    offsets = hit.get("matched_offsets") or []
    if not offsets:
        return hit

    positive_terms: list[str] = []
    negated_terms: set[str] = set()
    role = str(hit.get("role") or "")
    current_service_anchor = role == "dental_clinic" and any(
        term in raw_text
        for term in (
            "牙科复诊",
            "牙医复诊",
            "口腔复诊",
            "必须到",
            "已有预约",
            "已经约",
        )
    )
    for item in offsets:
        term = str(item.get("term") or "")
        if not term:
            continue
        try:
            position = int(item.get("position"))
        except (AttributeError, TypeError, ValueError):
            positive_terms.append(term)
            continue
        if not current_service_anchor and _term_has_negated_prefix(raw_text, position):
            negated_terms.add(term)
            continue
        positive_terms.append(term)

    if not positive_terms or not negated_terms:
        return hit

    result = dict(hit)
    result["matched_terms"] = _dedupe_keep_order(positive_terms)
    return result


def _park_hit_is_location_anchor_only(hit: dict[str, Any], raw_text: str) -> bool:
    if hit.get("role") != "park_scenic_walk":
        return False
    matched_terms = [str(term) for term in hit.get("matched_terms", [])]
    if matched_terms != ["公园"]:
        return False
    if not any(term in raw_text for term in ("公园附近", "公园周边", "公园旁", "公园出发")):
        return False
    return not any(
        term in raw_text
        for term in (
            "逛公园",
            "去公园",
            "公园玩",
            "公园散步",
            "公园走走",
            "散步",
            "看夜景",
            "夜景",
            "滨江走",
        )
    )


def _lodging_hit_is_destination_anchor_only(hit: dict[str, Any], raw_text: str) -> bool:
    if hit.get("role") != "lodging":
        return False
    if _has_explicit_lodging_booking_context(raw_text):
        return False

    offsets = hit.get("matched_offsets") or []
    if not offsets:
        return False

    anchor_patterns = (
        r"(回|送回|送到|返回|到|去|前往|抵达).{0,16}酒店",
        r"酒店.{0,8}(附近|门口|周边|旁边|集合|下车|开会|出发)",
        r"酒店.{0,6}(接送|送回|返回)",
    )
    for item in offsets:
        try:
            position = int(item.get("position"))
        except (AttributeError, TypeError, ValueError):
            return False
        sentence = _sentence_containing(raw_text, position)
        if not any(re.search(pattern, sentence) for pattern in anchor_patterns):
            return False
    return True


def _booking_only_dinner_hit_is_redundant(
    hit: dict[str, Any],
    role_set: set[str],
) -> bool:
    if hit.get("role") != "restaurant_dinner":
        return False
    matched_terms = {
        str(term).strip()
        for term in hit.get("matched_terms", [])
        if str(term).strip()
    }
    if not matched_terms:
        return False
    booking_only_terms = {"订座", "堂食", "庆祝"}
    if not matched_terms.issubset(booking_only_terms):
        return False
    concrete_restaurant_roles = {
        "restaurant_breakfast",
        "restaurant_lunch",
        "restaurant_specific",
        "cafe",
        "tea_house",
        "pet_cafe",
    }
    return bool(role_set.intersection(concrete_restaurant_roles))


def _drop_false_soft_tag_hits(
    hits: list[dict[str, Any]],
    *,
    state: PlanState,
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    has_child_context = _has_child_companion_context(state, constraints)
    has_citywalk_context = _has_citywalk_context(state, constraints)
    has_restaurant_context = _has_explicit_restaurant_context(state, constraints)
    has_other_specific_role = any(
        hit.get("role") not in {"restaurant_specific"}
        for hit in hits
    )
    role_set = {str(hit.get("role") or "") for hit in hits}
    raw_text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    for hit in hits:
        role = hit.get("role")
        if _booking_only_dinner_hit_is_redundant(hit, role_set):
            continue
        if _role_blocked_by_avoid(hit, constraints):
            continue
        if _family_activity_hit_is_companion_only(hit, raw_text):
            continue
        if _role_context_is_negative_or_anchor(hit, raw_text=raw_text):
            continue
        if role == "lodging" and _has_explicit_no_lodging_context(raw_text):
            continue
        if _lodging_hit_is_destination_anchor_only(hit, raw_text):
            continue
        if _park_hit_is_location_anchor_only(hit, raw_text):
            continue
        if role in {"family_activity", "family_indoor_play"} and not has_child_context:
            continue
        if (
            role == "family_activity"
            and {"citywalk_market", "exhibition"}.intersection(role_set)
            and not any(term in raw_text for term in ("亲子", "儿童乐园", "游乐园", "室内乐园", "淘气堡", "亲子活动"))
        ):
            continue
        if role == "citywalk_market" and not has_citywalk_context:
            continue
        if (
            role == "park_scenic_walk"
            and "商圈" in raw_text
            and not any(term in raw_text for term in ("散步", "看夜景", "夜景", "逛公园", "公园散步", "滨江走"))
        ):
            continue
        if role == "restaurant_specific" and has_other_specific_role and not has_restaurant_context:
            continue
        if (
            role == "wellness_massage"
            and "pet_grooming" in role_set
            and any(term in hit.get("matched_terms", []) for term in ("SPA", "spa", "宠物spa", "宠物SPA"))
        ):
            continue
        if role == "cafe" and "pet_cafe" in role_set:
            continue
        if (
            role == "internet_cafe"
            and "board_game_escape" in role_set
            and any(term in hit.get("matched_terms", []) for term in ("游戏", "推理游戏"))
        ):
            continue
        if (
            role == "talk_show"
            and "bar" in role_set
            and "酒吧" in raw_text
            and not any(term in raw_text for term in ("脱口秀", "喜剧", "剧场", "话剧", "livehouse", "Livehouse"))
        ):
            continue
        if (
            role == "wellness_massage"
            and "休息一下" in hit.get("matched_terms", [])
            and not any(term in raw_text for term in ("按摩", "足疗", "SPA", "spa", "推拿", "洗脚", "修脚", "捏脚"))
        ):
            continue
        result.append(_keep_positive_matched_terms(hit, raw_text=raw_text))
    return result


def _merge_restaurant_roles(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep meal intent readable without duplicating the same lunch/dinner node."""

    result: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    meal_roles = {"restaurant_breakfast", "restaurant_lunch", "restaurant_dinner"}

    def merge_offsets(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
        offsets = list(existing.get("matched_offsets") or [])
        offsets.extend(incoming.get("matched_offsets") or [])
        if offsets:
            offsets.sort(key=lambda item: int(item.get("position") or 0))
            existing["matched_offsets"] = offsets

    for hit in hits:
        role = hit["role"]
        if role == "restaurant_specific":
            has_specific_meal = any(
                existing["role"] in meal_roles
                for existing in result
            )
            if has_specific_meal:
                hit_position = int(hit.get("first_position") or 0)
                meal_targets = [
                    existing
                    for existing in result
                    if existing["role"] in meal_roles
                ]
                target = min(
                    meal_targets,
                    key=lambda item: abs(
                        int(item.get("first_position") or 0) - hit_position
                    ),
                )
                target["matched_terms"] = _dedupe_keep_order(
                    target.get("matched_terms", []) + hit.get("matched_terms", [])
                )
                merge_offsets(target, hit)
                target["label"] = "餐饮"
                continue
        elif role in meal_roles and "restaurant_specific" in seen_roles:
            for existing in result:
                if existing["role"] == "restaurant_specific":
                    existing["role"] = role
                    existing["label"] = hit.get("label") or "餐饮"
                    existing["default_duration_min"] = hit.get(
                        "default_duration_min",
                        existing.get("default_duration_min"),
                    )
                    existing["matched_terms"] = _dedupe_keep_order(
                        existing.get("matched_terms", []) + hit.get("matched_terms", [])
                    )
                    merge_offsets(existing, hit)
                    existing["first_position"] = min(
                        int(existing.get("first_position") or 0),
                        int(hit.get("first_position") or 0),
                    )
            seen_roles.discard("restaurant_specific")
            seen_roles.add(role)
            continue
        if role in seen_roles:
            continue
        seen_roles.add(role)
        result.append(hit)
    return result


def _has_overnight_context(text: str) -> bool:
    if _has_explicit_lodging_booking_context(text):
        return True
    return any(word in text for word in OVERNIGHT_WORDS if word != "酒店")


def _has_short_explicit_time_window(text: str) -> bool:
    if not any(word in text for word in SHORT_WINDOW_WORDS):
        return False

    clock_values: list[int] = []
    colon_matches = re.findall(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", text)
    hour_word_matches = [
        (match.group(1), "30" if match.group(2) else "00")
        for match in re.finditer(r"(?<!\d)(\d{1,2})点(半)?", text)
    ]
    for hour_text, minute_text in colon_matches + hour_word_matches:
        hour = int(hour_text)
        if hour > 24:
            continue
        minute = int(minute_text)
        if minute > 59:
            continue
        clock_values.append((hour % 24) * 60 + minute)
    if len(clock_values) < 2:
        return False

    start = min(clock_values)
    end = max(clock_values)
    direct_span = end - start
    cross_midnight_span = min(clock_values) + 1440 - max(clock_values)
    return min(direct_span, cross_midnight_span) <= 360


def _has_explicit_cross_day_context(text: str, constraints: dict[str, Any] | None) -> bool:
    constraints = constraints or {}
    if constraints.get("time_window") in CROSS_DAY_TIME_WINDOWS:
        return True
    if re.search(r"周[一二三四五六日天][^，。；;]*到周[一二三四五六日天]", text):
        return True
    if re.search(r"(明天|周[一二三四五六日天]).*(后天|次日|第二天)", text):
        return True
    return False


def _duration_hours_from_constraints(constraints: dict[str, Any]) -> tuple[float, float] | None:
    raw = constraints.get("duration_range") or constraints.get("duration")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        lower = float(raw[0])
        upper = float(raw[1])
    except (TypeError, ValueError):
        return None
    if lower > 48 or upper > 48:
        lower /= 60.0
        upper /= 60.0
    return lower, upper


def _planning_horizon(
    text: str,
    role_count: int,
    constraints: dict[str, Any] | None = None,
) -> str:
    constraints = constraints or {}
    time_window = str(constraints.get("time_window") or constraints.get("window") or "").lower()
    duration_hours = _duration_hours_from_constraints(constraints)
    if (
        _has_explicit_cross_day_context(text, constraints)
        or ("周六" in text and "周日" in text)
        or ("星期六" in text and "星期日" in text)
    ):
        return "two_day"
    if any(word in text for word in TWO_DAY_WORDS):
        return "two_day"
    if any(token in time_window for token in ("two_day", "2_day", "two-day")):
        return "two_day"
    if _has_overnight_context(text):
        return "overnight"
    if any(token in time_window for token in ("overnight", "with_lodging")):
        return "overnight"
    if _has_short_explicit_time_window(text):
        return "half_day"
    has_full_day_span = all(word in text for word in FULL_DAY_SPAN_WORDS)
    if (
        role_count >= 4
        or "full_day" in time_window
        or any(word in text for word in EXPLICIT_FULL_DAY_WORDS)
        or has_full_day_span
        or any(word in text for word in ("一整天", "全天", "玩一天", "一天"))
        or (duration_hours is not None and duration_hours[0] >= 6)
    ):
        return "full_day"
    return "half_day"


def _planning_days(horizon: str) -> int:
    return 2 if horizon in {"overnight", "two_day"} else 1


def _definition_for_role(role: str) -> RoleDefinition | None:
    for definition in ROLE_DEFINITIONS:
        if definition.role == role:
            return definition
    return None


def _make_intent_from_role(
    role: str,
    *,
    label: str | None = None,
    search_terms: list[str] | None = None,
    day_index: int | None = None,
) -> dict[str, Any]:
    definition = _definition_for_role(role)
    if definition is None:
        intent = {
            "role": role,
            "label": label or "活动",
            "supply_domain": "activity",
            "search_terms": search_terms or [],
            "default_duration_min": 120,
            "current_support": "legacy_activity",
        }
    else:
        intent = {
            "role": definition.role,
            "label": label or definition.label,
            "supply_domain": definition.supply_domain,
            "search_terms": search_terms or [],
            "default_duration_min": definition.default_duration_min,
            "current_support": definition.current_support,
        }
    if day_index is not None:
        intent["day_index"] = day_index
    return intent


def _renumber_intents(node_intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, item in enumerate(node_intents, start=1):
        item["node_id"] = f"intent_{index:02d}"
        item["sequence_index"] = index
    return node_intents


def _clone_intent_for_day(
    base_intent: dict[str, Any],
    *,
    day_index: int,
    label: str,
    search_terms: list[str],
) -> dict[str, Any]:
    role = str(base_intent.get("role") or "activity")
    intent = _make_intent_from_role(
        role,
        label=label,
        search_terms=search_terms,
        day_index=day_index,
    )
    if base_intent.get("supply_domain"):
        intent["supply_domain"] = base_intent.get("supply_domain")
    if base_intent.get("current_support"):
        intent["current_support"] = base_intent.get("current_support")
    return intent


def _with_day_index(item: dict[str, Any], day_index: int) -> dict[str, Any]:
    result = dict(item)
    result["day_index"] = day_index
    return result


def _chat_or_rest_role(text: str) -> str:
    if any(term in text for term in ("聊天", "坐坐", "下午茶", "咖啡")):
        return "cafe"
    return "restaurant_dinner"


def _healthy_meal_terms(text: str, base_terms: list[str]) -> list[str]:
    terms = list(base_terms)
    if any(term in text for term in ("健康", "清淡", "轻食", "低油", "低糖", "低卡")):
        terms.extend(["健康", "清淡", "轻食"])
    if any(term in text for term in ("不辣", "不能吃辣", "别太辣", "少辣")):
        terms.extend(["不辣", "少辣"])
    return terms


def _definition_intent(role: str) -> dict[str, Any]:
    definition = _definition_for_role(role)
    if definition is None:
        return _make_intent_from_role(role)
    return {
        "role": definition.role,
        "label": definition.label,
        "supply_domain": definition.supply_domain,
        "search_terms": [],
        "default_duration_min": definition.default_duration_min,
        "current_support": definition.current_support,
    }


def _expand_sparse_two_day_intents(
    node_intents: list[dict[str, Any]],
    *,
    text: str,
    horizon: str,
) -> list[dict[str, Any]]:
    if horizon != "two_day":
        return node_intents

    activity_roles = [
        item
        for item in node_intents
        if item.get("supply_domain") == "activity"
    ]
    restaurant_roles = [
        item
        for item in node_intents
        if item.get("supply_domain") == "restaurant"
    ]
    if (
        len(node_intents) >= TWO_DAY_MIN_NODE_COUNT
        and len(activity_roles) >= 2
        and len(restaurant_roles) >= 2
    ):
        return node_intents

    default_activity = (
        activity_roles[0]
        if activity_roles
        else _definition_intent("family_activity" if "孩子" in text or "亲子" in text else "cultural_photo")
    )
    chat_or_rest_role = _chat_or_rest_role(text)
    first_meal = restaurant_roles[0] if restaurant_roles else _definition_intent("restaurant_lunch")
    second_meal = (
        restaurant_roles[1]
        if len(restaurant_roles) > 1
        else _definition_intent("restaurant_lunch")
    )

    day_one: list[dict[str, Any]] = []
    day_two: list[dict[str, Any]] = []
    original_with_days = [item for item in node_intents if item.get("day_index")]
    if original_with_days:
        for item in node_intents:
            day = int(item.get("day_index") or 1)
            (day_two if day >= 2 else day_one).append(_with_day_index(item, min(day, 2)))
    else:
        split_after = max(2, len(node_intents) // 2)
        for index, item in enumerate(node_intents, start=1):
            day = 1 if index <= split_after else 2
            (day_one if day == 1 else day_two).append(_with_day_index(item, day))

    def ensure_day(day_items: list[dict[str, Any]], day_index: int) -> list[dict[str, Any]]:
        activity_count = sum(1 for item in day_items if item.get("supply_domain") == "activity")
        restaurant_count = sum(1 for item in day_items if item.get("supply_domain") == "restaurant")
        while activity_count < 1:
            day_items.append(
                _clone_intent_for_day(
                    default_activity,
                    day_index=day_index,
                    label="第一天活动" if day_index == 1 else "第二天活动",
                    search_terms=["轻松", "拍照" if "拍照" in text else "活动"],
                )
            )
            activity_count += 1
        while restaurant_count < 1:
            meal_role = "restaurant_lunch" if day_index == 2 else str(first_meal.get("role") or "restaurant_lunch")
            day_items.append(
                _make_intent_from_role(
                    meal_role,
                    label="第一天餐饮" if day_index == 1 else "第二天午餐",
                    search_terms=_healthy_meal_terms(text, ["餐饮", "午餐"]),
                    day_index=day_index,
                )
            )
            restaurant_count += 1
        while len(day_items) < 3:
            if activity_count < 2:
                day_items.append(
                    _clone_intent_for_day(
                        default_activity,
                        day_index=day_index,
                        label="第一天补充活动" if day_index == 1 else "第二天补充活动",
                        search_terms=["轻松", "休息", "补充活动"],
                    )
                )
                activity_count += 1
            else:
                meal_role = (
                    chat_or_rest_role
                    if day_index == 1
                    else str(second_meal.get("role") or "restaurant_lunch")
                )
                day_items.append(
                    _make_intent_from_role(
                        meal_role,
                        label="第一天休息餐饮" if day_index == 1 else "第二天餐饮",
                        search_terms=_healthy_meal_terms(text, ["休息", "餐饮"]),
                        day_index=day_index,
                    )
                )
                restaurant_count += 1
        return day_items

    expanded = ensure_day(day_one, 1) + ensure_day(day_two, 2)
    return _renumber_intents(expanded)


def _expand_sparse_full_day_intents(
    node_intents: list[dict[str, Any]],
    *,
    text: str,
    horizon: str,
) -> list[dict[str, Any]]:
    if horizon != "full_day":
        return node_intents

    activity_roles = [
        str(item.get("role") or "")
        for item in node_intents
        if item.get("supply_domain") == "activity"
    ]
    if not activity_roles:
        return node_intents
    restaurant_count = sum(
        1 for item in node_intents if item.get("supply_domain") == "restaurant"
    )
    if (
        len(node_intents) >= FULL_DAY_MIN_NODE_COUNT
        and len(activity_roles) >= 2
        and restaurant_count >= 1
    ):
        return node_intents

    activity_role = activity_roles[0] or "activity"
    supplemental = _make_intent_from_role(
        activity_role,
        label="下午补充活动",
        search_terms=["下午", "轻松", "补充活动"],
    )

    expanded = list(node_intents)
    has_restaurant = any(item.get("supply_domain") == "restaurant" for item in expanded)
    if not has_restaurant:
        expanded.append(
            _make_intent_from_role(
                "restaurant_lunch",
                label="午餐",
                search_terms=_healthy_meal_terms(text, ["午餐"]),
            )
        )

    insert_at = len(node_intents)
    for index, item in enumerate(expanded):
        if item.get("supply_domain") == "restaurant":
            insert_at = index + 1 if item.get("role") == "restaurant_lunch" else index
            break

    expanded.insert(insert_at, supplemental)
    while len(expanded) < FULL_DAY_MIN_NODE_COUNT:
        expanded.append(
            _make_intent_from_role(
                _chat_or_rest_role(text),
                label="傍晚餐饮/休息",
                search_terms=_healthy_meal_terms(text, ["傍晚", "餐饮", "休息"]),
            )
        )
    return _renumber_intents(expanded)


RESTAURANT_FOOD_TERMS = (
    "海鲜",
    "胶东菜",
    "鲁菜",
    "本地菜",
    "地方菜",
    "本帮菜",
    "上海菜",
    "江浙菜",
    "烤肉",
    "烧烤",
    "羊肉串",
    "火锅",
    "日料",
    "轻食",
    "清淡",
    "低卡",
    "健康餐",
    "少油",
    "咖啡",
    "甜品",
)
MEAL_CONTEXT_MARKERS = {
    "restaurant_breakfast": ("早餐", "早饭", "早上"),
    "restaurant_lunch": ("中午", "午餐", "午饭", "中饭"),
    "restaurant_dinner": ("晚上", "晚餐", "晚饭", "晚上吃"),
}
MEAL_CONTEXT_BOUNDARIES = (
    "早上",
    "上午",
    "中午",
    "午餐",
    "午饭",
    "下午",
    "晚上",
    "晚餐",
    "晚饭",
    "夜宵",
)


def _matched_restaurant_terms(text: str, hits: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for hit in hits:
        if str(hit.get("supply_domain") or "") == "restaurant":
            values.extend(str(term).strip() for term in hit.get("matched_terms", []) or [])
    values.extend(term for term in RESTAURANT_FOOD_TERMS if term in text)
    return _dedupe_keep_order(values)


def _meal_context_segment(text: str, role: str) -> str:
    markers = MEAL_CONTEXT_MARKERS.get(role) or ()
    positions = [
        (text.find(marker), marker)
        for marker in markers
        if text.find(marker) >= 0
    ]
    if not positions:
        return ""
    start, marker = min(positions, key=lambda item: item[0])
    segment_start = start + len(marker)
    segment_end = min(len(text), segment_start + 32)
    for boundary in MEAL_CONTEXT_BOUNDARIES:
        boundary_index = text.find(boundary, segment_start)
        if boundary_index > segment_start and boundary_index < segment_end:
            segment_end = boundary_index
    return text[start:segment_end]


def _meal_role_context_terms(text: str, role: str) -> list[str]:
    segment = _meal_context_segment(text, role)
    if not segment:
        return []
    return _dedupe_keep_order(term for term in RESTAURANT_FOOD_TERMS if term in segment)


def _matched_activity_terms(hits: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for hit in hits:
        if str(hit.get("supply_domain") or "") == "activity":
            values.extend(str(term).strip() for term in hit.get("matched_terms", []) or [])
    return _dedupe_keep_order(values)


def _default_hit(
    *,
    role: str,
    label: str,
    supply_domain: str,
    duration_min: int,
    support: str,
    terms: list[str] | None = None,
    position: int = 0,
) -> dict[str, Any]:
    return {
        "role": role,
        "label": label,
        "supply_domain": supply_domain,
        "default_duration_min": duration_min,
        "current_support": support,
        "matched_terms": terms or [],
        "first_position": position,
    }


def _expand_default_hits_for_horizon(
    hits: list[dict[str, Any]],
    *,
    horizon: str,
    text: str,
    state: PlanState,
) -> list[dict[str, Any]]:
    if horizon not in {"full_day", "two_day"}:
        return hits

    supported_pair = (
        len(hits) <= 2
        and all(str(hit.get("supply_domain") or "") in {"activity", "restaurant"} for hit in hits)
    )
    if not supported_pair:
        return hits

    scene_type = str(state.get("scene_type") or "")
    is_family = scene_type == "family" or any(word in text for word in ("亲子", "孩子", "小朋友", "带娃"))
    activity_terms = _matched_activity_terms(hits) or (
        ["亲子", "低强度"] if is_family else ["citywalk", "本地文化", "景观", "轻松"]
    )
    restaurant_terms = _matched_restaurant_terms(text, hits)
    morning_role = "family_activity" if is_family else "citywalk_market"
    afternoon_role = "family_indoor_play" if is_family else "park_scenic_walk"
    morning_label = "上午亲子活动" if is_family else "上午城市漫步"
    afternoon_label = "下午室内亲子活动" if is_family else "下午景观放松"

    if horizon == "two_day":
        return [
            _default_hit(
                role=morning_role,
                label="Day1活动",
                supply_domain="activity",
                duration_min=120,
                support="legacy_activity",
                terms=activity_terms,
                position=0,
            ),
            _default_hit(
                role="restaurant_dinner",
                label="Day1晚餐",
                supply_domain="restaurant",
                duration_min=80,
                support="legacy_restaurant",
                terms=restaurant_terms,
                position=1,
            ),
            _default_hit(
                role=afternoon_role,
                label="Day2活动",
                supply_domain="activity",
                duration_min=120,
                support="legacy_activity",
                terms=activity_terms,
                position=2,
            ),
            _default_hit(
                role="restaurant_lunch",
                label="Day2午餐",
                supply_domain="restaurant",
                duration_min=70,
                support="legacy_restaurant",
                terms=restaurant_terms,
                position=3,
            ),
        ]

    return [
        _default_hit(
            role=morning_role,
            label=morning_label,
            supply_domain="activity",
            duration_min=120,
            support="legacy_activity",
            terms=activity_terms,
            position=0,
        ),
        _default_hit(
            role="restaurant_lunch",
            label="午餐",
            supply_domain="restaurant",
            duration_min=70,
            support="legacy_restaurant",
            terms=restaurant_terms,
            position=1,
        ),
        _default_hit(
            role=afternoon_role,
            label=afternoon_label,
            supply_domain="activity",
            duration_min=120,
            support="legacy_activity",
            terms=activity_terms,
            position=2,
        ),
        _default_hit(
            role="restaurant_dinner",
            label="晚餐",
            supply_domain="restaurant",
            duration_min=80,
            support="legacy_restaurant",
            terms=restaurant_terms,
            position=3,
        ),
    ]


def _time_range_for_role(role: str, sequence_index: int, horizon: str) -> tuple[int, str, str, str]:
    """Return day, start, end, and part-of-day for an intent role."""

    if role == "lodging":
        if sequence_index > 1:
            return 1, "20:30", "次日10:00", "overnight"
        return 1, "15:00", "次日10:00", "overnight"
    if horizon == "two_day" and sequence_index >= 4:
        if role == "restaurant_breakfast":
            return 2, "08:30", "09:15", "breakfast"
        if role in {"restaurant_lunch", "restaurant_specific"}:
            return 2, "12:00", "13:10", "lunch"
        if role == "restaurant_dinner":
            return 2, "18:00", "19:20", "dinner"
        if role == "cafe":
            return 2, "15:00", "16:00", "afternoon"
        if role in {"souvenir_shopping", "convenience_store", "parking"}:
            return 2, "13:40", "14:20", "afternoon"
        return 2, "10:00", "12:00", "morning"
    if role == "restaurant_breakfast":
        return 1, "08:30", "09:15", "breakfast"
    if role == "restaurant_lunch":
        return 1, "12:00", "13:10", "lunch"
    if role == "restaurant_dinner":
        return 1, "18:00", "19:20", "dinner"
    if role == "restaurant_specific":
        return 1, "18:00", "19:15", "dinner"
    if role == "family_indoor_play":
        return 1, "14:30", "17:00", "afternoon"
    if role == "park_scenic_walk":
        if horizon in {"full_day", "two_day"} and sequence_index <= 3:
            return 1, "14:30", "16:30", "afternoon"
        return 1, "18:30", "19:45", "evening"
    if role == "river_cruise":
        return 1, "19:00", "21:00", "evening"
    if role == "cafe":
        return 1, "15:30", "16:30", "afternoon"
    if role == "board_game_escape":
        return 1, "16:00", "18:00", "late_afternoon"
    if role == "karaoke":
        return 1, "20:00", "22:00", "evening"
    if role == "bar":
        return 1, "22:00", "23:15", "late_evening"
    if role == "talk_show":
        return 1, "10:00", "11:40", "morning"
    if role == "theatre_performance":
        return 1, "19:00", "21:00", "evening"
    if role == "cinema":
        return 1, "19:30", "21:30", "evening"
    if role == "nail_salon":
        return 1, "15:00", "16:10", "afternoon"
    if role == "souvenir_shopping":
        return 1, "16:40", "17:30", "late_afternoon"
    if role == "beauty_cosmetics":
        return 1, "16:20", "16:55", "late_afternoon"
    if role == "flower_shop":
        return 1, "16:55", "17:20", "late_afternoon"
    if role == "wellness_massage":
        return 1, "16:30", "17:30", "late_afternoon"
    if role == "pet_grooming":
        return 1, "10:00", "11:30", "morning"
    if role == "pet_cafe":
        return 1, "12:00", "13:00", "lunch"
    if role == "pet_hospital":
        return 1, "14:00", "15:00", "afternoon"
    if role == "pet_store":
        return 1, "15:20", "15:55", "afternoon"
    if role == "convenience_store":
        return 1, "19:40", "20:00", "evening"
    if role == "parking":
        return 1, "20:00", "20:15", "evening"

    if horizon == "two_day" and sequence_index >= 4:
        return 2, "10:00", "12:00", "morning"
    if sequence_index <= 1 and horizon in {"full_day", "two_day"}:
        return 1, "10:00", "12:00", "morning"
    if sequence_index <= 2:
        return 1, "14:00", "16:00", "afternoon"
    return 1, "16:30", "18:00", "late_afternoon"


def _clock_to_minutes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def _format_clock(minutes: int) -> str:
    minutes = max(0, minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _combined_time_anchors(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for source in (
        constraints.get("time_anchors"),
        (constraints.get("explicit_constraints") or {}).get("time_anchors")
        if isinstance(constraints.get("explicit_constraints"), dict)
        else None,
    ):
        for anchor in _as_list(source):
            if isinstance(anchor, dict):
                anchors.append(anchor)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for anchor in anchors:
        key = (
            str(anchor.get("type") or ""),
            str(anchor.get("start_time") or ""),
            str(anchor.get("end_time") or ""),
            str(anchor.get("time") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(anchor)
    return deduped


def _part_of_day_for_minutes(minutes: int) -> str:
    clock = minutes % 1440
    if clock < 10 * 60:
        return "morning"
    if clock < 14 * 60:
        return "lunch"
    if clock < 17 * 60:
        return "afternoon"
    if clock < 21 * 60:
        return "evening"
    return "late_evening"


def _event_anchor_for_item(
    item: dict[str, Any],
    anchors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    role = str(item.get("role") or "")
    label = str(item.get("label") or "")
    search_terms = " ".join(str(term) for term in item.get("search_terms") or [])
    role_text = " ".join(value for value in (role, label, search_terms) if value)
    if role not in {"talk_show", "theatre_performance", "cinema", "river_cruise"} and not any(
        term in role_text
        for term in ("演出", "亲子剧", "儿童剧", "小剧场", "话剧", "脱口秀", "电影", "夜游")
    ):
        return None

    for anchor in anchors:
        if str(anchor.get("type") or "") != "event":
            continue
        if _clock_to_minutes(anchor.get("time")) is None:
            continue
        anchor_label = str(anchor.get("label") or "")
        if anchor_label and anchor_label in role_text:
            return anchor
        if any(term in role_text for term in ("演出", "亲子剧", "儿童剧", "小剧场", "话剧", "脱口秀", "电影", "夜游")):
            return anchor
    return None


def _apply_event_anchor_to_range(
    item: dict[str, Any],
    anchors: list[dict[str, Any]],
    *,
    fallback_start: str,
    fallback_end: str,
    fallback_part: str,
) -> tuple[str, str, str, dict[str, Any] | None]:
    anchor = _event_anchor_for_item(item, anchors)
    if anchor is None:
        return fallback_start, fallback_end, fallback_part, None

    start_minutes = _clock_to_minutes(anchor.get("time"))
    if start_minutes is None:
        return fallback_start, fallback_end, fallback_part, None
    try:
        duration = int(item.get("default_duration_min") or 100)
    except (TypeError, ValueError):
        duration = 100
    duration = max(45, min(duration, 150))
    return (
        _format_clock(start_minutes),
        _format_clock(start_minutes + duration),
        _part_of_day_for_minutes(start_minutes),
        anchor,
    )


def _protected_rest_slots(
    anchors: list[dict[str, Any]],
    *,
    day: int,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for anchor in anchors:
        if str(anchor.get("type") or "") != "rest":
            continue
        start_minutes = _clock_to_minutes(anchor.get("start_time"))
        end_minutes = _clock_to_minutes(anchor.get("end_time"))
        if start_minutes is None or end_minutes is None:
            continue
        if end_minutes <= start_minutes:
            end_minutes += 1440
        label = str(anchor.get("label") or "").strip() or "午睡/休息"
        slots.append(
            {
                "node_id": None,
                "role": "rest",
                "label": label,
                "supply_domain": "planning_guidance",
                "day": day,
                "start_time": _format_clock(start_minutes),
                "end_time": _format_clock(end_minutes),
                "part_of_day": "protected_rest",
                "duration_min": end_minutes - start_minutes,
                "execution_status": "protected_non_executable",
                "protected": True,
                "anchor_type": "rest",
                "sequence_index": start_minutes,
            }
        )
    return slots


def _time_bounds_for_skeleton_day(
    constraints: dict[str, Any],
    *,
    day: int,
    planning_days: int,
) -> tuple[int | None, int | None]:
    min_start = _clock_to_minutes(constraints.get("start_time")) if day == 1 else None
    max_end = (
        _clock_to_minutes(constraints.get("end_time"))
        if day == planning_days
        else None
    )
    if planning_days == 1 and min_start is not None and max_end is not None and max_end <= min_start:
        max_end += 1440
    return min_start, max_end


def _fit_slot_to_time_bounds(
    start: str,
    end: str,
    *,
    duration_min: Any,
    min_start: int | None,
    max_end: int | None,
) -> tuple[str, str]:
    start_minutes = _clock_to_minutes(start)
    end_minutes = _clock_to_minutes(end)
    if start_minutes is None or end_minutes is None:
        return start, end
    duration = end_minutes - start_minutes
    if duration <= 0:
        try:
            duration = int(duration_min)
        except (TypeError, ValueError):
            duration = 60
    duration = max(15, duration)

    if min_start is not None and start_minutes < min_start:
        start_minutes = min_start
        end_minutes = start_minutes + duration
    if max_end is not None and end_minutes > max_end:
        latest_start = max_end - duration
        if min_start is None or latest_start >= min_start:
            start_minutes = latest_start
            end_minutes = max_end
        else:
            start_minutes = min_start
            end_minutes = max_end
    return _format_clock(start_minutes), _format_clock(end_minutes)


def _time_range_minutes(start: str, end: str, duration_min: Any) -> tuple[int | None, int | None, int]:
    start_minutes = _clock_to_minutes(start)
    end_minutes = _clock_to_minutes(end)
    try:
        duration = int(duration_min)
    except (TypeError, ValueError):
        duration = 60
    duration = max(15, duration)
    if start_minutes is None or end_minutes is None:
        return start_minutes, end_minutes, duration
    if end_minutes <= start_minutes:
        end_minutes += 1440
    return start_minutes, end_minutes, end_minutes - start_minutes


def _duration_for_bounded_sequence(
    slot: dict[str, Any],
    *,
    max_end: int | None,
) -> int:
    start_minutes = _clock_to_minutes(str(slot.get("start_time") or ""))
    end_minutes = _clock_to_minutes(str(slot.get("end_time") or ""))
    if start_minutes is not None and end_minutes is not None:
        if max_end is not None and max_end > 1440 and end_minutes <= start_minutes:
            end_minutes += 1440
        if end_minutes > start_minutes:
            return max(1, end_minutes - start_minutes)
    try:
        duration = int(slot.get("duration_min"))
    except (TypeError, ValueError):
        duration = 60
    return max(1, duration)


def _slot_start_minutes(slot: dict[str, Any], *, default: int = 0) -> int:
    start = _clock_to_minutes(str(slot.get("start_time") or ""))
    if start is None:
        return default
    day = max(1, int(slot.get("day") or 1))
    return start + (day - 1) * 1440


def _slot_end_minutes(slot: dict[str, Any], *, max_end: int | None = None) -> int | None:
    start = _clock_to_minutes(str(slot.get("start_time") or ""))
    end = _clock_to_minutes(str(slot.get("end_time") or ""))
    if start is None or end is None:
        return None
    if end <= start:
        end += 1440
    if max_end is not None and max_end > 1440 and end < max_end - 720:
        end += 1440
    return end


def _slots_fit_existing_order(
    slots: list[dict[str, Any]],
    *,
    min_start: int | None,
    max_end: int | None,
) -> bool:
    previous_end = min_start
    for slot in slots:
        start = _clock_to_minutes(str(slot.get("start_time") or ""))
        end = _slot_end_minutes(slot, max_end=max_end)
        if start is None or end is None:
            return False
        if min_start is not None and start < min_start:
            return False
        if max_end is not None and end > max_end:
            return False
        if previous_end is not None and start < previous_end:
            return False
        previous_end = end
    return True


def _fit_durations_to_window(durations: list[int], window: int) -> list[int]:
    if not durations:
        return []
    if window <= 0:
        return [1 for _ in durations]
    total = sum(durations)
    if total <= window:
        return durations

    slot_count = len(durations)
    min_duration = 15 if window >= 15 * slot_count else max(1, window // slot_count)
    fitted = [min_duration for _ in durations]
    remaining_window = window - sum(fitted)
    reducible_total = sum(max(0, duration - min_duration) for duration in durations)
    if remaining_window > 0 and reducible_total > 0:
        for index, duration in enumerate(durations):
            extra = int((max(0, duration - min_duration) / reducible_total) * remaining_window)
            fitted[index] += extra

    remainder = max(0, window - sum(fitted))
    fitted[-1] += remainder
    return fitted


def _sequence_slots_for_day(
    slots: list[dict[str, Any]],
    *,
    min_start: int | None,
    max_end: int | None,
) -> list[dict[str, Any]]:
    """Keep same-day skeleton slots sequential after fixed role defaults are applied."""

    if len(slots) <= 1:
        return slots

    ordered = sorted(slots, key=lambda slot: int(slot.get("sequence_index") or 0))
    if (max_end is not None and min_start is None) or any(
        slot.get("part_of_day") == "overnight" for slot in slots
    ):
        ordered = sorted(
            slots,
            key=lambda slot: (
                _slot_start_minutes(slot),
                int(slot.get("sequence_index") or 0),
            ),
        )

    has_fixed_slots = any(
        slot.get("protected") is True or slot.get("anchor_type") == "event"
        for slot in ordered
    )
    if has_fixed_slots:
        ordered = sorted(
            ordered,
            key=lambda slot: (
                _slot_start_minutes(slot),
                0 if (slot.get("protected") is True or slot.get("anchor_type") == "event") else 1,
                int(slot.get("sequence_index") or 0),
            ),
        )
        fixed_starts = [
            _slot_start_minutes(slot)
            for slot in ordered
            if slot.get("protected") is True or slot.get("anchor_type") == "event"
        ]
        cursor = min_start
        fitted: list[dict[str, Any]] = []
        for slot in ordered:
            start_minutes, end_minutes, duration = _time_range_minutes(
                str(slot.get("start_time") or ""),
                str(slot.get("end_time") or ""),
                slot.get("duration_min"),
            )
            if start_minutes is None or end_minutes is None:
                fitted.append(slot)
                continue
            is_fixed = slot.get("protected") is True or slot.get("anchor_type") == "event"
            if is_fixed:
                if max_end is not None and start_minutes >= max_end:
                    continue
                if max_end is not None and end_minutes > max_end and slot.get("anchor_type") != "event":
                    end_minutes = max_end
                slot["start_time"] = _format_clock(start_minutes)
                slot["end_time"] = _format_clock(end_minutes)
                cursor = max(cursor or end_minutes, end_minutes)
                fitted.append(slot)
                continue

            if cursor is not None and start_minutes < cursor:
                start_minutes = cursor
                end_minutes = start_minutes + duration
            next_fixed_start = next(
                (fixed_start for fixed_start in fixed_starts if fixed_start > start_minutes),
                None,
            )
            if next_fixed_start is not None and end_minutes > next_fixed_start:
                latest_start = next_fixed_start - duration
                if cursor is None or latest_start >= cursor:
                    start_minutes = max(cursor or latest_start, latest_start)
                    end_minutes = next_fixed_start
                elif next_fixed_start - start_minutes >= 15:
                    end_minutes = next_fixed_start
                    slot["time_window_overflow_min"] = duration - (end_minutes - start_minutes)
                else:
                    slot["time_window_overflow_min"] = duration
                    continue
            if max_end is not None and end_minutes > max_end:
                end_minutes = max_end
                if end_minutes <= start_minutes:
                    continue
            slot["start_time"] = _format_clock(start_minutes)
            slot["end_time"] = _format_clock(end_minutes)
            cursor = end_minutes
            fitted.append(slot)
        return fitted

    if min_start is not None and max_end is not None and max_end > min_start:
        if _slots_fit_existing_order(ordered, min_start=min_start, max_end=max_end):
            return ordered

        window = max_end - min_start
        durations = [
            _duration_for_bounded_sequence(slot, max_end=max_end)
            for slot in ordered
        ]
        total_duration = sum(durations)
        if total_duration > window and window / max(1, total_duration) < 0.75:
            for slot in ordered:
                slot["time_window_overflow_min"] = total_duration - window
        else:
            durations = _fit_durations_to_window(durations, window)
        cursor = min_start
        for slot, duration in zip(ordered, durations):
            slot["start_time"] = _format_clock(cursor)
            cursor += duration
            slot["end_time"] = _format_clock(cursor)
        return ordered

    cursor = min_start
    fitted: list[dict[str, Any]] = []
    for slot in ordered:
        start_minutes, end_minutes, duration = _time_range_minutes(
            str(slot.get("start_time") or ""),
            str(slot.get("end_time") or ""),
            slot.get("duration_min"),
        )
        if start_minutes is None or end_minutes is None:
            fitted.append(slot)
            continue
        if max_end is not None and start_minutes >= max_end:
            continue
        if cursor is not None and start_minutes < cursor:
            start_minutes = cursor
            end_minutes = start_minutes + duration
        if max_end is not None and end_minutes > max_end:
            end_minutes = max_end
            if end_minutes <= start_minutes:
                continue
        slot["start_time"] = _format_clock(start_minutes)
        slot["end_time"] = _format_clock(end_minutes)
        cursor = end_minutes
        fitted.append(slot)
    return fitted


def _build_time_skeleton(
    node_intents: list[dict[str, Any]],
    *,
    horizon: str,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    constraints = constraints or {}
    anchors = _combined_time_anchors(constraints)
    raw_text = str(constraints.get("raw_text") or "")
    days: dict[int, list[dict[str, Any]]] = {}
    if horizon == "two_day":
        has_explicit_day_index = any(
            item.get("day_index") not in (None, "")
            for item in node_intents
        )
        two_day_split_after = (
            None
            if has_explicit_day_index
            else 1 if len(node_intents) <= 3 else max(2, len(node_intents) // 2)
        )
    else:
        two_day_split_after = None
    day_sequence_counts: dict[int, int] = {}
    for item in node_intents:
        role = str(item.get("role") or "")
        sequence_index = int(item.get("sequence_index") or 1)
        explicit_day: int | None = None
        if item.get("day_index") not in (None, ""):
            explicit_day = max(1, int(item.get("day_index") or 1))
            day_sequence_counts[explicit_day] = day_sequence_counts.get(explicit_day, 0) + 1
            sequence_index = day_sequence_counts[explicit_day]
        day, start, end, part = _time_range_for_role(
            role,
            sequence_index,
            horizon,
        )
        if explicit_day is not None:
            day = explicit_day
        if (
            horizon == "two_day"
            and role == "lodging"
            and day >= 2
            and not re.search(r"(第二天|次日|周日|星期日)[^，。；;]{0,18}(住|住宿|酒店|民宿)", raw_text)
        ):
            day = 1
            start, end, part = "20:30", "次日10:00", "overnight"
        if (
            explicit_day is None
            and two_day_split_after is not None
            and sequence_index > two_day_split_after
            and role != "lodging"
        ):
            day = 2
            if role == "restaurant_breakfast":
                start, end, part = "08:30", "09:15", "breakfast"
            elif role == "restaurant_lunch":
                start, end, part = "12:00", "13:10", "lunch"
            elif role == "restaurant_dinner":
                start, end, part = "18:00", "19:20", "dinner"
            elif role == "cafe":
                start, end, part = "10:00", "11:00", "morning"
            elif role == "talk_show":
                start, end, part = "14:00", "15:40", "afternoon"
            elif role == "souvenir_shopping":
                start, end, part = "16:10", "16:55", "late_afternoon"
            elif role not in {"lodging", "convenience_store", "parking"}:
                start, end, part = "10:00", "12:00", "morning"
        if role in {"restaurant_specific", "restaurant_dinner"} and any(
            term in (item.get("search_terms") or []) for term in ("夜宵", "宵夜")
        ):
            start, end, part = "21:20", "22:30", "late_evening"
        start, end, part, event_anchor = _apply_event_anchor_to_range(
            item,
            anchors,
            fallback_start=start,
            fallback_end=end,
            fallback_part=part,
        )
        duration_min = item.get("default_duration_min")
        planning_days = _planning_days(horizon)
        min_start, max_end = _time_bounds_for_skeleton_day(
            constraints,
            day=day,
            planning_days=planning_days,
        )
        start, end = _fit_slot_to_time_bounds(
            start,
            end,
            duration_min=duration_min,
            min_start=min_start,
            max_end=max_end,
        )
        entry = {
            "node_id": item.get("node_id"),
            "role": role,
            "label": item.get("label"),
            "supply_domain": item.get("supply_domain"),
            "day": day,
            "start_time": start,
            "end_time": end,
            "part_of_day": part,
            "duration_min": duration_min,
            "execution_status": "needs_candidate",
            "sequence_index": sequence_index,
        }
        if event_anchor is not None:
            entry["anchor_type"] = "event"
            entry["anchor_label"] = event_anchor.get("label")
        days.setdefault(day, []).append(entry)

    planning_days = _planning_days(horizon)
    if anchors:
        days.setdefault(1, []).extend(_protected_rest_slots(anchors, day=1))
    day_skeletons = []
    for day_index in range(1, planning_days + 1):
        slots = days.get(day_index, [])
        min_start, max_end = _time_bounds_for_skeleton_day(
            constraints,
            day=day_index,
            planning_days=planning_days,
        )
        slots = _sequence_slots_for_day(slots, min_start=min_start, max_end=max_end)
        for slot in slots:
            slot.pop("sequence_index", None)
        day_skeletons.append(
            {
                "day": day_index,
                "label": f"Day {day_index}",
                "slots": slots,
                "estimated_active_duration_min": sum(
                    int(slot.get("duration_min") or 0)
                    for slot in slots
                    if slot.get("part_of_day") not in {"overnight", "protected_rest"}
                ),
            }
        )

    return {
        "planning_days": planning_days,
        "planning_horizon": horizon,
        "days": day_skeletons,
    }


def build_b_itinerary_blueprint(
    state: PlanState,
    *,
    constraints: dict[str, Any] | None = None,
    scenario_activities: list[str] | None = None,
) -> dict[str, Any]:
    """Build an itinerary skeleton for B framework diagnostics and future planning."""

    del scenario_activities  # Future hook for A-provided node hints.
    constraints = constraints or state.get("constraints", {}) or {}
    text = _collect_text(state, constraints)
    hits = _drop_false_soft_tag_hits(
        _merge_restaurant_roles(_role_hits(text)),
        state=state,
        constraints=constraints,
    )

    if not hits:
        hits = [
            {
                "role": "activity",
                "label": "活动",
                "supply_domain": "activity",
                "default_duration_min": 120,
                "current_support": "legacy_activity",
                "matched_terms": [],
                "first_position": 0,
            },
            {
                "role": "restaurant_dinner",
                "label": "餐饮",
                "supply_domain": "restaurant",
                "default_duration_min": 80,
                "current_support": "legacy_restaurant",
                "matched_terms": [],
                "first_position": 1,
            },
        ]

    horizon = _planning_horizon(text, len(hits), constraints)
    hits = _expand_default_hits_for_horizon(
        hits,
        horizon=horizon,
        text=text,
        state=state,
    )
    for hit in hits:
        role = str(hit.get("role") or "")
        if role in MEAL_CONTEXT_MARKERS:
            hit["matched_terms"] = _dedupe_keep_order(
                hit.get("matched_terms", []) + _meal_role_context_terms(text, role)
            )

    node_intents: list[dict[str, Any]] = []
    for index, hit in enumerate(hits, start=1):
        node_intents.append(
            {
                "node_id": f"intent_{index:02d}",
                "role": hit["role"],
                "label": hit["label"],
                "supply_domain": hit["supply_domain"],
                "search_terms": hit.get("matched_terms", []),
                "default_duration_min": hit["default_duration_min"],
                "current_support": hit["current_support"],
                "sequence_index": index,
            }
        )

    horizon = _planning_horizon(text, len(node_intents), constraints=constraints)
    node_intents = _expand_sparse_full_day_intents(
        node_intents,
        text=text,
        horizon=horizon,
    )
    node_intents = _expand_sparse_two_day_intents(
        node_intents,
        text=text,
        horizon=horizon,
    )

    supported_domains = {"activity", "restaurant"}
    unsupported_roles = [
        item["role"]
        for item in node_intents
        if item["supply_domain"] not in supported_domains
    ]
    domain_counts = {
        "activity": sum(1 for item in node_intents if item["supply_domain"] == "activity"),
        "restaurant": sum(1 for item in node_intents if item["supply_domain"] == "restaurant"),
    }
    is_legacy_pair_shape = (
        len(node_intents) == 2
        and domain_counts["activity"] == 1
        and domain_counts["restaurant"] == 1
        and not unsupported_roles
    )
    has_non_pair_shape = (
        len(node_intents) > 2
        or bool(unsupported_roles)
        or (len(node_intents) == 2 and not is_legacy_pair_shape)
    )
    exact_entities = _named_entities(text)
    single_restaurant_search = (
        len(node_intents) == 1
        and node_intents[0].get("role") in {"restaurant_specific", "restaurant_lunch", "restaurant_dinner", "cafe"}
        and bool(node_intents[0].get("search_terms"))
    )
    requires_rag = bool(
        exact_entities
        or unsupported_roles
        or has_non_pair_shape
        or single_restaurant_search
        or any(word in text for word in ("附近有哪些", "有什么推荐", "哪吃", "哪里", "哪家"))
    )
    time_skeleton = _build_time_skeleton(
        node_intents,
        horizon=horizon,
        constraints=constraints,
    )

    return {
        "version": "b_itinerary_blueprint_v1",
        "source": "deterministic",
        "template_mode": "multi_node" if has_non_pair_shape else "legacy_pair",
        "planning_horizon": horizon,
        "planning_days": time_skeleton["planning_days"],
        "node_count": len(node_intents),
        "node_intents": node_intents,
        "time_skeleton": time_skeleton,
        "route_pattern": ["start"] + [item["role"] for item in node_intents],
        "unsupported_roles": unsupported_roles,
        "requires_rag": requires_rag,
        "named_entities": exact_entities,
        "sequence_markers": [word for word in SEQUENCE_WORDS if word in text],
        "framework_notes": _framework_notes(has_non_pair_shape, unsupported_roles, requires_rag),
    }


def _framework_notes(
    has_non_pair_shape: bool,
    unsupported_roles: list[str],
    requires_rag: bool,
) -> list[str]:
    notes: list[str] = []
    if has_non_pair_shape:
        notes.append("用户需求不是标准活动+餐厅半天模板，需要多节点 itinerary planner")
    if unsupported_roles:
        notes.append("当前结构化供给缺少部分业态，需要通用 POI/RAG 候选池补足")
    if requires_rag:
        notes.append("需要检索证据确认具体地点、营业时间、价格或用户指定活动")
    if not notes:
        notes.append("可继续使用 legacy activity+restaurant pair planner")
    return notes


def apply_b_itinerary_blueprint(
    state: PlanState,
    *,
    constraints: dict[str, Any],
    scenario_activities: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach the itinerary blueprint to constraints for downstream B nodes."""

    existing = constraints.get("b_itinerary_blueprint")
    if isinstance(existing, dict) and existing:
        return constraints, existing

    blueprint = build_b_itinerary_blueprint(
        state,
        constraints=constraints,
        scenario_activities=scenario_activities,
    )
    enhanced_constraints = dict(constraints)
    enhanced_constraints["b_itinerary_blueprint"] = blueprint
    return enhanced_constraints, blueprint
