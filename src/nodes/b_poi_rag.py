"""Local POI RAG retrieval for B-stage itinerary planning.

This node is the replaceable boundary between the future vector/RAG service and
B's deterministic planner.  It retrieves candidate POIs, not final itineraries,
and emits ``b_rag_candidate_evidence`` for ``candidate_generator_node``.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict

from .b_itinerary_blueprint import build_b_itinerary_blueprint
from .b_poi_memory_index import (
    DEFAULT_DOCUMENT_FIELDS,
    POI_MEMORY_INDEX_VERSION,
    build_poi_memory_index,
    load_poi_memory_index_cache,
    retrieve_poi_memory_candidates,
    save_poi_memory_index_cache,
)
from .b_requirement_compiler import apply_b_requirement_contract
from .b_semantics import (
    B_RESTAURANT_INTENT_GROUPS,
    B_SEMANTIC_GROUPS,
    b_semantic_terms,
    flatten_semantic_values,
    normalize_semantic_text,
    semantic_groups_in_values,
    semantic_terms_for_groups,
)
from .b_utils import to_float


DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "mock_data"
DEFAULT_INDEX_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "artifacts"
    / "b_poi_memory_index_cache"
)
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on", "auto"}
FALSY_ENV_VALUES = {"0", "false", "no", "off", "disabled"}
SUPPORTED_CONTRACT_VERSION = "b_rag_candidate_evidence_v1"
DEFAULT_TOP_K_PER_NODE = 12
DEFAULT_MAX_SCAN_PER_DOMAIN = 60000
GENERIC_POI_FALLBACK_VERSION = "generic_poi_fallback_v4"
FAST_ROLE_PREFILTER_CACHE_VERSION = "fast_role_prefilter_cache_v1"
DEFAULT_AUXILIARY_MAX_BYTES = 5_000_000
STRICT_ROLE_MATCH_ROLES = {
    "family_activity",
    "family_indoor_play",
    "exhibition",
    "river_cruise",
    "citywalk_market",
    "park_scenic_walk",
    "board_game_escape",
    "internet_cafe",
    "karaoke",
    "bar",
    "talk_show",
    "theatre_performance",
    "cinema",
    "cafe",
    "cultural_photo",
    "tea_house",
    "souvenir_shopping",
    "beauty_cosmetics",
    "nail_salon",
    "flower_shop",
    "convenience_store",
    "wellness_massage",
    "pet_grooming",
    "pet_hospital",
    "pet_store",
    "parking",
    "dental_clinic",
    "sports_training",
    "travel_agency",
}
FAST_DOMAIN_ROLE_PREFILTER_ROLES = {
    "bar",
    "karaoke",
    "exhibition",
    "board_game_escape",
    "internet_cafe",
    "talk_show",
    "theatre_performance",
    "cinema",
    "restaurant_specific",
    "dental_clinic",
    "sports_training",
    "travel_agency",
}

DOMAIN_STEMS = {
    "activity": ("activities",),
    "restaurant": ("restaurants",),
    "shopping": ("shopping", "shops", "retail"),
    "retail": ("retail", "shopping", "shops"),
    "hotel": ("hotels", "lodging"),
    "lodging": ("hotels", "lodging"),
    "wellness": ("wellness",),
    "local_service": ("local_service",),
    "pet_service": ("pet_service",),
    "beauty_service": ("beauty_service",),
    "transport_service": ("transport_services", "parking"),
}
GENERIC_POI_FALLBACK_DOMAINS = {
    "shopping",
    "retail",
    "hotel",
    "transport_service",
    "pet_service",
    "beauty_service",
    "local_service",
}
DOMAIN_FALLBACK_HINTS = {
    "shopping": (
        "购物服务",
        "便民商店",
        "便利店",
        "超级市场",
        "超市",
        "土特产",
        "特产",
        "礼品饰品",
        "礼品",
        "专卖店",
        "百货",
    ),
    "retail": (
        "购物服务",
        "个人用品",
        "化妆品",
        "美妆",
        "日化",
        "花鸟鱼虫",
        "花卉",
        "鲜花",
        "礼品饰品",
    ),
    "pet_service": (
        "宠物服务",
        "宠物美容",
        "宠物医院",
        "宠物诊所",
        "宠物用品",
        "宠物店",
    ),
    "beauty_service": (
        "生活服务",
        "美容美发",
        "美甲",
        "美睫",
        "丽人",
    ),
    "hotel": (
        "住宿服务",
        "酒店",
        "宾馆",
        "民宿",
        "公寓",
        "旅馆",
        "客栈",
    ),
    "transport_service": (
        "交通设施服务",
        "停车场",
        "停车",
    ),
    "local_service": (
        "医疗保健服务",
        "口腔",
        "牙科",
        "诊所",
        "科教文化服务",
        "培训机构",
        "体育培训",
        "足球",
        "生活服务",
        "旅行社",
        "签证",
        "出境游",
    ),
}
DOMAIN_DEFAULTS = {
    "shopping": {"type": "shopping", "price": 80.0, "duration_min": 25, "queue_time_min": 5},
    "retail": {"type": "shopping", "price": 120.0, "duration_min": 30, "queue_time_min": 5},
    "hotel": {"type": "hotel", "price": 420.0, "duration_min": 720, "queue_time_min": 10},
    "pet_service": {"type": "pet_service", "price": 180.0, "duration_min": 60, "queue_time_min": 10},
    "beauty_service": {"type": "beauty_service", "price": 160.0, "duration_min": 70, "queue_time_min": 10},
    "transport_service": {"type": "transport_service", "price": 25.0, "duration_min": 15, "queue_time_min": 5},
    "local_service": {"type": "local_service", "price": 180.0, "duration_min": 60, "queue_time_min": 10},
}
LOCATION_ANCHOR_TERMS = {
    "外滩",
    "黄浦江",
    "南京路",
    "南京东路",
    "南京西路",
    "人民广场",
    "上海博物馆",
    "静安",
    "静安寺",
    "大田路",
    "新天地",
    "淮海路",
    "豫园",
    "陆家嘴",
    "徐汇",
    "徐家汇",
    "徐家汇源",
    "徐汇滨江",
    "衡山路",
    "田子坊",
    "五角场",
    "复旦",
    "杨浦",
    "北外滩",
    "白玉兰广场",
    "虹桥",
    "松江",
    "嘉定",
    "浦东",
}

NAMED_EVENT_LOCATION_ANCHORS = {
    "zikawei": ("徐家汇", "徐汇", "徐家汇源", "徐汇滨江", "衡山路"),
    "海派zikawei": ("徐家汇", "徐汇", "徐家汇源", "徐汇滨江", "衡山路"),
    "夏日圆舞曲": ("徐家汇", "徐汇", "徐家汇源", "徐汇滨江", "衡山路"),
    "徐家汇源": ("徐家汇", "徐家汇源", "徐汇"),
    "古埃及文明大展": ("上海博物馆", "人民广场", "南京东路"),
    "埃及文明大展": ("上海博物馆", "人民广场", "南京东路"),
    "茶文化旅游节": ("静安", "大田路", "南京西路"),
    "儿童戏剧艺术节": ("静安", "南京西路"),
}

ROLE_QUERY_TERMS = {
    "family_activity": ("亲子", "儿童", "儿童友好", "室内", "低强度", "游乐园"),
    "family_indoor_play": ("室内乐园", "亲子乐园", "儿童乐园", "儿童游乐", "淘气堡", "蹦床", "游乐园"),
    "exhibition": ("展览", "看展", "博物馆", "美术馆", "艺术展", "文物展", "漆器展"),
    "citywalk_market": ("citywalk", "城市漫步", "市集", "街区", "本地生活", "历史文化", "历史建筑", "文化街区", "文化景区", "名街"),
    "park_scenic_walk": ("公园", "游园", "绿地", "滨江", "江边", "河边", "夜景", "散步", "夜游", "观景", "外滩"),
    "river_cruise": ("游船", "游轮", "邮轮", "黄浦江", "浦江游览", "夜游黄浦江", "包厢", "自助餐"),
    "board_game_escape": ("桌游", "棋牌", "剧本杀", "狼人杀", "密室", "密室逃脱", "推理馆"),
    "internet_cafe": ("网吧", "网咖", "电竞馆", "电竞", "上网", "电玩", "游戏", "通宵", "点播影院", "电影"),
    "karaoke": ("KTV", "唱歌", "卡拉OK", "练歌房", "欢唱"),
    "bar": ("酒吧", "清吧", "喝一杯", "小酌", "鸡尾酒", "精酿", "夜店"),
    "talk_show": ("脱口秀", "喜剧", "相声", "曲艺", "评弹", "剧场", "演出", "喜剧场", "livehouse"),
    "theatre_performance": ("话剧", "戏剧", "舞台剧", "儿童剧", "剧院", "剧场", "演出"),
    "cinema": ("电影", "影院", "电影院", "观影", "IMAX"),
    "restaurant_breakfast": ("早餐", "早饭", "早点", "包子", "馄饨"),
    "restaurant_lunch": ("午餐", "中饭", "正餐", "简餐"),
    "restaurant_dinner": ("晚餐", "晚饭", "正餐", "堂食"),
    "restaurant_specific": (
        "餐厅",
        "吃饭",
        "正餐",
        "聚餐",
        "宴请",
        "商务宴请",
        "重要客户",
        "夜宵",
        "扒房",
        "牛排",
        "西餐",
        "烧烤",
        "火锅",
        "本帮菜",
        "上海菜",
        "小笼包",
        "小笼",
        "生煎",
        "馄饨",
        "汤包",
    ),
    "cafe": ("咖啡", "咖啡馆", "下午茶", "甜品", "小坐"),
    "souvenir_shopping": ("特产", "伴手礼", "纪念品", "文创", "周边", "礼品", "礼物", "小礼物", "蛋糕", "生日蛋糕", "带回去"),
    "beauty_cosmetics": ("美妆", "日化", "化妆品", "护肤"),
    "nail_salon": ("美甲", "美睫", "甲油胶", "做指甲"),
    "flower_shop": ("鲜花", "花店", "花束", "花艺"),
    "convenience_store": ("便利店", "日用品", "超市", "买水", "零食"),
    "cultural_photo": ("汉服", "拍照", "写真", "摄影", "古装", "换装", "文化体验"),
    "tea_house": ("茶艺", "茶馆", "茶室", "品茶", "喝茶", "茶空间", "休息"),
    "wellness_massage": ("足疗", "按摩", "洗脚", "推拿", "休息"),
    "pet_grooming": ("宠物美容", "宠物spa", "宠物洗澡", "宠物洗护", "金毛", "毛发"),
    "pet_cafe": ("宠物友好咖啡", "宠物友好", "可带宠物", "带狗咖啡"),
    "pet_hospital": ("宠物医院", "宠物体检", "兽医", "动物医院"),
    "pet_store": ("宠物店", "宠物用品", "营养品", "狗粮", "猫粮"),
    "fitness": ("健身", "健身房", "瑜伽", "普拉提", "运动", "fitness", "yoga", "pilates"),
    "parking": ("停车", "停车场", "免费停车", "好停车"),
}

ROLE_REQUIRED_IDENTITY_TERMS = {
    "exhibition": (
        "美术馆",
        "博物馆",
        "展览",
        "展馆",
        "艺术馆",
        "画廊",
        "文化馆",
        "艺术",
        "历史",
    ),
    "family_activity": (
        "亲子",
        "儿童",
        "孩子",
        "小朋友",
        "带娃",
        "遛娃",
        "儿童友好",
        "儿童乐园",
        "室内乐园",
        "游乐园",
        "亲子馆",
        "手作",
        "DIY",
        "陶艺",
        "绘画",
        "乐高",
        "积木",
        "早教",
        "研学",
        "儿童剧",
        "绘本",
    ),
    "family_indoor_play": (
        "室内乐园",
        "亲子乐园",
        "儿童乐园",
        "儿童游乐",
        "室内游乐",
        "游乐园",
        "乐园",
        "淘气堡",
        "蹦床",
    ),
    "cultural_photo": (
        "汉服",
        "旗袍",
        "古风",
        "古装",
        "换装",
        "写真馆",
        "摄影馆",
        "照相馆",
    ),
    "tea_house": (
        "茶艺",
        "茶馆",
        "茶室",
        "品茶",
        "喝茶",
        "茶空间",
        "茶舍",
    ),
    "convenience_store": (
        "便利店",
        "便民商店",
        "超市",
        "全家",
        "罗森",
        "7-eleven",
        "711",
        "零食",
        "饮料",
    ),
    "parking": (
        "停车",
        "停车场",
        "车库",
        "车位",
        "parking",
    ),
    "board_game_escape": (
        "剧本杀",
        "密室",
        "桌游",
        "推理",
        "狼人杀",
        "血染钟楼",
    ),
    "internet_cafe": (
        "网吧",
        "网咖",
        "电竞",
        "电玩",
        "游戏机",
        "ps5",
        "ns",
    ),
    "bar": (
        "酒吧",
        "清吧",
        "精酿",
        "鸡尾酒",
        "live",
        "音乐",
    ),
    "talk_show": (
        "脱口秀",
        "喜剧",
        "剧场",
        "演出",
        "livehouse",
    ),
    "theatre_performance": (
        "话剧",
        "戏剧",
        "舞台剧",
        "儿童剧",
        "剧院",
        "剧场",
        "演出",
    ),
    "cinema": (
        "电影",
        "影院",
        "电影院",
        "观影",
        "IMAX",
        "imax",
    ),
    "citywalk_market": (
        "citywalk",
        "城市漫步",
        "街区",
        "市集",
        "老城",
        "老街",
        "历史",
        "文化",
        "文旅",
        "景区",
        "风景区",
        "观景",
        "海滨",
        "栈桥",
        "八大关",
        "中山路",
        "大鲍岛",
        "小麦岛",
        "奥帆",
        "博物馆",
        "美术馆",
        "故居",
        "旧址",
    ),
    "park_scenic_walk": (
        "公园",
        "游园",
        "绿地",
        "滨江",
        "江边",
        "河边",
        "夜景",
        "散步",
        "步道",
        "观景",
        "外滩",
    ),
    "river_cruise": (
        "游船",
        "游轮",
        "邮轮",
        "黄浦江",
        "浦江游览",
        "游览船",
        "观光船",
        "码头",
        "包厢",
        "自助餐",
    ),
    "dental_clinic": (
        "牙科",
        "口腔",
        "牙医",
        "洗牙",
        "补牙",
        "种植牙",
        "矫正",
    ),
    "sports_training": (
        "足球培训",
        "足球训练",
        "足球青训",
        "足球教练",
        "少儿足球",
        "青训",
    ),
    "travel_agency": (
        "旅行社",
        "出境游",
        "签证",
        "办签证",
        "旅游团",
        "跟团游",
        "特价旅游",
    ),
    "karaoke": (
        "KTV",
        "ktv",
        "唱歌",
        "卡拉OK",
        "卡拉ok",
        "练歌房",
        "欢唱",
        "karaoke",
    ),
    "flower_shop": (
        "鲜花",
        "花店",
        "花束",
        "花艺",
        "花坊",
        "flower",
    ),
    "beauty_cosmetics": (
        "美妆",
        "化妆品",
        "日化",
        "护肤",
        "彩妆",
        "香水",
        "cosmetics",
    ),
    "souvenir_shopping": (
        "特产",
        "土特产",
        "伴手礼",
        "纪念品",
        "文创",
        "周边",
        "礼品",
        "礼物",
        "老字号",
        "糕点",
        "蝴蝶酥",
        "带回去",
    ),
}

ROLE_EXCLUDED_IDENTITY_TERMS = {
    "exhibition": (
        "SPA",
        "spa",
        "足疗",
        "按摩",
        "推拿",
        "洗脚",
        "修脚",
        "健身",
        "瑜伽",
        "普拉提",
    ),
    "bar": (
        "火锅",
        "烤肉",
        "烧烤",
        "麻辣烫",
        "寿司",
        "牛排",
        "餐厅",
        "小吃",
    ),
    "park_scenic_walk": (
        "运动体验",
        "social_sports",
        "健身",
        "训练",
        "亲子乐园",
        "儿童乐园",
        "淘气堡",
        "蹦床",
    ),
    "citywalk_market": (
        "SPA",
        "spa",
        "足疗",
        "按摩",
        "推拿",
        "洗脚",
        "修脚",
        "瑜伽",
        "健身",
        "普拉提",
        "美容",
        "养生",
        "购物中心",
        "商场",
        "玩具",
        "专卖店",
        "专营店",
        "门店",
        "社区",
        "老年活动室",
        "活动室",
        "党群",
        "服务中心",
        "居委",
        "街道办",
        "酒吧",
        "清吧",
        "BAR",
        "bar",
        "club",
        "餐厅",
        "快餐",
        "真功夫",
        "肯德基",
        "麦当劳",
    ),
    "karaoke": (
        "展览",
        "展馆",
        "艺术",
        "画廊",
        "美术馆",
        "博物馆",
        "印象派",
        "餐厅",
        "酒吧",
        "livehouse",
        "足疗",
        "沐足",
        "按摩",
        "推拿",
        "洗脚",
        "修脚",
    ),
    "souvenir_shopping": (
        "咖啡",
        "咖啡馆",
        "咖啡厅",
        "cafe",
        "bar",
        "酒吧",
        "清吧",
        "餐厅",
        "便利店",
        "超市",
    ),
    "flower_shop": (
        "miniso",
        "名创",
        "优品",
        "购物中心",
        "商场",
        "超市",
        "便利店",
        "美妆",
        "日化",
        "餐厅",
    ),
    "restaurant_lunch": (
        "咖啡",
        "咖啡馆",
        "咖啡厅",
        "cafe",
        "bar",
        "酒吧",
        "清吧",
        "蛋糕",
        "甜品",
    ),
    "restaurant_dinner": (
        "咖啡",
        "咖啡馆",
        "咖啡厅",
        "cafe",
        "bar",
        "酒吧",
        "清吧",
        "蛋糕",
        "甜品",
    ),
    "river_cruise": (
        "公园",
        "绿地",
        "步道",
        "咖啡",
        "餐厅",
        "酒吧",
        "商场",
        "停车场",
        "写字楼",
    ),
    "family_activity": (
        "瑜伽",
        "普拉提",
        "健身",
        "健身中心",
        "SPA",
        "足疗",
        "按摩",
        "酒吧",
        "党群服务中心",
        "社区党群",
        "社区服务中心",
        "街道社区",
        "政务服务",
    ),
    "family_indoor_play": (
        "瑜伽",
        "普拉提",
        "健身",
        "SPA",
        "足疗",
        "按摩",
        "党群服务中心",
        "社区党群",
        "社区服务中心",
        "街道社区",
        "政务服务",
    ),
}

MEAL_ROLE_MARKERS = {
    "restaurant_breakfast": ("早上", "早餐", "早饭", "早茶", "早点"),
    "restaurant_lunch": ("中午", "午餐", "午饭", "中饭", "吃个中饭", "吃午饭"),
    "restaurant_dinner": ("晚上", "晚餐", "晚饭", "吃晚饭", "吃晚餐"),
}

MEAL_CONTEXT_STOP_WORDS = (
    "上午",
    "中午",
    "午餐",
    "午饭",
    "下午",
    "晚上",
    "晚餐",
    "晚饭",
    "饭后",
    "然后",
    "再",
    "最后",
)

MEAL_CONTEXT_TERM_EXPANSIONS = {
    "清淡": ("清淡", "轻食", "健康", "少油", "中式轻食", "沙拉"),
    "轻食": ("轻食", "健康", "低卡", "沙拉", "中式轻食"),
    "低卡": ("低卡", "轻食", "健康", "沙拉"),
    "减脂": ("减脂", "低卡", "轻食", "少油"),
    "少油": ("少油", "清淡", "健康"),
    "不油腻": ("不油腻", "清淡", "健康", "少油", "轻食"),
    "健康餐": ("健康餐", "健康", "轻食", "清淡", "少油", "低卡"),
    "素食": ("素食", "蔬食", "健康", "清淡"),
    "本帮": ("本帮菜", "上海菜", "家常菜"),
    "上海菜": ("上海菜", "本帮菜", "家常菜"),
    "小笼包": ("小笼包", "小笼", "汤包", "本帮小吃", "上海小吃"),
    "小笼": ("小笼包", "小笼", "汤包", "上海小吃"),
    "生煎": ("生煎", "生煎包", "上海小吃", "小吃"),
    "汤包": ("汤包", "小笼包", "上海小吃"),
    "火锅": ("火锅", "潮汕牛肉火锅", "川渝火锅"),
    "烤肉": ("烤肉", "烧烤", "日式烧肉", "炭火烤肉"),
    "烧烤": ("烧烤", "烤肉", "羊肉串", "炭火"),
    "羊肉串": ("羊肉串", "烧烤", "烤肉"),
    "咖啡": ("咖啡", "咖啡馆", "下午茶"),
    "甜品": ("甜品", "下午茶", "蛋糕"),
}

TEXT_FIELDS = (
    "name",
    "category",
    "sub_category",
    "restaurant_category",
    "primary_category",
    "gaode_keyword",
    "gaode_type",
    "address",
    "location",
    "business_area",
    "tags",
    "source_evidence",
    "review_keywords",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "package_options",
    "product_options",
    "deal_options",
    "_gaode_raw_business",
)
IDENTITY_TEXT_FIELDS = (
    "name",
    "category",
    "sub_category",
    "restaurant_category",
    "primary_category",
    "gaode_keyword",
    "gaode_type",
    "address",
    "location",
    "business_area",
    "tags",
    "review_keywords",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "_gaode_raw_business",
)
SERVICE_IDENTITY_TEXT_FIELDS = tuple(
    field
    for field in IDENTITY_TEXT_FIELDS
    if field not in {"address", "location", "business_area", "_gaode_raw_business"}
)
_TEXT_BLOB_CACHE_LIMIT = 120000
_TEXT_BLOB_CACHE: dict[tuple[str, tuple[str, ...]], tuple[str, set[str]]] = {}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY_ENV_VALUES


def _falsy(value: Any) -> bool:
    return str(value).strip().lower() in FALSY_ENV_VALUES


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


def _mock_data_dir() -> Path:
    override = os.environ.get("WF_B_RAG_DATA_DIR") or os.environ.get("WF_MOCK_DATA_DIR")
    if override:
        path = Path(override)
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
    return DEFAULT_MOCK_DATA_DIR


def _top_k_per_node() -> int:
    raw = os.environ.get("WF_B_RAG_TOP_K", "").strip()
    try:
        return max(1, min(50, int(raw)))
    except ValueError:
        return DEFAULT_TOP_K_PER_NODE


def _max_scan_per_domain() -> int:
    raw = os.environ.get("WF_B_RAG_MAX_SCAN", "").strip()
    try:
        return max(100, min(200000, int(raw)))
    except ValueError:
        return DEFAULT_MAX_SCAN_PER_DOMAIN


def _index_cache_enabled() -> bool:
    raw = os.environ.get("WF_B_RAG_INDEX_CACHE")
    if raw is None:
        return True
    return not _falsy(raw)


def _index_cache_dir() -> Path:
    override = os.environ.get("WF_B_RAG_INDEX_CACHE_DIR")
    if override:
        path = Path(override)
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
    return DEFAULT_INDEX_CACHE_DIR


def _records_signature(root: Path, stem: str) -> str:
    parts: list[str] = []
    for path in (root / f"{stem}.json", root / f"{stem}.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = root / shard_dir_name
        if not shard_dir.exists():
            continue
        for shard in sorted(shard_dir.glob("*.jsonl")):
            try:
                stat = shard.stat()
            except OSError:
                continue
            parts.append(f"{shard_dir.name}/{shard.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts) or f"{stem}:missing"


def _supply_signature(root: Path) -> str:
    stems = {
        "activities",
        "restaurants",
        "shopping",
        "shops",
        "retail",
        "hotels",
        "lodging",
        "wellness",
        "transport_services",
        "parking",
        "availability",
        "products",
        "deals",
        "merchants",
        "pet_service",
        "beauty_service",
    }
    return "|".join(_records_signature(root, stem) for stem in sorted(stems))


def _domain_signature(root: Path, domain: str) -> str:
    stems = DOMAIN_STEMS.get(domain, (domain,))
    parts = [
        f"index={POI_MEMORY_INDEX_VERSION}",
        f"fields={','.join(DEFAULT_DOCUMENT_FIELDS)}",
        f"max_scan={_max_scan_per_domain()}",
    ]
    parts.extend(_records_signature(root, stem) for stem in stems)
    if _normalize_domain(domain) in GENERIC_POI_FALLBACK_DOMAINS:
        parts.append(f"generic_fallback={GENERIC_POI_FALLBACK_VERSION}")
        parts.append(_records_signature(root, "deduped_pois"))
        parts.append(_records_signature(root, "raw_pois"))
        if _normalize_domain(domain) == "transport_service":
            parts.append(_records_signature(root, "activities"))
            parts.append(_records_signature(root, "restaurants"))
    return "|".join(parts)


def _cache_filename(root: Path, domain: str, signature: str) -> str:
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    root_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", root.name)[:80] or "mock_data"
    domain_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain)[:40] or "domain"
    return f"{root_name}_{domain_name}_{digest}.pkl"


def _fast_role_cache_filename(
    root: Path,
    *,
    domain: str,
    role: str,
    signature: str,
) -> str:
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    root_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", root.name)[:80] or "mock_data"
    domain_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain)[:40] or "domain"
    role_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", role)[:50] or "role"
    return f"{root_name}_{domain_name}_{role_name}_{digest}.fast.pkl"


def _load_fast_role_pool_cache(
    cache_path: Path,
    *,
    expected_signature: str,
) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    items = payload.get("items")
    if not isinstance(metadata, dict) or not isinstance(items, list):
        return None
    if metadata.get("cache_version") != FAST_ROLE_PREFILTER_CACHE_VERSION:
        return None
    if metadata.get("signature") != expected_signature:
        return None
    return [item for item in items if isinstance(item, dict)]


def _save_fast_role_pool_cache(
    cache_path: Path,
    *,
    signature: str,
    items: list[dict[str, Any]],
) -> bool:
    payload = {
        "metadata": {
            "cache_version": FAST_ROLE_PREFILTER_CACHE_VERSION,
            "signature": signature,
            "item_count": len(items),
        },
        "items": items,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        return True
    except Exception:
        return False


def _read_json(path: Path) -> Any:
    if not path.exists():
        return [] if path.suffix == ".json" else None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _items_from_json(raw: Any) -> list[dict[str, Any]]:
    items = raw.get("items", []) if isinstance(raw, dict) and isinstance(raw.get("items"), list) else raw
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    except Exception:
        return []
    return records


def _read_records(root: Path, stem: str) -> list[dict[str, Any]]:
    records = _items_from_json(_read_json(root / f"{stem}.json"))
    if records:
        return records

    records = _read_jsonl(root / f"{stem}.jsonl")
    if records:
        return records

    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = root / shard_dir_name
        if not shard_dir.exists():
            continue
        shard_records: list[dict[str, Any]] = []
        for shard in sorted(shard_dir.glob("*.jsonl")):
            shard_records.extend(_read_jsonl(shard))
        if shard_records:
            return shard_records
    return []


def _group_by_poi(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        poi_id = str(item.get("poi_id") or "").strip()
        if poi_id:
            grouped.setdefault(poi_id, []).append(item)
    return grouped


def _merchant_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        merchant_id = str(item.get("merchant_id") or "").strip()
        if merchant_id and merchant_id not in result:
            result[merchant_id] = item
    return result


@lru_cache(maxsize=8)
def _load_supply_bundle(root_key: str, signature: str) -> dict[str, Any]:
    del signature
    root = Path(root_key)
    return {
        "root": str(root),
        "domain_items": {},
        # Expensive auxiliary files are loaded lazily only after the structured
        # reranker has narrowed candidates. This keeps cold RAG startup focused
        # on the POI domain actually needed by the query.
        "availability": {},
        "products_by_poi": {},
        "deals_by_poi": {},
        "merchants": {},
    }


def _supply_bundle(root: Path) -> dict[str, Any]:
    return _load_supply_bundle(str(root.resolve()), _supply_signature(root))


def _bundle_availability(bundle: dict[str, Any]) -> dict[str, Any]:
    if not bundle.get("_availability_loaded"):
        root = Path(bundle.get("root") or _mock_data_dir())
        availability_raw = _read_json(root / "availability.json")
        bundle["availability"] = availability_raw if isinstance(availability_raw, dict) else {}
        bundle["_availability_loaded"] = True
    return bundle.get("availability", {})


def _bundle_products_by_poi(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not bundle.get("_products_by_poi_loaded"):
        root = Path(bundle.get("root") or _mock_data_dir())
        bundle["products_by_poi"] = _group_by_poi(_read_records(root, "products"))
        bundle["_products_by_poi_loaded"] = True
    return bundle.get("products_by_poi", {})


def _bundle_deals_by_poi(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not bundle.get("_deals_by_poi_loaded"):
        root = Path(bundle.get("root") or _mock_data_dir())
        bundle["deals_by_poi"] = _group_by_poi(_read_records(root, "deals"))
        bundle["_deals_by_poi_loaded"] = True
    return bundle.get("deals_by_poi", {})


def _bundle_merchants(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not bundle.get("_merchants_loaded"):
        root = Path(bundle.get("root") or _mock_data_dir())
        bundle["merchants"] = _merchant_by_id(_read_records(root, "merchants"))
        bundle["_merchants_loaded"] = True
    return bundle.get("merchants", {})


def _rag_auxiliary_merge_enabled(bundle: dict[str, Any]) -> bool:
    raw_enabled = os.environ.get("WF_B_RAG_AUXILIARY_ENABLED", "").strip()
    if raw_enabled:
        return _truthy(raw_enabled)
    try:
        max_bytes = int(
            os.environ.get("WF_B_RAG_AUXILIARY_MAX_BYTES", str(DEFAULT_AUXILIARY_MAX_BYTES))
        )
    except ValueError:
        max_bytes = DEFAULT_AUXILIARY_MAX_BYTES
    root = Path(bundle.get("root") or _mock_data_dir())
    total_bytes = 0
    for filename in ("availability.json", "products.json", "deals.json", "merchants.json"):
        path = root / filename
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        if total_bytes > max_bytes:
            return False
    return True


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
        name = str(item.get("name") or "").strip()
        key = poi_id or name
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_text(values: list[Any], *, limit: int = 80) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in flatten_semantic_values(values):
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _query_tokens(values: list[Any]) -> list[str]:
    raw: list[Any] = []
    for value in values:
        raw.extend(flatten_semantic_values(value))
        if isinstance(value, str):
            raw.extend(re.split(r"[\s,，。；;、/|+]+", value))
    raw.extend(b_semantic_terms(raw, include_auxiliary=True))
    tokens: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = normalize_semantic_text(str(value).strip())
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        tokens.append(text)
    return tokens


def _literal_query_tokens(values: list[Any]) -> list[str]:
    """Normalize exact identity guards without semantic broadening.

    Retrieval query terms can safely expand "公园" into neighboring concepts
    such as citywalk.  Identity guards are stricter: they decide whether a POI
    can truthfully fill a role, so expanding here would let a citywalk shop fill
    a park/night-walk node.
    """

    raw: list[Any] = []
    for value in values:
        raw.extend(flatten_semantic_values(value))
        if isinstance(value, str):
            raw.extend(re.split(r"[\s,，。；;、|+]+", value))
    tokens: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = normalize_semantic_text(str(value).strip())
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        tokens.append(text)
    return tokens


def _location_anchor_terms(values: list[Any]) -> list[str]:
    blob = " ".join(str(value) for value in flatten_semantic_values(values) if value not in (None, ""))
    return [term for term in sorted(LOCATION_ANCHOR_TERMS, key=len, reverse=True) if term in blob]


def _named_event_location_anchor_terms(state: PlanState, constraints: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(value)
        for value in (
            state.get("user_input"),
            constraints.get("raw_text"),
            constraints.get("named_entities"),
            state.get("b_itinerary_blueprint", {}).get("named_entities")
            if isinstance(state.get("b_itinerary_blueprint"), dict)
            else None,
        )
        if value not in (None, "")
    )
    normalized = normalize_semantic_text(text).lower()
    anchors: list[str] = []
    for marker, marker_anchors in NAMED_EVENT_LOCATION_ANCHORS.items():
        if normalize_semantic_text(marker).lower() in normalized:
            anchors.extend(marker_anchors)
    return _dedupe_text(anchors, limit=10)


def _inferred_location_anchor_terms(state: PlanState, constraints: dict[str, Any]) -> list[str]:
    """Infer narrow Shanghai anchors when the request describes a local style."""

    text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    city = str(constraints.get("city") or "")
    anchors: list[str] = []
    if "上海" in city or "上海" in text:
        anchors.extend(_named_event_location_anchor_terms(state, constraints))
        if "上海国际茶文化旅游节" in text or "茶文化旅游节" in text:
            anchors.extend(["静安", "大田路", "南京西路"])
        if "上海国际儿童戏剧艺术节" in text or "儿童戏剧艺术节" in text:
            anchors.extend(["静安", "南京西路"])
        if "古埃及文明大展" in text or "埃及文明大展" in text:
            anchors.extend(["上海博物馆", "人民广场", "南京东路"])
        if any(term in text for term in ("古风", "汉服", "古装", "茶馆", "茶艺")):
            anchors.extend(["豫园", "城隍庙"])
    if "静安" in text:
        anchors.extend(["静安", "静安寺", "南京西路"])
    if "复旦" in text:
        anchors.extend(["复旦", "五角场", "杨浦"])
    if "外滩" in text:
        anchors.extend(["外滩", "北外滩", "黄浦江", "白玉兰广场"])
    if "杨浦" in text:
        anchors.extend(["杨浦", "五角场"])
    if "浦东" in text:
        anchors.extend(["浦东", "陆家嘴"])
    if "徐家汇" in text:
        anchors.extend(["徐家汇"])
    return _dedupe_text(anchors, limit=8)


def _item_matches_location_terms(item: dict[str, Any], location_terms: list[str]) -> bool:
    if not location_terms:
        return False
    blob, values = _text_blob(item)
    return any(term in values or term in blob for term in location_terms)


def _identity_fields_for_role(role: str) -> tuple[str, ...]:
    if role in {"dental_clinic", "sports_training", "travel_agency"}:
        return SERVICE_IDENTITY_TEXT_FIELDS
    return IDENTITY_TEXT_FIELDS


def _text_blob(item: dict[str, Any]) -> tuple[str, set[str]]:
    return _text_blob_for_fields(item, TEXT_FIELDS)


def _text_blob_for_fields(item: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, set[str]]:
    item_key = str(
        item.get("poi_id")
        or item.get("id")
        or item.get("amap_id")
        or item.get("merchant_id")
        or item.get("name")
        or id(item)
    )
    field_signature = "|".join(str(item.get(field))[:160] for field in fields)
    cache_key = (f"{item_key}:{hash(field_signature)}", fields)
    cached = _TEXT_BLOB_CACHE.get(cache_key)
    if cached:
        return cached

    values: list[Any] = []
    for field in fields:
        if field == "_gaode_raw_business":
            values.extend(_raw_business_values(item))
        else:
            values.extend(flatten_semantic_values(item.get(field)))
    normalized = [
        normalize_semantic_text(str(value))
        for value in values
        if len(normalize_semantic_text(str(value))) >= 2
    ]
    result = ("\n".join(normalized), set(normalized))
    if len(_TEXT_BLOB_CACHE) >= _TEXT_BLOB_CACHE_LIMIT:
        _TEXT_BLOB_CACHE.clear()
    _TEXT_BLOB_CACHE[cache_key] = result
    return result


def _raw_business_values(item: dict[str, Any]) -> list[str]:
    raw = item.get("raw")
    if not isinstance(raw, dict):
        return []
    nested_raw = raw.get("raw") if isinstance(raw.get("raw"), dict) else {}
    candidates = [raw, nested_raw]
    business_blocks = []
    for payload in candidates:
        for key in ("biz_ext", "business"):
            value = payload.get(key)
            if isinstance(value, dict):
                business_blocks.append(value)

    values: list[Any] = []
    for payload in candidates:
        for key in ("name", "type", "address", "adname", "cityname", "pname"):
            values.append(payload.get(key))
    for block in business_blocks:
        for key in (
            "tag",
            "rectag",
            "keytag",
            "alias",
            "business_area",
            "opentime_today",
            "opentime_week",
        ):
            values.append(block.get(key))
    return [
        str(value).strip()
        for value in flatten_semantic_values(values)
        if str(value or "").strip()
    ]


def _matches_any_term(blob: str, values: set[str], terms: list[str]) -> bool:
    return any(term in values or term in blob for term in terms)


PARK_SCENIC_POSITIVE_TERMS = (
    "公园",
    "游园",
    "滨江",
    "江边",
    "河边",
    "夜景",
    "步道",
    "观景",
    "外滩",
    "风景名胜",
    "公园广场",
    "休闲场所",
)
PARK_SCENIC_FALSE_POSITIVE_TERMS = (
    "商务大厦",
    "写字楼",
    "办公楼",
    "公寓",
    "商场",
    "购物中心",
    "店)",
    "店）",
    "绿地缤纷",
    "绿地汇",
    "绿地商务",
    "绿地科创",
)


def _matches_park_scenic_identity(blob: str, values: set[str]) -> bool:
    if any(term in values or term in blob for term in PARK_SCENIC_POSITIVE_TERMS):
        return True
    if "绿地" in values or "绿地" in blob:
        return not any(term in blob for term in PARK_SCENIC_FALSE_POSITIVE_TERMS)
    return False


BROAD_NEGATIVE_AUXILIARY_TERMS = {
    "meat",
    "social",
    "lively",
    "atmosphere",
    "high_calorie",
    "restaurant",
}


def _negative_group_terms(group: str) -> list[str]:
    payload = B_SEMANTIC_GROUPS.get(group) or {}
    raw_terms = list(payload.get("primary") or [])
    raw_terms.extend(
        term
        for term in (payload.get("auxiliary") or [])
        if normalize_semantic_text(term) not in BROAD_NEGATIVE_AUXILIARY_TERMS
    )
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = normalize_semantic_text(term)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _item_has_negative_group_evidence(item: dict[str, Any], group: str) -> bool:
    terms = _negative_group_terms(group)
    if not terms:
        return False
    blob, values = _text_blob_for_fields(item, IDENTITY_TEXT_FIELDS)
    return _matches_any_term(blob, values, terms)


def _text_has_negated_restaurant_group(text: str, group: str) -> bool:
    terms = _negative_group_terms(group)
    if not terms:
        return False
    negation_prefixes = ("不要", "别", "别推荐", "不要推荐", "不推荐", "不吃", "避开")
    for term in terms:
        if any(f"{prefix}{term}" in text for prefix in negation_prefixes):
            return True
    return False


def _positive_requested_restaurant_groups(
    state: PlanState,
    constraints: dict[str, Any],
    raw_text: str,
) -> set[str]:
    values: list[Any] = []
    planning_preferences = constraints.get("planning_preferences") or {}
    if isinstance(planning_preferences, dict):
        values.extend(flatten_semantic_values(planning_preferences.get("food_type")))
        values.extend(flatten_semantic_values(planning_preferences.get("restaurant_type")))
    user_profile = state.get("user_profile", {}) or {}
    if isinstance(user_profile, dict):
        values.extend(flatten_semantic_values(user_profile.get("food_preference")))

    groups = semantic_groups_in_values(values).intersection({"烤肉", "火锅", "炸鸡小吃"})
    raw_groups = semantic_groups_in_values([raw_text]).intersection({"烤肉", "火锅", "炸鸡小吃"})
    for group in raw_groups:
        if not _text_has_negated_restaurant_group(raw_text, group):
            groups.add(group)
    return groups


def _forbidden_restaurant_groups_for_retrieval(
    state: PlanState,
    constraints: dict[str, Any],
) -> set[str]:
    values: list[Any] = []
    contract = constraints.get("b_requirement_contract") or {}
    if isinstance(contract, dict):
        values.extend(contract.get("forbidden_restaurant_groups") or [])
    values.extend(constraints.get("avoid") or [])
    user_profile = state.get("user_profile", {}) or {}
    if isinstance(user_profile, dict):
        values.extend(user_profile.get("avoid") or [])
    raw_text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    groups = semantic_groups_in_values(values)
    requested_groups = _positive_requested_restaurant_groups(state, constraints, raw_text)
    if (
        str(constraints.get("mom_diet") or "").lower() == "low_calorie"
        or any(term in raw_text for term in ("减脂", "减肥", "低卡", "轻食", "少油", "健康饮食", "健康餐", "不油腻", "清淡"))
    ):
        groups.update({"烤肉", "火锅", "炸鸡小吃"})
    return groups.intersection({"烤肉", "火锅", "炸鸡小吃"}).difference(requested_groups)


def _explicit_price_ceiling_yuan(constraints: dict[str, Any]) -> float | None:
    text = str(constraints.get("raw_text") or "")
    if not text:
        return None
    patterns = (
        r"(?:人均|每人|单人)?\s*(\d{1,4})\s*(?:元|块|块钱)\s*(?:以内|以下|封顶)",
        r"(?:不超过|控制在|低于|少于)\s*(\d{1,4})\s*(?:元|块|块钱)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = to_float(match.group(1), 0.0)
            if 0 < value <= 5000:
                return value
    return None


def _raw_gaode_cost(item: dict[str, Any]) -> float | None:
    raw = item.get("raw")
    if not isinstance(raw, dict):
        return None
    candidates = [raw]
    if isinstance(raw.get("raw"), dict):
        candidates.append(raw["raw"])
    for payload in candidates:
        for key in ("biz_ext", "business"):
            block = payload.get(key)
            if not isinstance(block, dict):
                continue
            cost = to_float(block.get("cost"), 0.0)
            if cost > 0:
                return cost
    return None


def _coordinates(item: dict[str, Any]) -> tuple[float, float] | None:
    coordinates = item.get("coordinates") or item.get("location")
    if isinstance(coordinates, str) and "," in coordinates:
        left, right = coordinates.split(",", 1)
    elif isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        left, right = coordinates[0], coordinates[1]
    else:
        left = item.get("longitude") or item.get("lng") or item.get("lon")
        right = item.get("latitude") or item.get("lat")
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        return None


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    import math

    lng1, lat1 = first
    lng2, lat2 = second
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _origin_coordinates(constraints: dict[str, Any]) -> tuple[float, float] | None:
    raw = constraints.get("origin_coordinates") or constraints.get("user_coordinates")
    if isinstance(raw, str) and "," in raw:
        left, right = raw.split(",", 1)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        left, right = raw[0], raw[1]
    else:
        return None
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        return None


def _candidate_distance_km(item: dict[str, Any], constraints: dict[str, Any]) -> float:
    origin = _origin_coordinates(constraints)
    coord = _coordinates(item)
    if origin and coord:
        return round(_haversine_km(origin, coord) * 1.35, 2)
    return round(to_float(item.get("distance_km") or item.get("distance"), 0.0), 2)


def _quality_fallback_pool(
    items: list[dict[str, Any]],
    *,
    constraints: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    def rank_key(item: dict[str, Any]) -> tuple[float, float]:
        distance = _candidate_distance_km(item, constraints)
        if distance <= 0:
            distance = 999.0
        return (distance, -to_float(item.get("rating") or item.get("score"), 4.0))

    return sorted(items, key=rank_key)[:limit]


SPARSE_LOCAL_SERVICE_ROLES = {
    "dental_clinic",
    "sports_training",
    "travel_agency",
}


def _sparse_role_identity_fallback_scores(
    pool: list[dict[str, Any]],
    *,
    role: str,
    constraints: dict[str, Any],
    required_identity_terms: list[str],
    excluded_identity_terms: list[str],
    identity_fields: tuple[str, ...],
) -> list[tuple[float, dict[str, Any], list[str], float]]:
    """Keep sparse local-service roles from disappearing after strict scoring.

    This fallback still requires the POI identity fields to match the target role,
    so address-only noise such as "next to a dental clinic" does not become a
    fake executable service.
    """

    if role not in SPARSE_LOCAL_SERVICE_ROLES:
        return []

    results: list[tuple[float, dict[str, Any], list[str], float]] = []
    for item in pool:
        identity_blob, identity_values = _text_blob_for_fields(item, identity_fields)
        full_blob, full_values = _text_blob(item)
        if required_identity_terms and not (
            _matches_any_term(
                identity_blob,
                identity_values,
                required_identity_terms,
            )
            or _matches_any_term(
                full_blob,
                full_values,
                required_identity_terms,
            )
        ):
            continue
        if excluded_identity_terms and (
            _matches_any_term(
                identity_blob,
                identity_values,
                excluded_identity_terms,
            )
            or _matches_any_term(
                full_blob,
                full_values,
                excluded_identity_terms,
            )
        ):
            continue
        if role == "dental_clinic" and not _matches_any_term(
            identity_blob,
            identity_values,
            required_identity_terms,
        ):
            continue
        distance_km = _candidate_distance_km(item, constraints)
        score = 3.0
        score += min(1.0, max(0.0, to_float(item.get("rating") or item.get("score"), 4.0) - 3.5))
        if distance_km:
            score -= min(2.0, distance_km * 0.06)
        results.append(
            (
                score,
                item,
                [
                    f"sparse local-service identity fallback: {role}",
                    "identity fields match requested service role",
                ],
                distance_km,
            )
        )
    results.sort(key=lambda row: row[0], reverse=True)
    return results


LOCATION_POOL_ROLE_NOISE = {
    "上海",
    "附近",
    "短距离",
    "餐厅",
    "吃饭",
    "正餐",
    "restaurant",
    "activity",
    "shopping",
    "retail",
    "hotel",
}


def _has_role_term_match(
    item: dict[str, Any],
    *,
    node_terms: list[str],
    location_terms: list[str],
) -> bool:
    role_terms = [
        term
        for term in node_terms
        if term not in location_terms
        and term not in LOCATION_ANCHOR_TERMS
        and term not in LOCATION_POOL_ROLE_NOISE
        and not term.startswith("restaurant")
    ]
    if not role_terms:
        return True
    blob, values = _text_blob(item)
    return any(term in values or term in blob for term in role_terms)


def _fast_role_prefilter_pool(
    *,
    bundle: dict[str, Any],
    domain: str,
    role: str,
    node_terms: list[str],
    global_terms: list[str],
    constraints: dict[str, Any],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cheap role-first retrieval for activity subdomains with strong identity terms."""

    normalized_domain = _normalize_domain(domain)
    if role not in FAST_DOMAIN_ROLE_PREFILTER_ROLES:
        return [], {}

    item_source = "domain_items"
    domain_cache = bundle.get("domain_items") if isinstance(bundle.get("domain_items"), dict) else {}
    items = list(domain_cache.get(normalized_domain) or [])
    memory_index: dict[str, Any] | None = None
    domain_size = len(items)
    if items:
        item_source = "domain_items"

    required_terms = _literal_query_tokens(list(ROLE_REQUIRED_IDENTITY_TERMS.get(role, ())))
    excluded_terms = _literal_query_tokens(list(ROLE_EXCLUDED_IDENTITY_TERMS.get(role, ())))
    if role == "restaurant_specific":
        explicit_groups = semantic_groups_in_values(node_terms).intersection(
            B_RESTAURANT_INTENT_GROUPS
        )
        if not explicit_groups:
            return [], {
                "retriever": "local_poi_fast_role_prefilter_v1",
                "candidate_pool_size": 0,
                "domain_size": domain_size,
                "item_source": item_source,
                "skipped": "restaurant_specific_without_explicit_cuisine",
            }
        required_terms = _literal_query_tokens(
            semantic_terms_for_groups(explicit_groups, include_auxiliary=True)
        )
    role_terms = [
        term
        for term in (required_terms or node_terms)
        if term not in LOCATION_ANCHOR_TERMS
        and term not in LOCATION_POOL_ROLE_NOISE
        and not term.startswith("restaurant")
    ]
    location_terms = [term for term in global_terms if term in LOCATION_ANCHOR_TERMS]
    if not role_terms:
        return [], {
            "retriever": "local_poi_fast_role_prefilter_v1",
            "candidate_pool_size": 0,
            "domain_size": domain_size,
            "item_source": item_source,
            "skipped": "no_role_terms",
        }

    memory_meta: dict[str, Any] = {}
    role_cache_path: Path | None = None
    role_cache_signature = ""
    if not items:
        root = Path(bundle.get("root") or _mock_data_dir())
        domain_signature = _domain_signature(root, normalized_domain)
        role_cache_signature = "|".join(
            [
                FAST_ROLE_PREFILTER_CACHE_VERSION,
                domain_signature,
                normalized_domain,
                role,
                ",".join(role_terms),
                ",".join(excluded_terms),
            ]
        )
        role_cache_path = _index_cache_dir() / _fast_role_cache_filename(
            root,
            domain=normalized_domain,
            role=role,
            signature=role_cache_signature,
        )
        cached_items = _load_fast_role_pool_cache(
            role_cache_path,
            expected_signature=role_cache_signature,
        )
        if cached_items is not None:
            items = cached_items
            domain_size = len(items)
            item_source = "fast_role_pool_cache"

    if not items:
        memory_index = _memory_index_for_domain(bundle, domain=normalized_domain)
        domain_size = int(memory_index.get("doc_count") or 0)
        retrieval_limit = min(
            domain_size,
            max(limit * 2, 600),
        )
        items, memory_meta = retrieve_poi_memory_candidates(
            memory_index,
            node_terms=_dedupe_text([role_terms, node_terms], limit=48),
            global_terms=location_terms,
            limit=retrieval_limit,
        )
        item_source = "memory_index_candidate_retrieval"
        if not items:
            return [], {
                "retriever": "local_poi_fast_role_prefilter_v1",
                "candidate_pool_size": 0,
                "domain_size": domain_size,
                "item_source": item_source,
                "memory_retrieval_meta": memory_meta,
                "skipped": "memory_retrieval_empty",
            }
        if role_cache_path is not None and role_cache_signature:
            _save_fast_role_pool_cache(
                role_cache_path,
                signature=role_cache_signature,
                items=items,
            )

    scored: list[tuple[float, dict[str, Any]]] = []
    location_scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        identity_blob, identity_values = _text_blob_for_fields(item, _identity_fields_for_role(role))
        matched_terms = [
            term for term in role_terms if term in identity_values or term in identity_blob
        ]
        if not matched_terms:
            continue
        if excluded_terms and _matches_any_term(identity_blob, identity_values, excluded_terms):
            continue
        distance_km = _candidate_distance_km(item, constraints)
        rating = to_float(item.get("rating") or item.get("score"), 4.0)
        score = len(matched_terms) * 10.0 + min(3.0, max(0.0, rating - 3.5) * 1.5)
        if distance_km:
            score -= min(4.0, distance_km * 0.1)
        row = (score, item)
        scored.append(row)
        if location_terms and _item_matches_location_terms(item, location_terms):
            location_scored.append((score + 6.0, item))

    active_scored = location_scored if len(location_scored) >= min(3, limit) else scored
    active_scored.sort(key=lambda row: row[0], reverse=True)
    pool = [item for _, item in active_scored[:limit]]
    return pool, {
        "retriever": "local_poi_fast_role_prefilter_v1",
        "query_terms": node_terms[:32],
        "matched_terms": role_terms[:32],
        "candidate_pool_size": len(pool),
        "raw_role_match_count": len(scored),
        "item_source": item_source,
        "location_anchor_terms": location_terms,
        "location_anchor_pool_size": len(location_scored),
        "domain_size": domain_size,
        "location_anchor_pool_skipped": (
            ""
            if not location_terms or active_scored is location_scored
            else "role_terms_too_sparse"
        ),
    }


def _merge_auxiliary_fields(item: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    merchant_id = str(item.get("merchant_id") or "").strip()
    enriched = dict(item)
    if not _rag_auxiliary_merge_enabled(bundle):
        enriched["_rag_auxiliary_merge_skipped"] = "large_auxiliary_files"
        raw_cost = _raw_gaode_cost(enriched)
        if raw_cost is not None:
            enriched["gaode_avg_cost"] = raw_cost
            current_price = to_float(enriched.get("price"), 0.0)
            if current_price <= 0 or raw_cost < current_price:
                enriched["price"] = raw_cost
                enriched["avg_price_per_person"] = raw_cost
                field_sources = dict(enriched.get("field_sources") or {})
                field_sources["price"] = "observed_gaode_raw_cost"
                enriched["field_sources"] = field_sources
        return enriched
    availability = _bundle_availability(bundle).get(poi_id)
    if isinstance(availability, dict):
        for key in (
            "available",
            "available_slots",
            "queue_time_min",
            "inventory_left",
            "reservation_required",
            "holiday_status",
        ):
            if key in availability and enriched.get(key) in (None, "", []):
                enriched[key] = availability[key]
    products = _bundle_products_by_poi(bundle).get(poi_id, [])[:3]
    deals = _bundle_deals_by_poi(bundle).get(poi_id, [])[:3]
    merchant = _bundle_merchants(bundle).get(merchant_id, {})
    if products:
        enriched["product_options"] = products
    if deals:
        enriched["deal_options"] = deals
        enriched["package_options"] = _dedupe_text(
            [
                deal.get("title")
                for deal in deals
            ]
            + [
                component
                for deal in deals
                for component in flatten_semantic_values(deal.get("package_components"))
            ],
            limit=12,
        )
    for key in ("trust_score", "review_count", "operation_stability_score", "business_capabilities"):
        if merchant.get(key) is not None and enriched.get(key) in (None, "", []):
            enriched[key] = merchant.get(key)
    raw_cost = _raw_gaode_cost(enriched)
    if raw_cost is not None:
        enriched["gaode_avg_cost"] = raw_cost
        current_price = to_float(enriched.get("price"), 0.0)
        if current_price <= 0 or raw_cost < current_price:
            enriched["price"] = raw_cost
            enriched["avg_price_per_person"] = raw_cost
            field_sources = dict(enriched.get("field_sources") or {})
            field_sources["price"] = "observed_gaode_raw_cost"
            enriched["field_sources"] = field_sources
    return enriched


def _normalize_domain(domain: str) -> str:
    if domain in {"lodging", "hotel"}:
        return "hotel"
    return str(domain or "")


def _generic_poi_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if "_generic_poi_records" not in bundle:
        root = Path(bundle.get("root") or _mock_data_dir())
        records = _read_records(root, "deduped_pois")
        if not records:
            records = _read_records(root, "raw_pois")
        bundle["_generic_poi_records"] = _dedupe_records(records)
    return bundle.get("_generic_poi_records", [])


def _generic_poi_text(item: dict[str, Any]) -> str:
    values: list[Any] = []
    for key in (
        "name",
        "type",
        "category",
        "address",
        "gaode_type",
        "primary_category",
        "primary_keyword",
        "business_area",
    ):
        values.extend(flatten_semantic_values(item.get(key)))
    for key in ("business", "biz_ext", "raw"):
        nested = item.get(key)
        if isinstance(nested, dict):
            for nested_key in ("tag", "rectag", "keytag", "business_area", "type", "address"):
                values.extend(flatten_semantic_values(nested.get(nested_key)))
    return " ".join(str(value) for value in values if str(value).strip())


def _generic_poi_matches_domain(item: dict[str, Any], domain: str) -> bool:
    text = _generic_poi_text(item)
    hints = DOMAIN_FALLBACK_HINTS.get(domain, ())
    if not hints:
        return False
    if domain == "hotel":
        type_text = str(item.get("type") or item.get("gaode_type") or "")
        raw_type = ""
        raw = item.get("raw")
        if isinstance(raw, dict):
            raw_type = str(raw.get("type") or "")
        type_blob = f"{type_text} {raw_type}"
        if not any(
            term in type_blob
            for term in ("宾馆酒店", "酒店", "民宿", "旅馆", "客栈", "公寓", "度假村")
        ):
            return False
    if domain == "transport_service":
        type_text = str(item.get("type") or item.get("gaode_type") or "")
        raw_type = ""
        raw = item.get("raw")
        if isinstance(raw, dict):
            raw_type = str(raw.get("type") or "")
        name_or_category = " ".join(
            str(item.get(key) or "")
            for key in ("name", "category", "primary_category", "gaode_keyword")
        )
        type_blob = f"{type_text} {raw_type}"
        return "停车场" in type_blob
    return any(hint in text for hint in hints)


def _parse_location(value: Any) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    if "," not in text:
        return None, None
    left, right = text.split(",", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None, None


def _normalize_generic_poi(item: dict[str, Any], domain: str) -> dict[str, Any]:
    defaults = DOMAIN_DEFAULTS.get(domain, {})
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    amap_id = str(item.get("amap_id") or item.get("id") or "").strip()
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    business = item.get("business") if isinstance(item.get("business"), dict) else {}
    biz_ext = item.get("biz_ext") if isinstance(item.get("biz_ext"), dict) else {}
    location = item.get("coordinates") or item.get("location") or raw.get("location")
    lng, lat = _parse_location(location)
    gaode_type = item.get("gaode_type") or item.get("type") or raw.get("type")
    category = (
        item.get("category")
        or item.get("primary_category")
        or business.get("keytag")
        or business.get("rectag")
        or biz_ext.get("keytag")
        or biz_ext.get("rectag")
        or gaode_type
        or domain
    )
    rating = item.get("rating") or biz_ext.get("rating") or business.get("rating")
    cost = item.get("price") or item.get("cost") or biz_ext.get("cost") or business.get("cost")
    normalized = {
        "poi_id": poi_id or (f"gaode_poi_{amap_id}" if amap_id else ""),
        "amap_id": amap_id or None,
        "merchant_id": item.get("merchant_id") or (f"m_gaode_poi_{amap_id}" if amap_id else None),
        "name": item.get("name") or raw.get("name"),
        "type": defaults.get("type") or domain,
        "supply_domain": domain,
        "category": category,
        "primary_category": category,
        "gaode_type": gaode_type,
        "address": item.get("address") or raw.get("address"),
        "location": location,
        "coordinates": location,
        "longitude": lng,
        "latitude": lat,
        "business_area": item.get("business_area") or business.get("business_area") or biz_ext.get("business_area"),
        "tags": [domain, category, gaode_type],
        "price": to_float(cost, defaults.get("price", 80.0)),
        "duration_min": int(defaults.get("duration_min", 30)),
        "queue_time_min": int(defaults.get("queue_time_min", 5)),
        "rating": to_float(rating, 4.2),
        "available": True,
        "source": item.get("source") or "gaode_generic_poi_fallback",
        "source_channel": item.get("source_channel") or "gaode_deduped_pois",
        "source_evidence": [
            f"Gaode generic POI fallback domain={domain}",
            f"amap_id={amap_id}" if amap_id else "",
        ],
        "raw": item,
    }
    return {key: value for key, value in normalized.items() if value not in (None, "", [])}


def _generic_domain_items(bundle: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    items = [
        _normalize_generic_poi(item, domain)
        for item in _generic_poi_records(bundle)
        if _generic_poi_matches_domain(item, domain)
    ]
    if domain == "transport_service" and not items:
        items.extend(_derived_parking_proxy_items(bundle))
    return items


def _parking_proxy_source_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if "_parking_proxy_source_records" not in bundle:
        root = Path(bundle.get("root") or _mock_data_dir())
        records = _read_records(root, "activities") + _read_records(root, "restaurants")
        bundle["_parking_proxy_source_records"] = _dedupe_records(records)
    return bundle.get("_parking_proxy_source_records", [])


def _item_has_parking_signal(item: dict[str, Any]) -> bool:
    if item.get("parking_available") is True:
        return True
    values = [
        item.get("parking_fee_policy"),
        item.get("decision_profile"),
        item.get("fulfillment_actions"),
        item.get("source_evidence"),
        item.get("location"),
        item.get("address"),
    ]
    blob = " ".join(str(value) for value in flatten_semantic_values(values))
    return "停车" in blob or "🅿" in blob


def _normalize_parking_proxy(item: dict[str, Any]) -> dict[str, Any]:
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    amap_id = str(item.get("amap_id") or item.get("id") or "").strip()
    location = item.get("coordinates") or item.get("location")
    lng, lat = _parse_location(location)
    policy = str(item.get("parking_fee_policy") or "附近可停车，需到店前再次确认").strip()
    source_name = str(item.get("name") or "目标地点").strip()
    source_key = poi_id or amap_id or hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    return {
        "poi_id": f"parking_proxy_{source_key}",
        "amap_id": amap_id or None,
        "merchant_id": f"m_parking_proxy_{source_key}",
        "name": f"{source_name} 停车指引",
        "type": "transport_service",
        "supply_domain": "transport_service",
        "category": "停车指引",
        "primary_category": "停车指引",
        "gaode_type": "派生停车服务;停车指引",
        "address": item.get("address") or item.get("location"),
        "location": location,
        "coordinates": location,
        "longitude": lng,
        "latitude": lat,
        "business_area": item.get("business_area"),
        "tags": ["停车", "停车场", "停车指引", policy],
        "price": 25.0,
        "duration_min": 15,
        "queue_time_min": 5,
        "rating": to_float(item.get("rating"), 4.0),
        "available": True,
        "parking_proxy": True,
        "parking_proxy_source_poi_id": poi_id or None,
        "parking_fee_policy": policy,
        "source": "b_derived_parking_proxy",
        "source_channel": "local_supply_parking_signal",
        "source_evidence": [
            "Derived parking proxy from local supply parking signal",
            f"source_poi_id={poi_id}" if poi_id else "",
            policy,
        ],
        "raw": item,
    }


def _derived_parking_proxy_items(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _normalize_parking_proxy(item)
        for item in _parking_proxy_source_records(bundle)
        if _item_has_parking_signal(item)
    ]


def _domain_items(bundle: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    domain = _normalize_domain(domain)
    domain_cache = bundle.setdefault("domain_items", {})
    if domain not in domain_cache:
        root = Path(bundle.get("root") or _mock_data_dir())
        items: list[dict[str, Any]] = []
        for stem in DOMAIN_STEMS.get(domain, (domain,)):
            items.extend(_read_records(root, stem))
        if not items and domain in GENERIC_POI_FALLBACK_DOMAINS:
            items.extend(_generic_domain_items(bundle, domain))
        domain_cache[domain] = _dedupe_records(items)
    items = domain_cache.get(domain, [])
    return items[: _max_scan_per_domain()]


def _memory_index_for_domain(
    bundle: dict[str, Any],
    *,
    domain: str,
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_domain = _normalize_domain(domain)
    cache = bundle.setdefault("_memory_indexes", {})
    cache_meta = bundle.setdefault("_memory_index_cache_meta", {})
    if normalized_domain in cache:
        cache_meta.setdefault(
            normalized_domain,
            {
                "status": "memory_hit",
                "doc_count": cache[normalized_domain].get("doc_count"),
            },
        )
        return cache[normalized_domain]

    root = Path(bundle.get("root") or _mock_data_dir())
    signature = ""
    cache_path: Path | None = None
    if _index_cache_enabled():
        signature = _domain_signature(root, normalized_domain)
        cache_path = _index_cache_dir() / _cache_filename(root, normalized_domain, signature)
        cached_index = load_poi_memory_index_cache(cache_path, expected_signature=signature)
        if cached_index is not None:
            cache[normalized_domain] = cached_index
            cache_meta[normalized_domain] = {
                "status": "disk_hit",
                "path": str(cache_path),
                "doc_count": cached_index.get("doc_count"),
            }
            return cached_index

    if items is None:
        items = _domain_items(bundle, normalized_domain)
    index = build_poi_memory_index(items)
    cache[normalized_domain] = index
    saved = False
    if cache_path is not None and signature:
        saved = save_poi_memory_index_cache(cache_path, signature=signature, index=index)
    cache_meta[normalized_domain] = {
        "status": "built_saved" if saved else ("built_unsaved" if cache_path else "disabled"),
        "path": str(cache_path) if cache_path else "",
        "doc_count": index.get("doc_count"),
    }
    return index


def _items_from_memory_index(index: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for document in index.get("documents", []) or []:
        item = document.get("item") if isinstance(document, dict) else None
        if isinstance(item, dict):
            items.append(item)
    return items


def _role_terms(intent: dict[str, Any]) -> list[str]:
    role = str(intent.get("role") or "")
    search_terms = flatten_semantic_values(intent.get("search_terms"))
    if role == "restaurant_specific":
        explicit_groups = semantic_groups_in_values(search_terms).intersection(
            B_RESTAURANT_INTENT_GROUPS
        )
        if explicit_groups:
            return _dedupe_text(
                [
                    search_terms,
                    semantic_terms_for_groups(explicit_groups, include_auxiliary=True),
                    intent.get("label"),
                    intent.get("supply_domain"),
                    role,
                ],
                limit=40,
            )
    return _dedupe_text(
        [
            search_terms,
            intent.get("label"),
            intent.get("supply_domain"),
            role,
            ROLE_QUERY_TERMS.get(role, ()),
        ],
        limit=40,
    )


def _node_query_terms(state: PlanState, constraints: dict[str, Any], intent: dict[str, Any]) -> tuple[list[str], list[str]]:
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    role = str(intent.get("role") or "")
    domain = str(intent.get("supply_domain") or "")
    node_terms = _role_terms(intent)
    use_local_meal_context = False
    if domain == "activity":
        if role in {"activity", "family_activity", "family_indoor_play"}:
            node_terms.extend(flatten_semantic_values(planning_preferences.get("activity_type")))
        node_terms.extend(flatten_semantic_values(planning_preferences.get("experience_type")))
    elif domain == "restaurant":
        meal_terms = _meal_context_terms(state, constraints, role)
        if meal_terms:
            use_local_meal_context = True
            node_terms.extend(meal_terms)
        else:
            node_terms.extend(flatten_semantic_values(planning_preferences.get("food_type")))
            node_terms.extend(flatten_semantic_values(planning_preferences.get("restaurant_type")))
    elif role in ROLE_QUERY_TERMS:
        node_terms.extend(ROLE_QUERY_TERMS[role])

    global_terms = [
        constraints.get("city"),
        constraints.get("district"),
        constraints.get("business_area"),
        constraints.get("hard_tags"),
        constraints.get("soft_tags"),
        constraints.get("avoid"),
        state.get("scenario_activities"),
    ]
    if not use_local_meal_context:
        global_terms.extend([state.get("user_input"), constraints.get("raw_text")])
    global_terms.extend(
        _location_anchor_terms(
            [
                state.get("user_input"),
                constraints.get("raw_text"),
                constraints.get("district"),
                constraints.get("business_area"),
            ]
        )
    )
    global_terms.extend(_named_event_location_anchor_terms(state, constraints))
    # Inferred anchors are only soft retrieval hints. Explicit user-provided
    # anchors remain in global_terms and may narrow the candidate pool below.
    node_terms.extend(_inferred_location_anchor_terms(state, constraints))
    return _query_tokens(node_terms), _query_tokens(global_terms)


def _meal_context_terms(state: PlanState, constraints: dict[str, Any], role: str) -> list[str]:
    markers = MEAL_ROLE_MARKERS.get(role)
    if not markers:
        return []
    text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    if not text:
        return []

    marker_positions = [
        (text.find(marker), marker)
        for marker in markers
        if text.find(marker) >= 0
    ]
    if not marker_positions:
        return []
    marker_index, marker = min(marker_positions, key=lambda item: item[0])
    segment_start = marker_index + len(marker)
    segment_end = min(len(text), segment_start + 36)
    for stop_word in MEAL_CONTEXT_STOP_WORDS:
        stop_index = text.find(stop_word, segment_start)
        if stop_index > segment_start and stop_index < segment_end:
            segment_end = stop_index
    segment = text[marker_index:segment_end]

    terms: list[Any] = [marker]
    for keyword, expansions in MEAL_CONTEXT_TERM_EXPANSIONS.items():
        if keyword in segment:
            terms.extend(expansions)
    return _dedupe_text(terms, limit=24)


def _score_item(
    item: dict[str, Any],
    *,
    node_terms: list[str],
    global_terms: list[str],
    constraints: dict[str, Any],
    identity_fields: tuple[str, ...] = IDENTITY_TEXT_FIELDS,
) -> tuple[float, list[str], float, int, int]:
    blob, values = _text_blob(item)
    identity_blob, identity_values = _text_blob_for_fields(item, identity_fields)
    score = 0.0
    evidence: list[str] = []
    matched = 0
    identity_matched = 0
    for term in node_terms:
        if term in LOCATION_ANCHOR_TERMS:
            if term in values:
                score += 1.0
                evidence.append(f"软空间提示匹配: {term}")
            elif term in blob:
                score += 0.4
                evidence.append(f"软空间提示匹配: {term}")
            continue
        if term in values:
            score += 9.0
            matched += 1
            evidence.append(f"字段精确匹配: {term}")
        elif term in blob:
            score += 5.0
            matched += 1
            evidence.append(f"文本匹配: {term}")
        if term in identity_values or term in identity_blob:
            identity_matched += 1
    for term in global_terms:
        if term in values:
            score += 2.0
        elif term in blob:
            score += 0.8
        if term in LOCATION_ANCHOR_TERMS and (term in identity_values or term in identity_blob):
            score += 5.0
            evidence.append(f"空间锚点匹配: {term}")

    rating = to_float(item.get("rating") or item.get("score"), 0.0)
    trust = to_float(item.get("trust_score"), 0.0)
    review_count = to_float(item.get("review_count") or item.get("verified_reviews"), 0.0)
    distance_km = _candidate_distance_km(item, constraints)
    score += min(2.5, max(0.0, rating - 3.5) * 1.2)
    score += min(1.5, trust * 1.2)
    score += min(1.0, review_count / 500)
    if distance_km:
        score -= min(4.0, distance_km * 0.12)
    if item.get("available") is False:
        score -= 5.0
        evidence.append("可用性风险: 当前标记不可用")
    if item.get("available_slots"):
        score += 0.7
        evidence.append("存在可预约/可用时段")
    if item.get("deal_options") or item.get("package_options"):
        score += 0.3
        evidence.append("包含套餐/优惠券线索")
    price_ceiling = _explicit_price_ceiling_yuan(constraints)
    price = to_float(item.get("avg_price_per_person") or item.get("price"), 0.0)
    if price_ceiling and price > 0:
        if price <= price_ceiling:
            score += 4.0
            evidence.append(f"价格满足上限: {price:.0f}元")
        elif price <= price_ceiling * 1.5:
            score += 1.5
            evidence.append(f"价格接近上限: {price:.0f}元")
        else:
            score -= min(8.0, (price - price_ceiling) / max(price_ceiling, 1.0))
    elif str(constraints.get("scene") or "") == "low_budget" and 0 < price <= 30:
        score += 2.0
        evidence.append(f"低预算价格友好: {price:.0f}元")
    if matched == 0:
        score -= 8.0
    return score, _dedupe_text(evidence, limit=6), distance_km, matched, identity_matched


def _field_sources(item: dict[str, Any]) -> dict[str, str]:
    existing = item.get("field_sources")
    if isinstance(existing, dict):
        return dict(existing)
    source = str(item.get("source") or item.get("source_channel") or "")
    observed = "observed_gaode" if "gaode" in source else "local_mock_or_observed"
    return {
        "identity": observed,
        "coordinates": observed if item.get("coordinates") or item.get("location") else "missing",
        "rating": "observed_gaode_or_mock",
        "price": "observed_gaode_or_mock",
        "business_hours": "observed_gaode_or_mock",
    }


RAG_CANDIDATE_PAYLOAD_FIELDS = (
    "type",
    "tags",
    "tag_groups",
    "price",
    "rating",
    "score",
    "queue_time_min",
    "available",
    "available_slots",
    "inventory_left",
    "capacity_limit",
    "reservation_required",
    "business_hours",
    "holiday_status",
    "refund_policy",
    "cancel_policy",
    "hidden_cost_risk",
    "commercial_features",
    "deal_ids",
    "merchant_id",
    "source_channel",
    "source_evidence",
    "gaode_keyword",
    "gaode_type",
    "adcode",
    "citycode",
    "tel",
    "category",
    "sub_category",
    "experience_type",
    "restaurant_category",
    "avg_price_per_person",
    "category_price_band",
    "health_tags",
    "menu_health_options",
    "wellness_tags",
    "local_flavor_tags",
    "atmosphere_tags",
    "emotion_tags",
    "weather_sensitivity",
    "indoor_backup",
    "trust_score",
    "verified_reviews",
    "review_count",
    "review_breakdown",
    "review_keywords",
    "ugc_summary",
    "package_options",
    "promotion_highlights",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "baby_chair_available",
    "parking_available",
    "parking_proxy",
    "parking_fee_policy",
    "private_room_available",
    "noise_level",
    "spice_level",
    "booking_policy",
    "dietary_options",
    "service_facilities",
    "physical_intensity",
    "weather_plan",
    "decision_profile",
    "fulfillment_actions",
    "substitution_strategy",
    "peak_risk_profile",
    "suitable_age",
    "age_range",
)


def _candidate_payload(
    item: dict[str, Any],
    *,
    intent: dict[str, Any],
    score: float,
    evidence: list[str],
    distance_km: float,
) -> dict[str, Any]:
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    payload = {
        key: item.get(key)
        for key in RAG_CANDIDATE_PAYLOAD_FIELDS
        if item.get(key) not in (None, "")
    }
    payload.update(
        {
            "poi_id": poi_id,
            "amap_id": item.get("amap_id") or item.get("id"),
            "name": item.get("name"),
            "supply_domain": intent.get("supply_domain") or item.get("supply_domain") or item.get("type"),
            "role": intent.get("role"),
            "node_id": intent.get("node_id"),
            "city": item.get("city") or "上海",
            "address": item.get("address") or item.get("location") or raw.get("address"),
            "location": item.get("address") or item.get("location") or raw.get("address"),
            "coordinates": item.get("coordinates") or raw.get("location") or item.get("location"),
            "latitude": item.get("latitude") or item.get("lat"),
            "longitude": item.get("longitude") or item.get("lng") or item.get("lon"),
            "retrieval_score": round(score, 4),
            "memory_retrieval_score": item.get("_memory_score"),
            "memory_rank": item.get("_memory_rank"),
            "memory_matched_terms": item.get("_memory_matched_terms"),
            "distance_km": distance_km,
            "source": "local_poi_rag",
            "evidence_text": "；".join(evidence[:4]) if evidence else "本地 POI RAG 候选，证据较弱",
            "evidence_fields": {
                "matched_evidence": evidence,
                "rating": item.get("rating"),
                "price": item.get("price"),
                "business_hours": item.get("business_hours"),
                "queue_time_min": item.get("queue_time_min"),
            },
            "field_sources": _field_sources(item),
        }
    )
    return payload


def _retrieve_for_node(
    *,
    state: PlanState,
    constraints: dict[str, Any],
    intent: dict[str, Any],
    bundle: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    domain = str(intent.get("supply_domain") or "")
    normalized_domain = _normalize_domain(domain)
    role = str(intent.get("role") or "")
    search_queries = _dedupe_text(
        [
            f"{constraints.get('city') or '上海'} {' '.join(_role_terms(intent)[:4])}",
            constraints.get("raw_text"),
        ],
        limit=5,
    )
    node_terms, global_terms = _node_query_terms(state, constraints, intent)
    location_terms = [term for term in global_terms if term in LOCATION_ANCHOR_TERMS]
    prefilter_limit = max(top_k * 30, 240)
    fast_pool, fast_meta = _fast_role_prefilter_pool(
        bundle=bundle,
        domain=normalized_domain,
        role=role,
        node_terms=node_terms,
        global_terms=global_terms,
        constraints=constraints,
        limit=prefilter_limit,
    )
    memory_index: dict[str, Any] | None = None
    used_fast_prefilter = bool(fast_pool)
    if used_fast_prefilter:
        domain_size = int(fast_meta.get("domain_size") or len(fast_pool))
        candidate_pool = fast_pool
        memory_pool: list[dict[str, Any]] = []
        retrieval_meta = fast_meta
        retrieval_meta["fallback_full_scan"] = "fast_role_prefilter"
        retrieval_meta["negative_groups"] = sorted(
            _forbidden_restaurant_groups_for_retrieval(state, constraints)
            if normalized_domain == "restaurant"
            else []
        )
        all_items: list[dict[str, Any]] | None = None
    else:
        memory_index = _memory_index_for_domain(bundle, domain=normalized_domain)
        domain_size = int(memory_index.get("doc_count") or 0)
        if domain_size <= 0:
            return {
                "node_id": intent.get("node_id"),
                "role": intent.get("role"),
                "label": intent.get("label"),
                "supply_domain": domain,
                "coverage_status": "missing_domain",
                "search_queries": search_queries,
                "candidates": [],
                "missing_reason": f"本地 POI 索引暂未加载 {domain} 供给",
            }
        pool_limit = min(domain_size, max(top_k * 100, 600))
        memory_pool, retrieval_meta = retrieve_poi_memory_candidates(
            memory_index,
            node_terms=node_terms,
            global_terms=location_terms,
            limit=pool_limit,
        )
        retrieval_meta["location_anchor_terms"] = location_terms
        retrieval_meta["domain_size"] = domain_size
        retrieval_meta["index_cache"] = (
            bundle.get("_memory_index_cache_meta", {}).get(normalized_domain, {})
        )
        retrieval_meta["negative_groups"] = sorted(
            _forbidden_restaurant_groups_for_retrieval(state, constraints)
            if normalized_domain == "restaurant"
            else []
        )
        all_items = None
        if memory_pool:
            candidate_pool = memory_pool
            retrieval_meta["fallback_full_scan"] = False
        elif role in STRICT_ROLE_MATCH_ROLES:
            all_items = _items_from_memory_index(memory_index)
            candidate_pool = all_items
            retrieval_meta["fallback_full_scan"] = True
        else:
            all_items = _items_from_memory_index(memory_index)
            candidate_pool = _quality_fallback_pool(all_items, constraints=constraints, limit=pool_limit)
            retrieval_meta["fallback_full_scan"] = "quality_pool"
    if location_terms:
        location_pool = [
            item
            for item in candidate_pool
            if _item_matches_location_terms(item, location_terms)
        ]
        role_matched_location_count = sum(
            1
            for item in location_pool
            if _has_role_term_match(
                item,
                node_terms=node_terms,
                location_terms=location_terms,
            )
        )
        if len(location_pool) < min(top_k, 3):
            if all_items is None:
                all_items = (
                    _domain_items(bundle, normalized_domain)
                    if used_fast_prefilter
                    else _items_from_memory_index(memory_index or {})
                )
            full_location_pool = [
                item
                for item in all_items
                if _item_matches_location_terms(item, location_terms)
            ]
            if len(full_location_pool) > len(location_pool):
                location_pool = full_location_pool
                role_matched_location_count = sum(
                    1
                    for item in location_pool
                    if _has_role_term_match(
                        item,
                        node_terms=node_terms,
                        location_terms=location_terms,
                    )
                )
        if (
            len(location_pool) >= min(top_k, 3)
            and role_matched_location_count >= min(top_k, 3)
        ):
            candidate_pool = location_pool
            retrieval_meta["location_anchor_pool_size"] = len(location_pool)
        else:
            retrieval_meta["location_anchor_pool_size"] = len(location_pool)
            retrieval_meta["location_anchor_pool_skipped"] = (
                "role_terms_too_sparse"
                if len(location_pool) >= min(top_k, 3)
                else "too_small"
            )
            retrieval_meta["location_anchor_role_matched_count"] = role_matched_location_count
    required_identity_terms = _literal_query_tokens(list(ROLE_REQUIRED_IDENTITY_TERMS.get(role, ())))
    excluded_identity_terms = _literal_query_tokens(list(ROLE_EXCLUDED_IDENTITY_TERMS.get(role, ())))
    forbidden_groups = (
        _forbidden_restaurant_groups_for_retrieval(state, constraints)
        if normalized_domain == "restaurant"
        else set()
    )

    def _score_pool(pool: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any], list[str], float]]:
        pool_scores: list[tuple[float, dict[str, Any], list[str], float]] = []
        for item in pool:
            if forbidden_groups and any(
                _item_has_negative_group_evidence(item, group)
                for group in forbidden_groups
            ):
                continue
            score, evidence, distance_km, matched_count, identity_match_count = _score_item(
                item,
                node_terms=node_terms,
                global_terms=global_terms,
                constraints=constraints,
                identity_fields=_identity_fields_for_role(role),
            )
            if role in STRICT_ROLE_MATCH_ROLES and matched_count <= 0:
                continue
            if role in STRICT_ROLE_MATCH_ROLES and identity_match_count <= 0:
                continue
            if required_identity_terms or excluded_identity_terms:
                identity_blob, identity_values = _text_blob_for_fields(
                    item,
                    _identity_fields_for_role(role),
                )
                if role == "park_scenic_walk" and not _matches_park_scenic_identity(
                    identity_blob,
                    identity_values,
                ):
                    continue
                if role != "park_scenic_walk" and required_identity_terms and not _matches_any_term(
                    identity_blob,
                    identity_values,
                    required_identity_terms,
                ):
                    continue
                if excluded_identity_terms and _matches_any_term(
                    identity_blob,
                    identity_values,
                    excluded_identity_terms,
                ):
                    continue
            if score <= -4.0:
                continue
            pool_scores.append((score, item, evidence, distance_km))
        return pool_scores

    scored = _score_pool(candidate_pool)
    if not scored and role in SPARSE_LOCAL_SERVICE_ROLES:
        scored = _sparse_role_identity_fallback_scores(
            candidate_pool,
            role=role,
            constraints=constraints,
            required_identity_terms=required_identity_terms,
            excluded_identity_terms=excluded_identity_terms,
            identity_fields=_identity_fields_for_role(role),
        )
        if scored:
            retrieval_meta["fallback_full_scan"] = "sparse_local_service_identity_guard"
    if not scored and role in SPARSE_LOCAL_SERVICE_ROLES and used_fast_prefilter and fast_pool:
        scored = _sparse_role_identity_fallback_scores(
            fast_pool,
            role=role,
            constraints=constraints,
            required_identity_terms=required_identity_terms,
            excluded_identity_terms=excluded_identity_terms,
            identity_fields=_identity_fields_for_role(role),
        )
        if scored:
            retrieval_meta["fallback_full_scan"] = "sparse_local_service_identity_guard_relaxed_location"
            retrieval_meta["location_anchor_relaxed_for_sparse_role"] = role
    if not scored and memory_pool:
        if all_items is None:
            all_items = _items_from_memory_index(memory_index or {})
        scored = _score_pool(all_items)
        retrieval_meta["fallback_full_scan"] = "strict_recall_guard"
    elif role in STRICT_ROLE_MATCH_ROLES and memory_pool and len(scored) < min(top_k, 3):
        if all_items is None:
            all_items = _items_from_memory_index(memory_index or {})
        supplemental = _score_pool(all_items)
        seen_ids = {
            str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "")
            for _, item, _, _ in scored
        }
        for row in supplemental:
            _, item, _, _ = row
            item_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "")
            if item_id and item_id in seen_ids:
                continue
            scored.append(row)
            if item_id:
                seen_ids.add(item_id)
        retrieval_meta["fallback_full_scan"] = "strict_sparse_recall_guard"
    scored.sort(key=lambda row: row[0], reverse=True)
    candidates = [
        _candidate_payload(
            _merge_auxiliary_fields(item, bundle),
            intent=intent,
            score=score,
            evidence=evidence,
            distance_km=distance_km,
        )
        for score, item, evidence, distance_km in scored[:top_k]
    ]
    if candidates:
        weakest_score = min(to_float(item.get("retrieval_score"), 0.0) for item in candidates)
        coverage = "weak" if weakest_score < 2.0 else "covered"
    else:
        coverage = "weak"
    return {
        "node_id": intent.get("node_id"),
        "role": intent.get("role"),
        "label": intent.get("label"),
        "supply_domain": domain,
        "coverage_status": coverage,
        "search_queries": search_queries,
        "retrieval_meta": retrieval_meta,
        "candidates": candidates,
        "missing_reason": "" if candidates else "本地 POI RAG 没有找到足够强的候选",
    }


def _should_run(state: PlanState, blueprint: dict[str, Any]) -> bool:
    constraints = state.get("constraints", {}) or {}
    explicit = state.get("b_poi_rag_enabled", constraints.get("b_poi_rag_enabled"))
    if explicit is not None:
        if isinstance(explicit, str):
            return not _falsy(explicit)
        return bool(explicit)

    env_value = os.environ.get("WF_B_RAG_ENABLED")
    if env_value is not None:
        if _falsy(env_value):
            return False
        return _truthy(env_value)

    if state.get("b_replan_request") or constraints.get("b_replan_request"):
        return True
    if state.get("b_rag_candidate_evidence") or state.get("b_rag_node_candidates"):
        return False
    if constraints.get("b_rag_candidate_evidence") or constraints.get("b_rag_node_candidates"):
        return False
    return bool(
        blueprint.get("requires_rag")
        or blueprint.get("template_mode") == "multi_node"
        or blueprint.get("unsupported_roles")
    )


def _allow_llm_requirement_compiler_for_rag(blueprint: dict[str, Any]) -> bool:
    raw_mode = os.environ.get("WF_B_RAG_REQUIREMENT_COMPILER_MODE", "").strip().lower()
    if raw_mode in {"longcat", "llm", "ai", "remote"}:
        return True
    if raw_mode in {"deterministic", "rule", "rules", "off", "0", "false", "no"}:
        return False
    return False


def b_poi_rag_node(state: PlanState) -> dict[str, Any]:
    """Retrieve local POI evidence for B and emit ``b_rag_candidate_evidence``."""

    execution_log = list(state.get("execution_log", []) or [])
    constraints = dict(state.get("constraints", {}) or {})
    initial_blueprint = (
        state.get("b_itinerary_blueprint")
        or constraints.get("b_itinerary_blueprint")
        or build_b_itinerary_blueprint(state, constraints=constraints)
    )

    if not _should_run(state, initial_blueprint):
        return {}

    constraints, requirement_contract, requirement_metadata = apply_b_requirement_contract(
        state,
        constraints=constraints,
        allow_llm=_allow_llm_requirement_compiler_for_rag(initial_blueprint),
    )
    if requirement_metadata and requirement_metadata.get("success"):
        execution_log.append("[B] b_poi_rag_node applied LongCat requirement compiler before retrieval")
    elif requirement_metadata and requirement_metadata.get("skipped"):
        execution_log.append("[B] b_poi_rag_node used deterministic requirement compiler before retrieval")
    elif requirement_contract.get("hard_requirements"):
        execution_log.append("[B] b_poi_rag_node applied deterministic requirement compiler before retrieval")

    blueprint = (
        constraints.get("b_itinerary_blueprint")
        if isinstance(constraints.get("b_itinerary_blueprint"), dict)
        else None
    ) or build_b_itinerary_blueprint(state, constraints=constraints)
    constraints["b_itinerary_blueprint"] = blueprint

    root = _mock_data_dir()
    bundle = _supply_bundle(root)
    top_k = _top_k_per_node()
    node_evidence = [
        _retrieve_for_node(
            state=state,
            constraints=constraints,
            intent=intent,
            bundle=bundle,
            top_k=top_k,
        )
        for intent in blueprint.get("node_intents", []) or []
    ]
    covered = [
        block.get("node_id")
        for block in node_evidence
        if block.get("coverage_status") == "covered" and block.get("candidates")
    ]
    weak_or_missing = [
        {
            "node_id": block.get("node_id"),
            "role": block.get("role"),
            "coverage_status": block.get("coverage_status"),
            "missing_reason": block.get("missing_reason"),
        }
        for block in node_evidence
        if block.get("coverage_status") != "covered" or not block.get("candidates")
    ]
    evidence = {
        "version": SUPPORTED_CONTRACT_VERSION,
        "query": str(state.get("user_input") or constraints.get("raw_text") or ""),
        "city": constraints.get("city") or "上海",
        "blueprint_version": blueprint.get("version"),
        "blueprint_template_mode": blueprint.get("template_mode"),
        "node_evidence": node_evidence,
        "retrieval_trace": [
            {
                "step": "local_poi_rag",
                "retriever": "local_poi_memory_bm25_v1+structured_rerank",
                "data_dir": str(root),
                "top_k_per_node": top_k,
                "covered_node_count": len(covered),
                "weak_or_missing_node_count": len(weak_or_missing),
                "named_event_location_anchors": _named_event_location_anchor_terms(state, constraints),
            }
        ],
        "coverage_summary": {
            "covered_node_ids": covered,
            "weak_or_missing_nodes": weak_or_missing,
        },
    }
    execution_log.append(
        "[B] b_poi_rag_node retrieved local POI evidence "
        f"(nodes={len(node_evidence)}, covered={len(covered)}, data_dir={root.name})"
    )
    return {
        "constraints": constraints,
        "b_itinerary_blueprint": blueprint,
        "b_rag_candidate_evidence": evidence,
        "b_poi_rag_metadata": evidence["retrieval_trace"][0],
        "execution_log": execution_log,
    }
