"""Prompt-wrapped intent parser for the WeekendFlow A-stage demo."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from src.nodes._utils import append_log, merge_tool_results
from src.nodes.longcat_client import (
    DEFAULT_LONGCAT_BASE_URL,
    DEFAULT_LONGCAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    TRUTHY_VALUES,
    LongCatConfig,
    chat_completion,
    sanitize_longcat_error,
)
from src.nodes.taxonomy import (
    SCENE_TYPES,
    canonicalize_tags,
    dedupe,
    tags_by_category,
    tags_from_text,
    to_chinese_tags,
)
from src.state import PlanState


INTENT_PARSER_PROMPT = """你是 WeekendFlow 的 Intent Parser。
请把用户的本地生活需求解析为 JSON intent，字段包含:
task_type, goal, scene, time, people, location, budget,
planning_preferences, constraints, missing_slots, confidence。
只使用下列解析关键词做槽位抽取和 planning tags，不要补充商家、价格、距离、库存或预约结果:
people: family, wife, partner, child, friends, group_activity, group_friendly, social
activity: parent_child, kid_friendly, low_intensity, indoor, outdoor, citywalk, local_market, micro_vacation, wellness, karaoke, sports
food: low_calorie, light_food, healthy, japanese, hotpot, bbq, dine_in, takeaway_only
emotion: relaxation, healing, ritual, quiet, atmosphere, lively, romantic, comfortable, novelty
route: nearby, short_distance, same_area, cross_area_ok, driving, walking, transit, bicycling
budget: budget, low_budget, value_for_money, per_person_budget, total_budget
risk: long_queue, crowded_mall, crowded, high_calorie, too_far
execution: bookable, ticket_required, reservation_needed, walk_in_ok, has_inventory, has_time_slot
只输出结构化 JSON，不输出解释。
"""

A_LLM_INTENT_SYSTEM_PROMPT = """你是 WeekendFlow A 阶段的 Intent Parser。
你的任务是把用户真实自然语言请求解析成稳定 JSON，供下游 B/C 阶段直接消费。

必须只返回 JSON object，不要 Markdown，不要解释。
JSON schema:
{
  "task_type": "local_life_plan" | "clarify_request",
  "goal": string,
  "scene": "family" | "friends" | "couple" | "low_budget" | "solo" | "unknown",
  "time": {
    "window": string,  // 保留用户表达的相对日期/时段，如 today_afternoon, tomorrow_night, weekend, full_day, unspecified
    "duration_range": [number, number],  // 单位为小时；若用户给出明确起止时间，按起止时间计算
    "start_time": "HH:MM" | null,
    "end_time": "HH:MM" | null
  },
  "people": [
    {
      "role": "self" | "wife" | "partner" | "child" | "friends",
      "age": number | null,
      "state": string | null,
      "needs": [string]
    }
  ],
  "location": {
    "origin": string,
    "current_location": string | null,
    "route_origin": string | null,
    "distance_preference": "nearby" | "flexible" | "cross_area_ok" | "unknown",
    "max_distance_km": number | null,
    "transport_mode": "driving" | "walking" | "transit" | "bicycling" | "unknown",
    "route_mode": "driving" | "walking" | "transit" | "bicycling" | "unknown",
    "city": string | null,
    "current_city": string | null,
    "trip_city": string | null,
    "destination_city": string | null
  },
  "budget": {"amount": number | null, "type": "total" | "per_person" | null, "sensitivity": string},
  "planning_preferences": {
    "activity_type": [string],
    "food_type": [string],
    "emotion_type": [string],
    "atmosphere_type": [string],
    "experience_type": [string],
    "restaurant_type": [string],
    "pace": string
  },
  "constraints": {"hard": [string], "soft": [string], "avoid": [string]},
  "people_count": number,
  "ritual_need": boolean,
  "emotion_need": [string],
  "missing_slots": [string],
  "confidence": object,
  "raw_text": string
}

解析关键词只允许来自下列集合，中文原词也可以保留在 raw_text:
people: family, wife, partner, child, friends, group_activity, group_friendly, social
activity: parent_child, kid_friendly, low_intensity, indoor, outdoor, citywalk, local_market, micro_vacation, wellness, karaoke, sports
food: low_calorie, light_food, healthy, japanese, hotpot, bbq, dine_in, takeaway_only
emotion: relaxation, healing, ritual, quiet, atmosphere, lively, romantic, comfortable, novelty
route: nearby, short_distance, same_area, cross_area_ok, driving, walking, transit, bicycling
budget: budget, low_budget, value_for_money, per_person_budget, total_budget
risk: long_queue, crowded_mall, crowded, high_calorie, too_far
execution: bookable, ticket_required, reservation_needed, walk_in_ok, has_inventory, has_time_slot
时间必须尽量保留用户说出的完整区间；例如“下午2点到5点”应输出 start_time="14:00", end_time="17:00", duration_range=[3,3]。
跨城请求必须拆分当前位置与出游目的地；例如“我人在上海，周末去青岛”应输出 current_city="上海", trip_city="青岛", destination_city="青岛", city="青岛", route_origin=null。
current_city/current_location 只表示用户当前所在城市或地点，不能当成本地路线起点；只有明确坐标或“从某具体地点出发”才能填写 route_origin。
预算只抽取用户明说的金额；“省钱/便宜/预算有限”只能体现为 sensitivity 和 low_budget/value_for_money 标签，amount 必须为 null。
不要编造商家、价格、距离、库存或预约结果。
"""

A_LLM_ENABLE_ENV_KEYS = ("WF_A_LLM_ENABLED", "WF_A_AI_ENABLED")
A_LLM_API_FORMAT = "openai"
A_LLM_PROVIDER = "longcat"


CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

DAY_LEVEL_PLAN_TERMS = ("全天", "一整天", "一天", "一日", "全日")
REST_WINDOW_TERMS = ("午睡", "小睡", "睡觉", "休息", "小憩", "休整")
MEAL_ANCHOR_DEFINITIONS = {
    "lunch": {
        "label": "午餐",
        "meal_type": "午餐",
        "part_of_day": "午间",
        "start_time": "12:00",
        "end_time": "13:10",
        "compatibility_role": "restaurant_lunch",
        "terms": ("午饭", "午餐", "中饭", "中午吃", "中午用餐", "中午就餐"),
    },
    "dinner": {
        "label": "晚餐",
        "meal_type": "晚餐",
        "part_of_day": "晚间",
        "start_time": "18:00",
        "end_time": "19:20",
        "compatibility_role": "restaurant_dinner",
        "terms": ("晚饭", "晚餐", "晚上吃", "晚上用餐", "晚上就餐"),
    },
}
CHILD_COMPANION_TERMS = ("孩子", "小孩", "小朋友", "儿童", "亲子", "宝宝", "带娃")
ADULT_COMPANION_TERMS = (
    "老婆",
    "妻子",
    "太太",
    "媳妇",
    "对象",
    "伴侣",
    "爱人",
    "女朋友",
    "男朋友",
)
PARENT_COMPANION_TERMS = (
    "爸妈",
    "父母",
    "爸爸妈妈",
    "妈妈爸爸",
    "爸爸",
    "父亲",
    "妈妈",
    "母亲",
    "外婆",
    "外公",
    "爷爷",
    "奶奶",
    "姥姥",
    "姥爷",
)
SPOUSE_COMPANION_TERMS = ("老婆", "妻子", "太太", "媳妇")
PARTNER_COMPANION_TERMS = ("对象", "情侣", "约会", "女朋友", "男朋友", "伴侣", "爱人")
CHILD_EXCLUSION_TERMS = (
    "不带孩子",
    "孩子不带",
    "孩子去外婆家",
    "孩子在外婆家",
    "孩子去了外婆家",
    "孩子放外婆家",
    "孩子交给外婆",
    "不带小孩",
    "小孩不带",
    "不带小朋友",
    "小朋友不带",
    "不带娃",
    "娃不带",
    "不带宝宝",
    "宝宝不带",
    "没有孩子",
    "这次没孩子",
    "这次没有孩子",
    "这次孩子不去",
    "孩子不去",
    "孩子不同行",
    "孩子不一起",
    "不是亲子",
    "不要亲子",
    "别按亲子",
    "别再给我排亲子",
    "排除亲子",
)
FRIEND_COMPANION_TERMS = (
    "朋友",
    "同事",
    "同学",
    "哥们",
    "闺蜜",
    "伙伴",
    "客户",
    "客人",
    "女生",
    "男生",
    "姐妹",
)
PARTNER_FRIEND_SUBSTRINGS = ("女朋友", "男朋友")
NEGATED_RESTAURANT_GROUP_TERMS = {
    "hotpot": ("火锅",),
    "bbq": ("烤肉", "烧烤"),
}
NEGATED_ACTIVITY_GROUP_TERMS = {
    "parent_child": ("亲子", "亲子乐园", "儿童乐园", "游乐园"),
    "kid_friendly": ("亲子", "亲子乐园", "儿童乐园"),
    "amusement": ("亲子乐园", "儿童乐园", "游乐园"),
    "museum": ("博物馆", "展览", "看展", "美术馆"),
    "karaoke": ("KTV", "ktv", "唱歌", "欢唱", "卡拉OK", "卡拉ok"),
}
CITY_NAMES = (
    "上海",
    "北京",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "南京",
    "苏州",
    "青岛",
)


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _read_float(env: Mapping[str, str], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        raw_value = env.get(key)
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return default


def _read_int(env: Mapping[str, str], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        raw_value = env.get(key)
        if not raw_value:
            continue
        try:
            return int(raw_value)
        except ValueError:
            continue
    return default


def is_a_llm_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether A-stage LLM parsing should be attempted."""

    env = _env_mapping(env)
    for key in A_LLM_ENABLE_ENV_KEYS:
        if key in env:
            return env.get(key, "").strip().lower() in TRUTHY_VALUES
    return any(
        (env.get(key) or "").strip()
        for key in (
            "WF_A_LLM_API_KEY",
            "WF_A_LLM_APP_KEY",
            "LONGCAT_API_KEY",
            "LONGCAT_APP_KEY",
        )
    )


def load_a_llm_config(env: Mapping[str, str] | None = None) -> LongCatConfig | None:
    """Load A-stage OpenAI-compatible LongCat config when available."""

    env = _env_mapping(env)
    if not is_a_llm_enabled(env):
        return None

    api_key = (
        env.get("WF_A_LLM_API_KEY")
        or env.get("WF_A_LLM_APP_KEY")
        or env.get("LONGCAT_API_KEY")
        or env.get("LONGCAT_APP_KEY")
        or ""
    ).strip()
    if not api_key:
        return None

    base_url = (
        env.get("WF_A_LLM_BASE_URL")
        or env.get("LONGCAT_BASE_URL")
        or DEFAULT_LONGCAT_BASE_URL
    ).strip().rstrip("/")
    model = (env.get("WF_A_LLM_MODEL") or env.get("LONGCAT_MODEL") or DEFAULT_LONGCAT_MODEL).strip()

    return LongCatConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=_read_float(
            env,
            ("WF_A_LLM_TIMEOUT_SECONDS", "LONGCAT_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
        ),
        max_tokens=_read_int(
            env,
            ("WF_A_LLM_MAX_TOKENS", "LONGCAT_MAX_TOKENS"),
            max(DEFAULT_MAX_TOKENS, 900),
        ),
        temperature=_read_float(
            env,
            ("WF_A_LLM_TEMPERATURE", "LONGCAT_TEMPERATURE"),
            DEFAULT_TEMPERATURE,
        ),
    )


def _sanitize_a_llm_error(error: BaseException, env: Mapping[str, str] | None = None) -> str:
    text = sanitize_longcat_error(error, env=env)
    env = _env_mapping(env)
    for key_name in ("WF_A_LLM_API_KEY", "WF_A_LLM_APP_KEY"):
        key_value = env.get(key_name)
        if key_value:
            text = text.replace(key_value, "<redacted>")
    return text


def normalize_user_input(raw_input: Any) -> str:
    """Normalize supported user input shapes into a single text string."""

    if raw_input is None:
        return ""

    if isinstance(raw_input, str):
        return raw_input.strip()

    if isinstance(raw_input, dict):
        for key in ("content", "text", "input", "query", "user_input"):
            value = raw_input.get(key)
            if value:
                return normalize_user_input(value)
        return ""

    if isinstance(raw_input, (list, tuple)):
        for item in reversed(raw_input):
            if isinstance(item, dict):
                role = str(item.get("role", item.get("type", ""))).lower()
                if role and role not in {"user", "human"}:
                    continue
            text = normalize_user_input(item)
            if text:
                return text
        return ""

    return str(raw_input).strip()


def _extract_age(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*岁", text)
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 < age < 18 else None


def _extract_budget(text: str) -> tuple[int | None, str | None]:
    per_person_patterns = (
        r"(?:人均|每人|一人|单人)\s*(?:预算|消费|花费)?\s*"
        r"(?:不超过|别超过|不要超过|控制在|最多|大概|约|左右|以内|以下|不超)?\s*(\d{2,5})",
        r"(?:不超过|别超过|不要超过|控制在|最多|不超)?\s*(\d{2,5})\s*元?\s*/\s*人",
        r"(\d{2,5})\s*元?\s*(?:每人|一人|人均)",
    )
    for pattern in per_person_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), "per_person"

    total_patterns = (
        r"(?:总预算|预算|总共|一共|合计|总价|全部)\s*"
        r"(?:不超过|别超过|不要超过|控制在|最多|大概|约|左右|以内|以下|不超)?\s*(\d{2,5})",
        r"(?:不超过|别超过|不要超过|控制在|最多|不超)\s*(\d{2,5})",
        r"(\d{2,5})\s*元?\s*(?:以内|以下)(?:\s*(?:总共|一共|总预算|全部))?",
    )
    for pattern in total_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), "total"

    return None, None


def _extract_people_count(text: str) -> int | None:
    count_token = r"(\d{1,2}|[一二两俩三四五六七八九十])"
    companion_terms = r"(?:朋友|同事|同学|哥们|闺蜜|伙伴|客户|客人|女生|男生|姐妹)"
    companion_modifier = (
        r"(?:(?:外地|本地|重要|商务|公司|大学|高中|初中|毕业|毕业前)\s*)*"
    )
    role_terms = r"(?:大人|成人|成年人|孩子|小孩|儿童|老人|长辈|朋友|同事|同学|哥们|闺蜜|伙伴|客户|客人|女生|男生|姐妹)"

    def parse_count(raw: str) -> int | None:
        if raw.isdigit():
            count = int(raw)
        else:
            count = CHINESE_NUMBER_MAP.get(raw)
        return count if count is not None and 0 < count <= 20 else None

    total_match = re.search(
        rf"(?:我们|咱们|一共|总共|共|合计|总计|总人数)\s*(?:有|是)?\s*"
        rf"{count_token}\s*(?:个)?\s*(?:个人|人|位)",
        text,
    )
    if total_match:
        return parse_count(total_match.group(1))

    family_match = re.search(rf"一家\s*{count_token}\s*口", text)
    if family_match:
        return parse_count(family_match.group(1))

    explicit_group_total_match = re.search(
        rf"(?:我们|咱们)\s*{count_token}\s*(?:个|位)?\s*"
        rf"{companion_modifier}\s*{companion_terms}",
        text,
    )
    if explicit_group_total_match:
        return parse_count(explicit_group_total_match.group(1))

    bare_total_boundary = (
        r"(?=\s*(?:$|[，。；;,.、]|总预算|预算|人均|每人|左右|以内|以下|"
        r"\d{1,2}\s*(?:[:：]|点)|"
        r"一起|在|出发|集合|碰头|聚|聚会|吃|玩|去|散|同行|同去|都))"
    )
    bare_total_match = re.search(
        rf"(?<!孩子)(\d{{1,2}})\s*(?:个人|人|位)(?!\s*{role_terms})"
        rf"{bare_total_boundary}",
        text,
    )
    if bare_total_match:
        count = int(bare_total_match.group(1))
        return count if 0 < count <= 20 else None

    bare_total_match = re.search(
        rf"([一二两俩三四五六七八九十])\s*(?:个人|人|位)(?!\s*{role_terms})"
        rf"{bare_total_boundary}",
        text,
    )
    if bare_total_match:
        return CHINESE_NUMBER_MAP.get(bare_total_match.group(1))

    adult_count = None
    adult_label = None
    adult_match = re.search(rf"{count_token}\s*(?:位|个)?\s*(大人|成人|成年人)", text)
    if adult_match:
        adult_count = parse_count(adult_match.group(1))
        adult_label = adult_match.group(2)
    elif re.search(r"(?:我们|咱们)\s*(?:这些|几个)?\s*大人|大人们", text):
        adult_count = 2

    adult_count_is_group_total = bool(
        adult_count is not None
        and re.search(
            rf"(?:只有|只|就|仅|仅有)?\s*{count_token}\s*(?:位|个)?\s*(?:大人|成人|成年人)",
            text,
        )
        and _contains_any(text, CHILD_EXCLUSION_TERMS)
    )

    child_count = 0
    child_match = re.search(
        rf"(?<!周)(?<!星期)(?<!礼拜){count_token}\s*(?:位|个)?\s*(?:孩子|小孩|儿童)",
        text,
    )
    if child_match:
        child_count = parse_count(child_match.group(1)) or 0
    elif _has_current_child_companion_signal(text):
        child_count = 1

    if adult_count is not None and adult_label == "成年人" and not child_count:
        return adult_count

    elder_count = 0
    elder_match = re.search(rf"{count_token}\s*(?:位|个)?\s*(?:老人|长辈)", text)
    if elder_match:
        elder_count = parse_count(elder_match.group(1)) or 0
    elif _has_current_parent_companion_signal(text) and _contains_any(
        text, ("爸妈", "父母", "爸爸妈妈", "妈妈爸爸")
    ):
        elder_count = 2
    elif _has_current_parent_companion_signal(text):
        parent_hits = 0
        if _contains_any(text, ("爸爸", "父亲", "外公", "爷爷", "姥爷")):
            parent_hits += 1
        if _contains_any(text, ("妈妈", "母亲", "外婆", "奶奶", "姥姥")):
            parent_hits += 1
        elder_count = max(1, parent_hits)

    if adult_count_is_group_total and not child_count:
        return adult_count

    if child_count or elder_count:
        if (
            adult_count is None
            and child_count
            and _contains_any(text, ("我们", "咱们"))
            and _contains_any(text, ("大人", "成人"))
        ):
            adult_count = 2
        if adult_count is None and _contains_any(text, ADULT_COMPANION_TERMS):
            adult_count = 2
        responsible_adults = adult_count if adult_count is not None else 1
        family_count = responsible_adults + child_count + elder_count
        if 0 < family_count <= 20:
            return family_count

    companion_match = re.search(
        rf"(?:和|跟|约|带上|带|叫上|邀|邀请)\s*{count_token}\s*(?:个|位)?\s*"
        rf"{companion_modifier}\s*{companion_terms}",
        text,
    )
    if companion_match:
        count = parse_count(companion_match.group(1))
        if count is not None and count < 20:
            return count + 1

    hosted_customer_match = re.search(
        rf"{count_token}\s*(?:个|位)?\s*{companion_modifier}\s*(?:客户|客人)",
        text,
    )
    colleague_host_match = re.search(
        rf"(?:我|我们|咱们)[^，。；;,.]{{0,12}}(?:和|跟)"
        rf"\s*(?:(?P<count>{count_token})\s*(?:个|位)?)?\s*同事",
        text,
    )
    if hosted_customer_match and colleague_host_match:
        customer_count = parse_count(hosted_customer_match.group(1))
        colleague_raw = colleague_host_match.group("count")
        colleague_count = parse_count(colleague_raw) if colleague_raw else 1
        if customer_count is not None and colleague_count is not None:
            total = customer_count + colleague_count + 1
            if 0 < total <= 20:
                return total

    if hosted_customer_match and re.search(
        r"(?:我|我们|咱们|临时)?[^，。；;,.]{0,12}(?:接待|陪|带|送|招待)",
        text,
    ):
        customer_count = parse_count(hosted_customer_match.group(1))
        if customer_count is not None and customer_count < 20:
            return customer_count + 1

    hosted_visitor_match = re.search(
        rf"{count_token}\s*(?:个|位)?\s*{companion_modifier}\s*"
        rf"(?:朋友|同学|闺蜜|伙伴|姐妹)"
        r"[^，。；;,.]{0,24}(?:来|到|抵达|过来)",
        text,
    )
    if hosted_visitor_match and re.search(
        r"(?:我|我们|咱们)[^，。；;,.]{0,24}(?:接|陪|带|送|招待|碰头)",
        text,
    ):
        visitor_count = parse_count(hosted_visitor_match.group(1))
        if visitor_count is not None and visitor_count < 20:
            return visitor_count + 1

    group_match = re.search(
        rf"(?<![和跟约带邀]){count_token}\s*(?:个|位)?\s*{companion_modifier}\s*{companion_terms}",
        text,
    )
    if group_match:
        return parse_count(group_match.group(1))

    return None


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _has_explicit_child_exclusion(text: str) -> bool:
    if _contains_any(text, CHILD_EXCLUSION_TERMS):
        return True
    return bool(
        re.search(
            r"(?:孩子|小孩|小朋友|娃|宝宝)"
            r"[^，。；;,.]{0,12}"
            r"(?:放|去|在|留在|交给|托给)"
            r"[^，。；;,.]{0,12}"
            r"(?:外婆|外公|爷爷|奶奶|姥姥|姥爷|长辈|老人|家里)"
            r"[^，。；;,.]{0,10}"
            r"(?:不带|不同行|不一起|不去|不来)?",
            text,
        )
    )


def _negated_restaurant_groups(text: str) -> set[str]:
    groups: set[str] = set()
    for group, terms in NEGATED_RESTAURANT_GROUP_TERMS.items():
        for term in terms:
            if re.search(
                rf"(?:不要|别|不想|不吃|避免|避开|不要安排|别安排)[^，。；;,.]{{0,12}}{term}",
                text,
            ):
                groups.add(group)
                break
    return groups


def _negated_activity_groups(text: str) -> set[str]:
    groups: set[str] = set()
    for group, terms in NEGATED_ACTIVITY_GROUP_TERMS.items():
        for term in terms:
            if re.search(
                rf"(?:不要|别|不想|避免|避开|不要安排|别安排|别推荐|别再给我排)"
                rf"[^，。；;,.]{{0,16}}{term}",
                text,
            ):
                groups.add(group)
                break
    return groups


def _remove_group_tags(values: list[str], groups: set[str]) -> list[str]:
    if not groups:
        return values
    blocked = set(groups)
    if "hotpot" in groups:
        blocked.add("火锅")
    if "bbq" in groups:
        blocked.update({"烤肉", "烧烤"})
    if "karaoke" in groups:
        blocked.update({"KTV欢唱", "唱歌", "欢唱", "lively"})
    return [value for value in values if value not in blocked]


def _without_tags(values: list[str], blocked: set[str]) -> list[str]:
    return [value for value in values if value not in blocked]


def _has_negated_diet_preference(text: str) -> bool:
    diet_terms = ("减脂", "低卡", "轻食", "控制饮食", "健康餐", "减肥")
    direct_rejection = re.search(
        r"(?:不是|不用|不要|不按|别按|别再按|别套用|别提醒|不想被提醒)"
        r"[^，。；;,.]{0,16}(?:减脂|低卡|轻食|控制饮食|健康餐|减肥)",
        text,
    )
    if direct_rejection:
        return True
    explicit_limit_rejection = re.search(
        r"(?:不想|不愿|不用|不要)[^，。；;,.]{0,8}"
        r"(?:被|受)?[^，。；;,.]{0,8}"
        r"(?:减脂|低卡|轻食|控制饮食|健康餐|减肥)"
        r"[^，。；;,.]{0,8}(?:限制|约束|绑住|影响)",
        text,
    )
    if explicit_limit_rejection:
        return True

    old_markers = ("以前", "历史", "旧偏好", "平时", "之前", "上次", "那种", "那套")
    reject_markers = ("不要按", "别按", "别再按", "别把", "别套", "不要套用", "不套用")
    for clause in re.split(r"[，。；;,.]", text):
        if (
            _contains_any(clause, diet_terms)
            and _contains_any(clause, old_markers)
            and _contains_any(clause, reject_markers)
        ):
            return True
    return False


def _has_day_level_plan(text: str) -> bool:
    return _contains_any(text, DAY_LEVEL_PLAN_TERMS)


def _is_deadline_context(text: str, start_index: int, end_index: int) -> bool:
    context = text[max(0, start_index - 8) : min(len(text), end_index + 6)]
    return bool(
        re.search(
            r"(?:最晚|不晚于|别超过|不要超过|不能超过|不超过|别超|不超)[^，。；;,.]{0,8}$",
            text[max(0, start_index - 16) : start_index],
        )
        or re.search(
            r"^(?:前|之前|以前|内|以内)",
            text[end_index : min(len(text), end_index + 6)],
        )
        or re.search(r"(?:前|之前|以前|内|以内)", context)
    )


def _is_rest_window_context(text: str, start_index: int, end_index: int) -> bool:
    context = text[max(0, start_index - 12) : min(len(text), end_index + 18)]
    return _contains_any(context, REST_WINDOW_TERMS)


def _format_clock_time(hour: int, minute: int, period: str = "") -> str | None:
    if period in {"下午", "晚上", "今晚"} and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"

    return None


def _default_hour_for_period(period: str | None) -> int | None:
    return {
        "早上": 9,
        "上午": 9,
        "中午": 12,
        "下午": 14,
        "晚上": 18,
        "今晚": 18,
    }.get(period or "")


def _weekday_index(token: str) -> int | None:
    if token in {"周一", "星期一", "礼拜一"}:
        return 1
    if token in {"周二", "星期二", "礼拜二"}:
        return 2
    if token in {"周三", "星期三", "礼拜三"}:
        return 3
    if token in {"周四", "星期四", "礼拜四"}:
        return 4
    if token in {"周五", "星期五", "礼拜五"}:
        return 5
    if token in {"周六", "星期六", "礼拜六"}:
        return 6
    if token in {"周日", "周天", "星期日", "星期天", "礼拜日", "礼拜天"}:
        return 7
    return None


WEEKDAY_TOKEN_PATTERN = (
    r"周[一二三四五六日天]|星期[一二三四五六日天]|礼拜[一二三四五六日天]"
)


def _has_cross_weekday_span(text: str) -> bool:
    return bool(
        re.search(
            rf"(?:这个|本)?\s*(?:{WEEKDAY_TOKEN_PATTERN})"
            rf"[^，。；;,.]{{0,16}}(?:到|至|[-~—－])"
            rf"[^，。；;,.]{{0,16}}(?:这个|本)?\s*(?:{WEEKDAY_TOKEN_PATTERN})",
            text,
        )
    )


def _cross_weekday_range_end_has_explicit_clock(text: str) -> bool:
    pattern = re.compile(
        rf"(?:这个|本)?\s*(?:{WEEKDAY_TOKEN_PATTERN})"
        rf"[^，。；;,.：:]{{0,16}}(?:到|至|[-~—－])"
        rf"[^，。；;,.：:]{{0,16}}(?:这个|本)?\s*(?:{WEEKDAY_TOKEN_PATTERN})"
        r"(?P<tail>[^，。；;,.：:]{0,12})"
    )
    for match in pattern.finditer(text):
        if re.search(r"\d{1,2}\s*(?:[:：]|点)", match.group("tail")):
            return True
    return False


def _time_from_parts(
    period: str | None,
    hour: str | None,
    colon_minute: str | None,
    half: str | None,
    minute_text: str | None,
) -> str | None:
    parsed_hour = int(hour) if hour else _default_hour_for_period(period)
    if parsed_hour is None:
        return None
    minute = int(colon_minute) if colon_minute else 30 if half else int(minute_text or 0)
    return _format_clock_time(parsed_hour, minute, period or "")


def _is_immediate_pre_deadline_context(text: str, start_index: int, end_index: int) -> bool:
    context = text[start_index : min(len(text), end_index + 4)]
    return bool(
        re.search(
            r"\d{1,2}(?:\s*[:：]\s*\d{1,2}|\s*点(?:半|\d{1,2}分?)?)"
            r"\s*(?:前|之前|以前|内|以内)",
            context,
        )
    )


def _extract_start_time(text: str) -> str | None:
    for match in re.finditer(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", text):
        if _is_deadline_context(text, match.start(), match.end()):
            continue
        return _format_clock_time(int(match.group(1)), int(match.group(2)))

    pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*(\d{1,2})\s*点(?:半|(\d{1,2})分?)?"
    )
    for match in pattern.finditer(text):
        if _is_deadline_context(text, match.start(), match.end()):
            continue
        minute = 30 if "半" in match.group(0) else int(match.group(3) or 0)
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")
    return None


def _extract_post_existing_service_start_time(text: str) -> str | None:
    has_existing_service_anchor = bool(
        re.search(
            r"(?:已|已经)?约好(?:了)?[^，。；;,.]{0,16}(?:不需要|不用)?"
            r"|(?:少儿足球|足球试听|足球训练|足球培训|试听课)[^，。；;,.]{0,28}"
            r"(?:已约好|已经约好|约好了|不需要再|不用再|不需要|不用)",
            text,
        )
    )
    if not has_existing_service_anchor:
        return None

    pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)"
        r"\s*(?:以后|之后|后)[^，。；;,.]{0,18}"
        r"(?:吃饭|晚饭|晚餐|餐厅|接|去接|开始|出发)"
    )
    for match in pattern.finditer(text):
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")
    return None


def _extract_anchored_start_time(text: str) -> str | None:
    pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)"
        r"\s*(?:左右|前后|以后|之后|后)?[^，。；;,.]{0,18}"
        r"(?:出发|开始|集合)"
    )
    for match in pattern.finditer(text):
        if _is_immediate_pre_deadline_context(text, match.start(), match.end()):
            continue
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")

    period_only_pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)"
        r"\s*(?:左右|前后|以后|之后|后)?[^，。；;,.]{0,18}"
        r"(?:出发|开始|集合)"
    )
    for match in period_only_pattern.finditer(text):
        if _is_immediate_pre_deadline_context(text, match.start(), match.end()):
            continue
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        hour = _default_hour_for_period(match.group(1))
        if hour is not None:
            return _format_clock_time(hour, 0, "")

    hosted_after_pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)"
        r"\s*(?:以后|之后|后)[^，。；;,.]{0,18}"
        r"(?:我|我们|咱们)[^，。；;,.]{0,20}(?:带|陪|安排|去|看)"
    )
    for match in hosted_after_pattern.finditer(text):
        if _is_immediate_pre_deadline_context(text, match.start(), match.end()):
            continue
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")
    return None


def _extract_deadline_time(text: str) -> str | None:
    leading_pattern = re.compile(
        r"(?:最晚|不晚于|别超过|不要超过|不能超过|不超过|别超|不超)\s*"
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)"
    )
    for match in leading_pattern.finditer(text):
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")

    anchored_pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)"
        r"\s*(?:左右|前后)?[^，。；;,.]{0,24}"
        r"(?:散|结束|离开|送站|送[^，。；;,.]{0,8}(?:走|回)|回程)"
    )
    for match in anchored_pattern.finditer(text):
        if len(re.findall(r"\d{1,2}\s*(?:[:：]|点)", match.group(0))) > 1:
            continue
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")

    pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)"
        r"\s*(?:前|之前|以前)"
        r"(?:[^，。；;,.]{0,12}(?:散|结束|回|到|离开|赶到|抵达))?"
    )
    for match in pattern.finditer(text):
        if _is_rest_window_context(text, match.start(), match.end()):
            continue
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        return _format_clock_time(int(match.group(2)), minute, match.group(1) or "")
    return None


def _clock_to_minutes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d{2}):(\d{2})", value)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _duration_hours_from_times(start_time: str | None, end_time: str | None) -> float | None:
    start_minutes = _clock_to_minutes(start_time)
    end_minutes = _clock_to_minutes(end_time)
    if start_minutes is None or end_minutes is None:
        return None
    if end_minutes <= start_minutes:
        end_minutes += 24 * 60
    duration = (end_minutes - start_minutes) / 60
    return duration if 0 < duration <= 24 else None


def _extract_time_range(text: str) -> tuple[str | None, str | None, float | None]:
    cross_day_pattern = re.compile(
        rf"(?:这个|本)?\s*({WEEKDAY_TOKEN_PATTERN})"
        r"\s*(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(?:(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?)?"
        r"\s*(?:到|至|[-~—－])\s*"
        rf"(?:这个|本)?\s*({WEEKDAY_TOKEN_PATTERN})"
        r"\s*(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(?:(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?)?"
    )
    for match in cross_day_pattern.finditer(text):
        start_time = _time_from_parts(
            match.group(2),
            match.group(3),
            match.group(4),
            match.group(5),
            match.group(6),
        )
        end_time = _time_from_parts(
            match.group(8),
            match.group(9),
            match.group(10),
            match.group(11),
            match.group(12),
        )
        if not start_time or not end_time:
            continue
        start_minutes = _clock_to_minutes(start_time)
        end_minutes = _clock_to_minutes(end_time)
        if start_minutes is None or end_minutes is None:
            continue
        start_day = _weekday_index(match.group(1))
        end_day = _weekday_index(match.group(7))
        day_delta = 1 if start_day is not None and end_day is not None and end_day > start_day else 0
        duration = (end_minutes + day_delta * 24 * 60 - start_minutes) / 60
        if duration <= 0:
            duration += 24
        return start_time, end_time, duration

    range_pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?"
        r"\s*(?:到|至|[-~—－])\s*"
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?"
    )
    for match in range_pattern.finditer(text):
        if _is_rest_window_context(text, match.start(), match.end()) and (
            _has_day_level_plan(text)
            or _contains_any(text, ("周末", "两天", "两日", "2天", "两天一夜", "住一晚"))
        ):
            continue
        matched_text = match.group(0)
        has_time_unit = any(token in matched_text for token in ("点", ":", "："))
        has_period = bool(match.group(1) or match.group(6))
        if not has_time_unit and not has_period:
            continue

        start_period = match.group(1) or ""
        end_period = match.group(6) or start_period
        start_minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        end_minute = (
            int(match.group(8))
            if match.group(8)
            else 30
            if match.group(9)
            else int(match.group(10) or 0)
        )
        start_time = _format_clock_time(int(match.group(2)), start_minute, start_period)
        end_time = _format_clock_time(int(match.group(7)), end_minute, end_period)
        duration = _duration_hours_from_times(start_time, end_time)
        if start_time and end_time and duration is not None:
            return start_time, end_time, duration
    return None, None, None


def _extract_separated_weekend_duration(
    text: str,
    start_time: str | None,
    end_time: str | None,
) -> float | None:
    if not start_time or not end_time:
        return None
    if not _contains_any(text, ("周六", "星期六", "礼拜六")):
        return None
    if not _contains_any(text, ("周日", "周天", "星期日", "星期天", "礼拜日", "礼拜天")):
        return None
    if not _contains_any(text, ("两天", "两日", "2天", "两天一夜", "周末")):
        return None

    start_minutes = _clock_to_minutes(start_time)
    end_minutes = _clock_to_minutes(end_time)
    if start_minutes is None or end_minutes is None:
        return None
    duration = (end_minutes + 24 * 60 - start_minutes) / 60
    return duration if 12 <= duration <= 48 else None


def _extract_route_mode(text: str) -> str:
    if _contains_any(text, ("开车", "自驾", "打车")):
        return "driving"
    if _contains_any(text, ("步行", "走路")):
        return "walking"
    if _contains_any(text, ("公交", "地铁")):
        return "transit"
    if _contains_any(text, ("骑车", "单车")):
        return "bicycling"
    return "unknown"


def _normalize_city_name(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("市"):
        text = text[:-1]
    for city in CITY_NAMES:
        if city == text or city in text:
            return city
    return text or None


def _extract_city(text: str) -> str | None:
    city_pattern = "|".join(re.escape(city) for city in CITY_NAMES)
    destination_patterns = (
        rf"目的地城市[:：]?\s*({city_pattern})",
        rf"(?:去|到)({city_pattern})(?:玩|过周末|旅行|旅游|度假|出差|逛|吃|住)?",
        rf"(?:在|想在|周末在|周末是在)({city_pattern})(?:玩|过周末|旅行|旅游|度假|逛|吃)",
    )
    for pattern in destination_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    for city in CITY_NAMES:
        if city in text:
            return city
    return None


def _extract_current_city(text: str) -> str | None:
    for city in CITY_NAMES:
        current_patterns = (
            rf"(?:我|本人)?人\s*在\s*{city}",
            rf"(?:我|本人)?(?:现在|目前|当前|此刻)\s*(?:在|位于)\s*{city}",
            rf"(?:我|本人)?(?:住在|常住|定位在)\s*{city}",
            rf"从\s*{city}\s*(?:出发)?\s*(?:去|到|前往|飞|开车去|自驾去)",
        )
        if any(re.search(pattern, text) for pattern in current_patterns):
            return city
    return None


def _extract_destination_city(text: str, current_city: str | None = None) -> str | None:
    destination_hits: list[tuple[int, str]] = []
    for city in CITY_NAMES:
        destination_patterns = (
            rf"(?:去|到|前往|飞去|飞|开车去|自驾去|周末去|想去|打算去|计划去|准备去)\s*{city}",
            rf"{city}\s*(?:玩|旅游|旅行|度假|出差|两天|周末|一日游|海边)",
        )
        for pattern in destination_patterns:
            match = re.search(pattern, text)
            if match:
                destination_hits.append((match.start(), city))
                break

    if not destination_hits:
        return None
    if current_city and any(city != current_city for _, city in destination_hits):
        return next(city for _, city in sorted(destination_hits) if city != current_city)
    return sorted(destination_hits)[0][1]


def _extract_trip_city_fields(text: str) -> dict[str, str | None]:
    current_city = _extract_current_city(text)
    destination_city = _extract_destination_city(text, current_city=current_city)
    city = destination_city or _extract_city(text)
    if city is None and current_city is not None:
        city = current_city
    trip_city = destination_city or city
    return {
        "current_city": current_city,
        "trip_city": trip_city,
        "destination_city": destination_city or trip_city,
        "city": city,
    }


def _extract_location_origin(text: str) -> str:
    coordinate = re.search(r"(\d{2,3}\.\d+)\s*,\s*(\d{1,2}\.\d+)", text)
    if coordinate:
        return f"{coordinate.group(1)},{coordinate.group(2)}"

    arrival_match = re.search(
        r"(?:朋友|客户|客人|同事|同学|闺蜜|伙伴)?[^，。,.；;]{0,12}"
        r"\d{1,2}\s*[:：]\s*\d{1,2}\s*(?:到|抵达|到达)"
        r"([^，。,.；;]{2,18}?)(?:$|[，。,.；;])",
        text,
    )
    if arrival_match and not re.search(r"(?:送|赶|去|前往)$", text[: arrival_match.start()]):
        return arrival_match.group(1).strip()

    match = re.search(
        r"在([^，。,.；;]{2,18}?)(?:那边|附近)[^，。,.；;]{0,18}"
        r"(?:少儿足球|足球试听|足球训练|足球培训|试听课|训练课)",
        text,
    )
    if match:
        return match.group(1).strip()

    match = re.search(r"(?:在|到)([^，。,.]{2,18}?)(?:接到|接上|接|集合|碰头)", text)
    if match:
        return match.group(1).strip()

    match = re.search(r"从([^，。,.]{2,24}?)(?:出发|附近|开始|走|开车)", text)
    if match:
        if "送" in match.group(0) or re.search(r"(?:飞|航班|飞机)\s*走?", match.group(0)):
            return "unknown"
        return match.group(1).strip()

    match = re.search(r"([^，。,.]{2,18}?)(?:附近|周边)", text)
    if match:
        return match.group(1).strip()

    if _contains_any(text, ("离家", "家附近", "从家")):
        return "home"

    return "unknown"


def _extend_unique(target: list[str], values: Any) -> None:
    target[:] = dedupe(target + canonicalize_tags(values))


def build_intent_prompt(user_input: str) -> str:
    """Build the actual prompt text used by the simple parser."""

    return f"{INTENT_PARSER_PROMPT}\n用户输入: {user_input}\nJSON:"


def _mock_intent(user_input: str) -> dict[str, Any]:
    """Build a schema-complete mock intent used as LLM normalization defaults."""

    return {
        "task_type": "local_life_plan" if user_input else "clarify_request",
        "goal": "安排一次本地生活出行计划" if user_input else "等待用户提供本地生活需求",
        "scene": "unknown",
        "time": {
            "window": "unspecified",
            "duration_range": [3, 6] if user_input else [0, 0],
            "start_time": None,
            "end_time": None,
        },
        "people": [{"role": "self", "needs": []}] if user_input else [],
        "location": {
            "origin": "unknown",
            "current_location": None,
            "route_origin": None,
            "distance_preference": "unknown",
            "max_distance_km": None,
            "transport_mode": "unknown",
            "route_mode": "unknown",
            "city": None,
            "current_city": None,
            "trip_city": None,
            "destination_city": None,
        },
        "budget": {
            "amount": None,
            "type": None,
            "sensitivity": "unknown",
        },
        "planning_preferences": {
            "activity_type": [],
            "food_type": [],
            "emotion_type": [],
            "atmosphere_type": [],
            "experience_type": [],
            "restaurant_type": [],
            "pace": "unknown",
        },
        "constraints": {
            "hard": [],
            "soft": [],
            "avoid": [],
        },
        "people_count": 1 if user_input else 0,
        "ritual_need": False,
        "emotion_need": [],
        "missing_slots": [] if user_input else ["user_input"],
        "confidence": {"user_input": 1.0 if user_input else 0.0},
        "raw_text": user_input,
        "parse_source": "mock",
    }


def build_llm_intent_messages(user_input: str, mock_intent: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for the primary A-stage LLM parser."""

    payload = {
        "user_input": user_input,
        "schema_defaults": mock_intent,
        "allowed_scene_types": sorted(SCENE_TYPES | {"unknown"}),
    }
    return [
        {"role": "system", "content": A_LLM_INTENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _strip_json_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_jsonish(content: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(content)
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.find("{") : cleaned.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _as_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        raw_values = values
    else:
        raw_values = [values]
    return dedupe([str(value).strip() for value in raw_values if str(value).strip()])


def _as_optional_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_optional_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any, default: int | None = None) -> int | None:
    number = _as_optional_float(value)
    if number is None:
        return default
    return int(number)


def _merge_chinese_tags(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        merged.extend(_as_str_list(value))
    return to_chinese_tags(merged)


def _preserve_activity_compatibility_tags(preferences: dict[str, Any]) -> None:
    activity_type = preferences.get("activity_type")
    if not isinstance(activity_type, list):
        return
    if "多人活动" in activity_type and "group_activity" not in activity_type:
        activity_type.append("group_activity")


def _partition_risk_tags(values: Any) -> tuple[list[str], list[str]]:
    canonical_tags = canonicalize_tags(values)
    risk_tags = set(tags_by_category(canonical_tags)["risk"])
    non_risk = [tag for tag in canonical_tags if tag not in risk_tags]
    risks = [tag for tag in canonical_tags if tag in risk_tags]
    return non_risk, risks


def _normalize_constraint_buckets(
    hard: Any,
    soft: Any,
    avoid: Any,
) -> dict[str, list[str]]:
    hard_tags, hard_risks = _partition_risk_tags(hard)
    soft_tags, soft_risks = _partition_risk_tags(soft)
    avoid_tags = canonicalize_tags(avoid) + hard_risks + soft_risks
    return {
        "hard": to_chinese_tags(hard_tags),
        "soft": to_chinese_tags(soft_tags),
        "avoid": to_chinese_tags(avoid_tags),
    }


def _duration_range(raw_value: Any, fallback: list[Any]) -> list[Any]:
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value > 0:
            return [int(value), int(value)] if value == int(value) else [value, value]
    values = raw_value if isinstance(raw_value, list) else fallback
    if not isinstance(values, list) or len(values) < 2:
        return fallback
    start = _as_optional_float(values[0])
    end = _as_optional_float(values[1])
    if start is None or end is None:
        return fallback
    if start > end:
        start, end = end, start
    if start == int(start) and end == int(end):
        return [int(start), int(end)]
    return [start, end]


def _duration_range_from_times(
    start_time: str | None,
    end_time: str | None,
    fallback: list[Any],
) -> list[Any]:
    duration = _duration_hours_from_times(start_time, end_time)
    if duration is None:
        return fallback
    if duration == int(duration):
        return [int(duration), int(duration)]
    return [duration, duration]


def _normalize_clock_time(value: Any, fallback: str | None = None) -> str | None:
    text = _as_optional_str(value)
    if text is None:
        return fallback
    match = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
    if match:
        return _format_clock_time(int(match.group(1)), int(match.group(2))) or fallback
    return _extract_start_time(text) or fallback


def _normalize_scene(raw_scene: Any, fallback: str) -> str:
    scene = str(raw_scene or "").strip()
    aliases = {
        "亲子": "family",
        "家庭": "family",
        "朋友": "friends",
        "多人": "friends",
        "情侣": "couple",
        "约会": "couple",
        "低预算": "low_budget",
        "省钱": "low_budget",
        "单人": "solo",
        "独自": "solo",
    }
    if scene in SCENE_TYPES or scene == "unknown":
        return scene
    return aliases.get(scene, fallback)


def _has_current_child_companion_signal(text: str) -> bool:
    if _has_explicit_child_exclusion(text):
        return False
    return any(term in text for term in CHILD_COMPANION_TERMS) or _extract_age(text) is not None


def _has_current_adult_companion_signal(text: str, terms: tuple[str, ...]) -> bool:
    if not _contains_any(text, terms):
        return False
    term_pattern = "|".join(re.escape(term) for term in terms)
    negated = re.search(
        rf"(?:不带|没带|没有|不是和|不是跟|不和|不跟|别按|别套用)"
        rf"[^，。；;,.]{{0,8}}(?:{term_pattern})",
        text,
    )
    return not negated


def _has_current_parent_companion_signal(text: str) -> bool:
    """Return whether the current request includes parent or elder companions."""

    if not _contains_any(text, PARENT_COMPANION_TERMS):
        return False
    if re.search(r"孩子[^，。；;,.]{0,8}(?:去|在|放|交给)[^，。；;,.]{0,6}外婆家?", text) and not re.search(
        r"(?:带|和|跟|同|陪)[^，。；;,.]{0,6}(?:外婆|老人|长辈)",
        text,
    ):
        return False
    return not re.search(
        r"(?:不带|没带|没有|不是和|不是跟|不和|不跟|别按|别套用)"
        r"[^，。；;,.]{0,8}(?:爸妈|父母|爸爸|妈妈|父亲|母亲|老人|长辈)",
        text,
    )


def _has_current_friend_companion_signal(text: str) -> bool:
    """Return whether the current request has friend/group companions.

    `女朋友` and `男朋友` are partner mentions, not friend-group mentions.
    """

    friend_text = text
    for term in PARTNER_FRIEND_SUBSTRINGS:
        friend_text = friend_text.replace(term, "")
    return _contains_any(friend_text, FRIEND_COMPANION_TERMS)


def _normalize_people(
    raw_people: Any,
    fallback: list[dict[str, Any]],
    *,
    current_text: str = "",
) -> list[dict[str, Any]]:
    if not isinstance(raw_people, list):
        return fallback

    role_aliases = {
        "老婆": "wife",
        "妻子": "wife",
        "太太": "wife",
        "对象": "partner",
        "伴侣": "partner",
        "女朋友": "partner",
        "男朋友": "partner",
        "孩子": "child",
        "小孩": "child",
        "朋友": "friends",
        "同事": "friends",
    }
    people: list[dict[str, Any]] = []
    for item in raw_people:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        role = role_aliases.get(role, role)
        if role not in {"self", "wife", "partner", "child", "friends"}:
            continue
        if role == "child" and current_text and not _has_current_child_companion_signal(current_text):
            continue
        if (
            role == "wife"
            and current_text
            and not _has_current_adult_companion_signal(current_text, SPOUSE_COMPANION_TERMS)
        ):
            continue
        if (
            role == "partner"
            and current_text
            and not _has_current_adult_companion_signal(current_text, PARTNER_COMPANION_TERMS)
        ):
            continue
        if (
            role == "friends"
            and current_text
            and not _has_current_friend_companion_signal(current_text)
        ):
            continue
        normalized = {
            "role": role,
            "needs": _as_str_list(item.get("needs")),
        }
        age = _as_optional_int(item.get("age"))
        if age is not None:
            normalized["age"] = age
        state = _as_optional_str(item.get("state"))
        if state:
            normalized["state"] = state
        count = _as_optional_int(item.get("count"))
        if count is not None:
            normalized["count"] = count
        people.append(normalized)

    if not people:
        return fallback
    if not any(item.get("role") == "self" for item in people):
        people.insert(0, {"role": "self", "needs": []})
    return people


def _extract_rest_time_anchors(text: str) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?"
        r"\s*(?:到|至|[-~—－])\s*"
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?"
    )
    for match in pattern.finditer(text):
        if not _is_rest_window_context(text, match.start(), match.end()):
            continue
        start_period = match.group(1) or ""
        end_period = match.group(6) or start_period
        start_minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        end_minute = (
            int(match.group(8))
            if match.group(8)
            else 30
            if match.group(9)
            else int(match.group(10) or 0)
        )
        start_time = _format_clock_time(int(match.group(2)), start_minute, start_period)
        end_time = _format_clock_time(int(match.group(7)), end_minute, end_period)
        if start_time and end_time:
            anchors.append({"type": "rest", "start_time": start_time, "end_time": end_time})
    return anchors


def _extract_event_time_anchors(text: str) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    pattern = re.compile(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*"
        r"(\d{1,2})(?:\s*[:：]\s*(\d{1,2})|\s*点(半)?(?:(\d{1,2})分?)?)?"
        r"\s*(?:左右|前后)?[^，。；;,.]{0,18}"
        r"(演出|亲子剧|脱口秀|小剧场|评弹)"
    )
    for match in pattern.finditer(text):
        minute = (
            int(match.group(3))
            if match.group(3)
            else 30
            if match.group(4)
            else int(match.group(5) or 0)
        )
        clock = _format_clock_time(int(match.group(2)), minute, match.group(1) or "")
        if clock:
            anchors.append({"type": "event", "time": clock, "label": match.group(6)})

    return anchors


def _extract_meal_time_anchors(text: str, dietary: list[str]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    dietary_labels = to_chinese_tags(dietary)

    for definition in MEAL_ANCHOR_DEFINITIONS.values():
        matched_terms = [term for term in definition["terms"] if term in text]
        if not matched_terms:
            continue
        anchor = {
            "type": "meal",
            "label": definition["label"],
            "meal_type": definition["meal_type"],
            "part_of_day": definition["part_of_day"],
            "start_time": definition["start_time"],
            "end_time": definition["end_time"],
            "compatibility_role": definition["compatibility_role"],
            "source_terms": matched_terms,
        }
        if dietary_labels:
            anchor["dietary"] = dietary_labels
        anchors.append(anchor)

    return anchors


def _explicit_constraint_tags(text: str) -> dict[str, list[str]]:
    dietary: list[str] = []
    accessibility: list[str] = []
    logistics: list[str] = []

    if "坚果过敏" in text or "花生过敏" in text:
        dietary.extend(["allergy_aware", "nut_free"])
    if _contains_any(
        text,
        ("海鲜过敏", "不含海鲜", "避开海鲜", "不吃海鲜", "海鲜不吃", "不能吃海鲜", "海鲜不能吃"),
    ):
        dietary.extend(["allergy_aware", "seafood_free"])
    if _contains_any(
        text,
        (
            "乳糖不耐",
            "乳糖不耐受",
            "乳制品过敏",
            "奶制品过敏",
            "牛奶过敏",
            "奶油过敏",
            "不能吃奶制品",
            "不能吃乳制品",
        ),
    ):
        dietary.extend(["allergy_aware", "lactose_free"])
    if _contains_any(text, ("穆斯林", "清真")):
        dietary.extend(["halal_friendly", "no_pork"])
    if _contains_any(text, ("不含猪肉", "避开猪肉", "无猪肉", "不吃猪肉", "不能吃猪肉")):
        dietary.append("no_pork")
    if _contains_any(
        text,
        (
            "不吃牛羊肉",
            "不能吃牛羊肉",
            "避开牛羊肉",
            "不含牛羊肉",
            "牛羊肉不吃",
            "牛羊肉不能吃",
        ),
    ):
        dietary.append("no_beef_lamb")
    if _contains_any(text, ("吃素", "素食")):
        dietary.append("vegetarian")
    if _contains_any(text, ("不辣", "不能辣", "不能吃辣", "别太辣", "少辣")):
        dietary.append("non_spicy")
    if _contains_any(text, ("不喝酒", "不太喝酒", "不怎么喝酒", "不含酒精", "无酒精", "别喝酒", "不能喝酒")):
        dietary.append("no_alcohol")
    if _contains_any(text, ("低盐", "少盐")):
        dietary.extend(["low_salt", "light_food"])
    if _contains_any(text, ("软一点", "软一些", "别太硬", "不能太硬", "不要太硬")):
        dietary.extend(["soft_food", "light_food"])
    if _contains_any(text, ("控糖", "血糖高", "低糖", "少糖", "不太甜", "不能吃太甜", "别太甜")):
        dietary.append("low_sugar")

    if _contains_any(
        text,
        ("膝盖不好", "膝盖不太好", "不能走太多", "别安排暴走", "少步行", "少走路"),
    ):
        accessibility.extend(["low_walk", "low_intensity", "short_distance"])
    if _contains_any(text, ("婴儿车", "推婴儿车", "推车", "少楼梯", "无障碍")):
        accessibility.extend(["stroller_accessible", "low_walk"])
    if _contains_any(
        text,
        (
            "轮椅",
            "借轮椅",
            "电梯方便",
            "有电梯",
            "少台阶",
            "少楼梯",
            "小手术",
            "术后",
            "刚做完手术",
        ),
    ):
        accessibility.extend(["low_walk", "low_intensity", "short_distance"])
    if _contains_any(text, ("宠物友好", "小狗", "带狗", "带一只狗", "带一只小狗")):
        logistics.append("pet_friendly")
    if _contains_any(text, ("停车", "自驾", "开车")):
        logistics.append("easy_parking")

    return {
        "dietary": dedupe(dietary),
        "accessibility": dedupe(accessibility),
        "logistics": dedupe(logistics),
    }


def _enrich_intent_with_explicit_constraints(
    intent: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    tags = _explicit_constraint_tags(text)
    dietary = tags["dietary"]
    accessibility = tags["accessibility"]
    logistics = tags["logistics"]
    hard_tags = dietary + accessibility
    soft_tags = logistics

    constraints = intent.setdefault("constraints", {"hard": [], "soft": [], "avoid": []})
    constraints["hard"] = _merge_chinese_tags(constraints.get("hard", []), hard_tags)
    constraints["soft"] = _merge_chinese_tags(constraints.get("soft", []), soft_tags)
    constraints["avoid"] = _merge_chinese_tags(constraints.get("avoid", []))

    preferences = intent.setdefault("planning_preferences", {})
    preferences["food_type"] = _merge_chinese_tags(preferences.get("food_type", []), dietary)
    preferences["activity_type"] = _merge_chinese_tags(
        preferences.get("activity_type", []),
        accessibility + logistics,
    )
    _preserve_activity_compatibility_tags(preferences)

    for person in intent.get("people", []):
        role = person.get("role")
        if role == "child" and dietary:
            person["needs"] = _merge_chinese_tags(person.get("needs", []), dietary)
        elif role in {"wife", "partner"} and dietary:
            person["needs"] = _merge_chinese_tags(person.get("needs", []), dietary)
        if role != "self" and accessibility:
            person["needs"] = _merge_chinese_tags(person.get("needs", []), accessibility)

    rest_anchors = _extract_rest_time_anchors(text)
    event_anchors = _extract_event_time_anchors(text)
    meal_anchors = _extract_meal_time_anchors(text, dietary)
    intent["explicit_constraints"] = {
        "dietary": to_chinese_tags(dietary),
        "accessibility": to_chinese_tags(accessibility),
        "logistics": to_chinese_tags(logistics),
        "time_anchors": meal_anchors + rest_anchors + event_anchors,
        "meal_anchors": meal_anchors,
    }
    return intent


def _normalize_confidence(raw_confidence: Any, fallback: dict[str, Any]) -> dict[str, float]:
    confidence: dict[str, float] = {}
    if isinstance(fallback, dict):
        for key, value in fallback.items():
            number = _as_optional_float(value)
            if number is not None:
                confidence[str(key)] = max(0.0, min(1.0, number))
    if isinstance(raw_confidence, dict):
        for key, value in raw_confidence.items():
            number = _as_optional_float(value)
            if number is not None:
                confidence[str(key)] = max(0.0, min(1.0, number))
    return confidence


def _normalize_llm_intent(
    raw_intent: dict[str, Any],
    mock_intent: dict[str, Any],
    user_input: str,
) -> dict[str, Any]:
    """Coerce an LLM response into the intent schema."""

    intent = dict(mock_intent)

    task_type = str(raw_intent.get("task_type") or intent["task_type"]).strip()
    intent["task_type"] = task_type if task_type in {"local_life_plan", "clarify_request"} else intent["task_type"]
    intent["goal"] = _as_optional_str(raw_intent.get("goal"), intent.get("goal")) or intent["goal"]
    intent["scene"] = _normalize_scene(raw_intent.get("scene"), intent["scene"])
    if intent["scene"] == "solo" and mock_intent.get("scene") == "friends":
        intent["scene"] = "friends"
    if (
        intent["scene"] == "friends"
        and mock_intent.get("scene") == "couple"
        and not _has_current_friend_companion_signal(user_input)
    ):
        intent["scene"] = "couple"

    raw_time = raw_intent.get("time") if isinstance(raw_intent.get("time"), dict) else {}
    baseline_time = intent.get("time", {}) or {}
    start_time = _normalize_clock_time(
        raw_time.get("start_time") or raw_intent.get("start_time"),
        baseline_time.get("start_time"),
    )
    end_time = _normalize_clock_time(
        raw_time.get("end_time") or raw_intent.get("end_time"),
        baseline_time.get("end_time"),
    )
    duration_range = _duration_range(
        raw_time.get("duration_range")
        or raw_time.get("duration_hours")
        or raw_intent.get("duration_range")
        or raw_intent.get("duration_hours"),
        baseline_time.get("duration_range", [3, 6]),
    )
    separated_weekend_duration = _extract_separated_weekend_duration(
        user_input,
        start_time,
        end_time,
    )
    if separated_weekend_duration is not None:
        duration_range = [separated_weekend_duration, separated_weekend_duration]
    elif (
        raw_time.get("duration_range") in (None, "")
        and raw_time.get("duration_hours") in (None, "")
        and raw_intent.get("duration_range") in (None, "")
        and raw_intent.get("duration_hours") in (None, "")
    ):
        duration_range = _duration_range_from_times(start_time, end_time, duration_range)
    intent["time"] = {
        "window": _as_optional_str(
            raw_time.get("window") or raw_intent.get("time_window"),
            baseline_time.get("window"),
        )
        or "unspecified",
        "duration_range": duration_range,
        "start_time": start_time,
        "end_time": end_time,
    }
    if (
        _has_day_level_plan(user_input)
        and start_time
        and end_time
        and _duration_hours_from_times(start_time, end_time) is not None
        and _duration_hours_from_times(start_time, end_time) <= 2
        and _contains_any(user_input, REST_WINDOW_TERMS)
    ):
        deterministic_time = parse_intent(user_input).get("time", {})
        deterministic_window = deterministic_time.get("window")
        if deterministic_window not in {None, "", "unspecified"}:
            intent["time"] = {
                "window": deterministic_window,
                "duration_range": deterministic_time.get("duration_range", intent["time"]["duration_range"]),
                "start_time": deterministic_time.get("start_time"),
                "end_time": deterministic_time.get("end_time"),
            }

    raw_location = raw_intent.get("location") if isinstance(raw_intent.get("location"), dict) else {}
    baseline_location = intent.get("location", {}) or {}
    route_mode = _as_optional_str(
        raw_location.get("route_mode") or raw_location.get("transport_mode"),
        baseline_location.get("route_mode") or baseline_location.get("transport_mode"),
    )
    if route_mode not in {"driving", "walking", "transit", "bicycling", "unknown"}:
        route_mode = baseline_location.get("route_mode") or baseline_location.get("transport_mode") or "unknown"
    detected_city_fields = _extract_trip_city_fields(user_input)
    current_city = (
        _normalize_city_name(raw_location.get("current_city"))
        or _normalize_city_name(baseline_location.get("current_city"))
        or detected_city_fields.get("current_city")
    )
    trip_city = (
        _normalize_city_name(raw_location.get("trip_city"))
        or _normalize_city_name(raw_location.get("destination_city"))
        or _normalize_city_name(baseline_location.get("trip_city"))
        or _normalize_city_name(baseline_location.get("destination_city"))
        or detected_city_fields.get("trip_city")
    )
    destination_city = (
        _normalize_city_name(raw_location.get("destination_city"))
        or trip_city
        or _normalize_city_name(baseline_location.get("destination_city"))
        or detected_city_fields.get("destination_city")
    )
    if (
        detected_city_fields.get("destination_city")
        and detected_city_fields.get("destination_city") != current_city
    ):
        destination_city = detected_city_fields.get("destination_city")
        trip_city = destination_city
    city = (
        destination_city
        or trip_city
        or _normalize_city_name(raw_location.get("city"))
        or _normalize_city_name(baseline_location.get("city"))
    )
    route_origin = _as_optional_str(
        raw_location.get("route_origin"),
        baseline_location.get("route_origin"),
    )
    if route_origin and current_city and route_origin.rstrip("市") == current_city and "," not in route_origin:
        route_origin = None
    intent["location"] = {
        "origin": _as_optional_str(raw_location.get("origin"), baseline_location.get("origin")) or "unknown",
        "current_location": _as_optional_str(
            raw_location.get("current_location"),
            baseline_location.get("current_location"),
        ),
        "route_origin": route_origin,
        "distance_preference": _as_optional_str(
            raw_location.get("distance_preference"),
            baseline_location.get("distance_preference"),
        )
        or "unknown",
        "max_distance_km": _as_optional_float(
            raw_location.get("max_distance_km"),
            baseline_location.get("max_distance_km"),
        ),
        "transport_mode": route_mode,
        "route_mode": route_mode,
        "city": city,
        "current_city": current_city,
        "trip_city": trip_city,
        "destination_city": destination_city,
    }

    raw_budget = raw_intent.get("budget") if isinstance(raw_intent.get("budget"), dict) else {}
    baseline_budget = intent.get("budget", {}) or {}
    raw_budget_amount = (
        raw_budget.get("amount")
        if raw_budget.get("amount") not in (None, "")
        else raw_budget.get("max_amount")
        if raw_budget.get("max_amount") not in (None, "")
        else raw_budget.get("upper_bound")
        if raw_budget.get("upper_bound") not in (None, "")
        else raw_budget.get("total_amount")
        if raw_budget.get("total_amount") not in (None, "")
        else raw_budget.get("per_person_amount")
    )
    budget_type = _as_optional_str(raw_budget.get("type"), baseline_budget.get("type"))
    budget_type_aliases = {
        "总预算": "total",
        "总价": "total",
        "全部": "total",
        "total_budget": "total",
        "人均": "per_person",
        "每人": "per_person",
        "单人": "per_person",
        "per_person_budget": "per_person",
    }
    budget_type = budget_type_aliases.get(budget_type or "", budget_type)
    if budget_type is None and raw_budget.get("per_person_amount") not in (None, ""):
        budget_type = "per_person"
    if budget_type is None and raw_budget.get("total_amount") not in (None, ""):
        budget_type = "total"
    if budget_type not in {"total", "per_person", None}:
        budget_type = baseline_budget.get("type")
    intent["budget"] = {
        "amount": _as_optional_int(raw_budget_amount, baseline_budget.get("amount")),
        "type": budget_type,
        "sensitivity": _as_optional_str(raw_budget.get("sensitivity"), baseline_budget.get("sensitivity"))
        or "unknown",
    }

    raw_preferences = (
        raw_intent.get("planning_preferences")
        if isinstance(raw_intent.get("planning_preferences"), dict)
        else {}
    )
    baseline_preferences = intent.get("planning_preferences", {}) or {}
    preference_keys = (
        "activity_type",
        "food_type",
        "emotion_type",
        "atmosphere_type",
        "experience_type",
        "restaurant_type",
    )
    planning_preferences = {}
    for key in preference_keys:
        planning_preferences[key] = _merge_chinese_tags(
            baseline_preferences.get(key, []),
            raw_preferences.get(key, []),
        )
    planning_preferences["pace"] = _as_optional_str(
        raw_preferences.get("pace"),
        baseline_preferences.get("pace"),
    ) or "relaxed"
    intent["planning_preferences"] = planning_preferences

    raw_constraints = raw_intent.get("constraints") if isinstance(raw_intent.get("constraints"), dict) else {}
    baseline_constraints = intent.get("constraints", {}) or {}
    intent["constraints"] = _normalize_constraint_buckets(
        _merge_chinese_tags(
            baseline_constraints.get("hard", []),
            raw_constraints.get("hard", raw_constraints.get("hard_tags", [])),
        ),
        _merge_chinese_tags(
            baseline_constraints.get("soft", []),
            raw_constraints.get("soft", raw_constraints.get("soft_tags", [])),
        ),
        _merge_chinese_tags(
            baseline_constraints.get("avoid", []),
            raw_constraints.get("avoid", raw_constraints.get("avoid_tags", [])),
        ),
    )

    intent["people"] = _normalize_people(
        raw_intent.get("people"),
        intent.get("people", []),
        current_text=user_input,
    )
    explicit_people_count = _extract_people_count(user_input)
    people_count = (
        explicit_people_count
        if explicit_people_count is not None
        else _as_optional_int(raw_intent.get("people_count"), intent.get("people_count"))
    )
    intent["people_count"] = max(0, people_count or 0)
    intent["ritual_need"] = bool(raw_intent.get("ritual_need", intent.get("ritual_need", False)))
    intent["emotion_need"] = _merge_chinese_tags(
        intent.get("emotion_need", []),
        raw_intent.get("emotion_need", []),
    )
    if "missing_slots" in raw_intent:
        intent["missing_slots"] = _as_str_list(raw_intent.get("missing_slots"))
    else:
        intent["missing_slots"] = _as_str_list(intent.get("missing_slots"))
    intent["confidence"] = _normalize_confidence(raw_intent.get("confidence"), intent.get("confidence", {}))
    intent["raw_text"] = _as_optional_str(raw_intent.get("raw_text"), user_input) or user_input
    intent["parse_source"] = "llm"
    return _enrich_intent_with_explicit_constraints(intent, user_input)


def maybe_parse_intent_with_llm(
    user_input: str,
    mock_intent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Parse intent with the primary LLM path when configured."""

    if not is_a_llm_enabled():
        return mock_intent, None

    config = load_a_llm_config()
    if config is None:
        return mock_intent, {
            "enabled": True,
            "provider": A_LLM_PROVIDER,
            "api_format": A_LLM_API_FORMAT,
            "success": False,
            "fallback": True,
            "reason": "missing_api_key",
        }

    messages = build_llm_intent_messages(user_input, mock_intent)
    try:
        response = chat_completion(messages, config=config)
        raw_intent = _parse_jsonish(response["content"])
        if not raw_intent:
            raise ValueError("A-stage LLM response did not contain a JSON object")
        intent = _normalize_llm_intent(raw_intent, mock_intent, user_input)
    except Exception as exc:
        return mock_intent, {
            "enabled": True,
            "provider": A_LLM_PROVIDER,
            "api_format": A_LLM_API_FORMAT,
            "model": config.model,
            "base_url": config.base_url,
            "success": False,
            "fallback": True,
            "error_type": type(exc).__name__,
            "error": _sanitize_a_llm_error(exc)[:300],
        }

    return intent, {
        "enabled": True,
        "provider": A_LLM_PROVIDER,
        "api_format": A_LLM_API_FORMAT,
        "base_url": config.base_url,
        "model": response.get("model") or config.model,
        "success": True,
        "fallback": False,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
    }


def parse_intent(user_input: str) -> dict[str, Any]:
    """Parse a natural language local-life request into a structured intent."""

    text = normalize_user_input(user_input)
    if not text:
        return {
            "task_type": "clarify_request",
            "goal": "等待用户提供本地生活需求",
            "scene": "unknown",
            "time": {
                "window": "unspecified",
                "duration_range": [0, 0],
                "start_time": None,
                "end_time": None,
            },
            "people": [],
            "location": {
                "origin": "unknown",
                "current_location": None,
                "route_origin": None,
                "distance_preference": "unknown",
                "max_distance_km": None,
                "transport_mode": "unknown",
                "route_mode": "unknown",
                "city": None,
                "current_city": None,
                "trip_city": None,
                "destination_city": None,
            },
            "budget": {
                "amount": None,
                "type": None,
                "sensitivity": "unknown",
            },
            "planning_preferences": {
                "activity_type": [],
                "food_type": [],
                "emotion_type": [],
                "atmosphere_type": [],
                "experience_type": [],
                "restaurant_type": [],
                "pace": "unknown",
            },
            "constraints": {
                "hard": [],
                "soft": [],
                "avoid": [],
            },
            "people_count": 0,
            "missing_slots": ["user_input"],
            "confidence": {"user_input": 0.0},
            "raw_text": "",
        }

    child_age = _extract_age(text)
    explicit_people_count = _extract_people_count(text)
    people: list[dict[str, Any]] = [{"role": "self", "needs": []}]
    avoid: list[str] = []
    hard_tags: list[str] = []
    soft_tags: list[str] = []
    activity_type: list[str] = []
    food_type: list[str] = []
    emotion_type: list[str] = []
    atmosphere_type: list[str] = []
    experience_type: list[str] = []
    restaurant_type: list[str] = []
    confidence: dict[str, float] = {}
    text_tags = tags_from_text(text)
    negated_food_groups = _negated_restaurant_groups(text)
    negated_activity_groups = _negated_activity_groups(text)
    text_tags = _remove_group_tags(text_tags, negated_food_groups)
    text_tags = _remove_group_tags(text_tags, negated_activity_groups)
    if not _has_current_child_companion_signal(text):
        text_tags = _without_tags(text_tags, {"parent_child", "kid_friendly", "low_age_child"})
    if _has_negated_diet_preference(text):
        text_tags = _without_tags(text_tags, {"low_calorie", "light_food", "healthy"})
    text_groups = tags_by_category(text_tags)
    _extend_unique(activity_type, text_groups["activity"])
    _extend_unique(food_type, text_groups["food"])
    _extend_unique(emotion_type, text_groups["emotion"])
    _extend_unique(
        atmosphere_type,
        [
            tag
            for tag in text_groups["emotion"]
            if tag in {"quiet", "atmosphere", "romantic"}
        ],
    )
    _extend_unique(
        restaurant_type,
        [
            tag
            for tag in text_groups["food"]
            if tag in {"dine_in", "takeaway_only", "japanese", "hotpot", "bbq"}
        ],
    )
    _extend_unique(
        experience_type,
        [
            tag
            for tag in text_tags
            if tag in {"hands_on_parent_child", "local_discovery", "local_culture"}
        ],
    )

    spouse_present = _has_current_adult_companion_signal(text, SPOUSE_COMPANION_TERMS)
    partner_present = _has_current_adult_companion_signal(text, PARTNER_COMPANION_TERMS)
    parent_present = _has_current_parent_companion_signal(text)
    friends_present = _has_current_friend_companion_signal(text)
    couple_present = partner_present

    if spouse_present:
        wife_needs = []
        wife_state = None
        if not _has_negated_diet_preference(text) and _contains_any(
            text, ("减肥", "减脂", "控卡", "低脂", "少油", "低卡", "清淡")
        ):
            wife_state = "dieting"
            wife_needs.extend(["低卡", "轻食"])
            _extend_unique(soft_tags, ["low_calorie", "light_food"])
            _extend_unique(food_type, ["low_calorie", "light_food"])
            confidence["wife_dieting"] = 0.9
        people.append(
            {
                "role": "wife",
                "state": wife_state,
                "needs": wife_needs or ["comfortable"],
            }
        )

    if partner_present and not spouse_present:
        people.append(
            {
                "role": "partner",
                "needs": ["comfortable", "atmosphere"],
            }
        )
        _extend_unique(activity_type, ["date_activity"])
        _extend_unique(soft_tags, ["romantic", "atmosphere"])
        _extend_unique(atmosphere_type, ["romantic", "atmosphere"])
        confidence["couple"] = 0.82

    if friends_present:
        people.append(
            {
                "role": "friends",
                "count": max(1, (explicit_people_count or 2) - 1),
                "needs": ["group_friendly", "social"],
            }
        )
        _extend_unique(activity_type, ["group_activity"])
        _extend_unique(soft_tags, ["group_friendly", "social"])
        confidence["friends"] = 0.85

    if _has_current_child_companion_signal(text):
        child_needs = ["儿童友好"]
        if "parent_child" not in negated_activity_groups:
            _extend_unique(activity_type, ["parent_child"])
        if child_age is not None and child_age <= 6:
            child_needs.append("低强度")
            _extend_unique(hard_tags, ["kid_friendly"])
            _extend_unique(soft_tags, ["low_intensity"])
            _extend_unique(activity_type, ["light_activity"])
            confidence["child_age"] = 0.95
        people.append({"role": "child", "age": child_age, "needs": child_needs})

    if any(
        word in text
        for word in ("别太远", "别离家太远", "不太远", "附近", "近一点", "离家近")
    ):
        distance_preference = "nearby"
        max_distance_km = 8.0
        _extend_unique(soft_tags, ["nearby"])
        _extend_unique(avoid, ["too_far"])
    else:
        distance_preference = "flexible"
        max_distance_km = 15.0

    if _contains_any(text, ("跨区也可以", "远一点也行", "跨区")):
        distance_preference = "cross_area_ok"
        max_distance_km = 20.0
        _extend_unique(soft_tags, ["cross_area_ok"])

    range_start_time, range_end_time, explicit_duration = _extract_time_range(text)
    post_existing_service_start_time = _extract_post_existing_service_start_time(text)
    start_time = (
        post_existing_service_start_time
        or range_start_time
        or _extract_anchored_start_time(text)
        or _extract_start_time(text)
    )
    deadline_time = _extract_deadline_time(text)
    if post_existing_service_start_time and deadline_time:
        range_end_time = None
        explicit_duration = _duration_hours_from_times(post_existing_service_start_time, deadline_time)
    cross_weekday_span = _has_cross_weekday_span(text)
    if (
        deadline_time
        and range_end_time
        and cross_weekday_span
        and not _cross_weekday_range_end_has_explicit_clock(text)
    ):
        end_time = deadline_time
        explicit_duration = _duration_hours_from_times(start_time, end_time) or explicit_duration
    else:
        end_time = range_end_time or deadline_time
    if cross_weekday_span:
        time_window = "weekend"
        duration_range = [12, 36]
        confidence["time_window"] = 0.82
    elif _has_day_level_plan(text):
        if "明天" in text:
            time_window = "tomorrow_full_day"
        elif "今天" in text:
            time_window = "today_full_day"
        else:
            time_window = "full_day"
        duration_range = [6, 10]
        confidence["time_window"] = 0.82
    elif _contains_any(text, ("下午", "午后")):
        if "明天" in text:
            time_window = "tomorrow_afternoon"
        elif "今天" in text:
            time_window = "today_afternoon"
        else:
            time_window = "afternoon"
        duration_range = [4, 6]
        confidence["time_window"] = 0.85 if "今天" in text else 0.7
    elif "后天" in text:
        time_window = "day_after_tomorrow"
        duration_range = [3, 6]
        confidence["time_window"] = 0.72
    elif "明天" in text:
        time_window = "tomorrow"
        duration_range = [3, 6]
        confidence["time_window"] = 0.72
    elif "今天" in text:
        time_window = "today"
        duration_range = [3, 6]
        confidence["time_window"] = 0.72
    elif _contains_any(text, ("周末", "周六", "周日", "周天", "星期六", "星期天", "星期日")):
        time_window = "weekend"
        duration_range = [4, 8]
        confidence["time_window"] = 0.8
    elif _contains_any(text, ("晚上", "今晚", "明晚")):
        time_window = "tomorrow_night" if _contains_any(text, ("明天晚上", "明晚")) else "tonight"
        duration_range = [2, 4]
        confidence["time_window"] = 0.85
    else:
        time_window = "unspecified"
        duration_range = [3, 6]
        confidence["time_window"] = 0.35

    if explicit_duration is not None:
        if explicit_duration == int(explicit_duration):
            duration_range = [int(explicit_duration), int(explicit_duration)]
        else:
            duration_range = [explicit_duration, explicit_duration]
    else:
        separated_weekend_duration = _extract_separated_weekend_duration(
            text,
            start_time,
            end_time,
        )
        if separated_weekend_duration is not None:
            time_window = "weekend"
            if separated_weekend_duration == int(separated_weekend_duration):
                duration_range = [int(separated_weekend_duration), int(separated_weekend_duration)]
            else:
                duration_range = [separated_weekend_duration, separated_weekend_duration]

    if _contains_any(text, ("排队", "等位", "人多")):
        _extend_unique(avoid, ["long_queue", "crowded"])
    else:
        _extend_unique(avoid, ["long_queue"])

    if _contains_any(text, ("商场", "人挤", "人少点")):
        _extend_unique(avoid, ["crowded_mall"])

    if _contains_any(text, ("不要大油", "不想高热量", "高热量", "大油")):
        _extend_unique(avoid, ["high_calorie"])

    _extend_unique(avoid, sorted(negated_food_groups))
    _extend_unique(avoid, sorted(negated_activity_groups))

    if _contains_any(text, ("外带", "打包")) and _contains_any(
        text, ("不要外带", "只要堂食")
    ):
        _extend_unique(avoid, ["takeaway_only"])

    if _contains_any(text, ("踩雷", "靠谱", "评价")):
        _extend_unique(avoid, ["few_reviews", "new_merchant"])

    budget, budget_type = _extract_budget(text)
    low_budget_request = any(
        word in text for word in ("省钱", "便宜", "预算有限", "平价", "低预算")
    )
    if low_budget_request:
        _extend_unique(activity_type, ["budget"])
        _extend_unique(soft_tags, ["budget", "low_budget", "value_for_money"])

    if budget_type == "per_person":
        _extend_unique(soft_tags, ["per_person_budget"])
    elif budget_type == "total":
        _extend_unique(soft_tags, ["total_budget"])

    route_mode = _extract_route_mode(text)
    if route_mode != "unknown":
        _extend_unique(soft_tags, [route_mode])

    route_origin = None
    location_origin = _extract_location_origin(text)
    if re.fullmatch(r"\d{2,3}\.\d+,\d{1,2}\.\d+", location_origin):
        route_origin = location_origin

    city_fields = _extract_trip_city_fields(text)
    city = city_fields.get("city")
    if "上海" in text or city is None:
        city = city or "上海"
    destination_city = city_fields.get("destination_city")
    if destination_city is None and city != city_fields.get("current_city"):
        destination_city = city
    trip_city = city_fields.get("trip_city") or destination_city or city

    ritual_need = "ritual" in text_tags
    if ritual_need:
        _extend_unique(soft_tags, ["ritual"])

    for tag in text_tags:
        category = tags_by_category([tag])
        if category["risk"]:
            continue
        if tag in {"dine_in"}:
            _extend_unique(hard_tags, [tag])
        elif tag in {"takeaway_only"}:
            _extend_unique(soft_tags, [tag])
        else:
            _extend_unique(soft_tags, [tag])

    hard_tags = _remove_group_tags(hard_tags, negated_food_groups)
    soft_tags = _remove_group_tags(soft_tags, negated_food_groups)
    food_type = _remove_group_tags(food_type, negated_food_groups)
    restaurant_type = _remove_group_tags(restaurant_type, negated_food_groups)

    has_child = any(item["role"] == "child" for item in people)
    if has_child or parent_present:
        scene = "family"
    elif friends_present:
        scene = "friends"
    elif spouse_present or couple_present:
        scene = "couple"
    elif low_budget_request:
        scene = "low_budget"
    else:
        scene = "solo"

    people_count = explicit_people_count or len(people)
    confidence["scene"] = 0.92 if scene == "family" else 0.7
    confidence["distance"] = 0.85 if distance_preference == "nearby" else 0.45

    origin = location_origin
    missing_slots = []
    if budget is None:
        missing_slots.append("budget")
    if origin == "unknown":
        missing_slots.append("exact_origin")
    if route_mode == "unknown":
        missing_slots.append("transport_mode")

    display_activity_type = to_chinese_tags(activity_type) or ["轻量活动"]
    if (
        ("group_activity" in activity_type or "多人活动" in display_activity_type)
        and "group_activity" not in display_activity_type
    ):
        display_activity_type.append("group_activity")

    intent = {
        "task_type": "local_life_plan",
        "goal": "安排一次本地生活出行计划",
        "scene": scene,
        "time": {
            "window": time_window,
            "duration_range": duration_range,
            "start_time": start_time,
            "end_time": end_time,
        },
        "people": people,
        "location": {
            "origin": origin,
            "current_location": None,
            "route_origin": route_origin,
            "distance_preference": distance_preference,
            "max_distance_km": max_distance_km,
            "transport_mode": route_mode,
            "route_mode": route_mode,
            "city": city,
            "current_city": city_fields.get("current_city"),
            "trip_city": trip_city,
            "destination_city": destination_city,
        },
        "budget": {
            "amount": budget,
            "type": budget_type,
            "sensitivity": (
                "high"
                if low_budget_request or (budget is not None and budget <= 300)
                else "unknown"
            ),
        },
        "planning_preferences": {
            "activity_type": display_activity_type,
            "food_type": to_chinese_tags(food_type),
            "emotion_type": to_chinese_tags(emotion_type),
            "atmosphere_type": to_chinese_tags(atmosphere_type),
            "experience_type": to_chinese_tags(experience_type),
            "restaurant_type": to_chinese_tags(restaurant_type),
            "pace": "relaxed",
        },
        "constraints": {
            "hard": to_chinese_tags(hard_tags),
            "soft": to_chinese_tags(soft_tags),
            "avoid": to_chinese_tags(avoid),
        },
        "people_count": people_count,
        "ritual_need": ritual_need,
        "emotion_need": to_chinese_tags(emotion_type),
        "missing_slots": missing_slots,
        "confidence": confidence,
        "raw_text": text,
    }
    return _enrich_intent_with_explicit_constraints(intent, text)


def _infer_sequence_preference(raw_text: str) -> str:
    text = str(raw_text or "")
    if any(token in text for token in ("吃完", "饭后", "餐后", "用餐后", "吃完饭", "吃完火锅")):
        return "restaurant_then_activity"
    if re.search(
        r"先.{0,24}(?:吃|晚饭|午饭|餐厅|饭)"
        r".{0,50}(?:再|然后|之后)"
        r".{0,30}(?:KTV|唱|演出|亲子剧|脱口秀|小剧场|看|电影|影院|清吧|酒吧|坐)",
        text,
    ):
        return "restaurant_then_activity"
    if re.search(
        r"(?:想|要|准备|打算)?[^，。；;,.]{0,12}(?:吃|晚饭|午饭|餐厅|饭)"
        r"[^。；;.]{0,80}(?:再|然后|之后)"
        r"[^，。；;,.]{0,30}(?:看|电影|影院|清吧|酒吧|坐)",
        text,
    ):
        return "restaurant_then_activity"
    return "activity_then_restaurant"


def constraints_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Flatten the intent into fields expected by downstream planning modules."""

    companions = [item for item in intent["people"] if item["role"] != "self"]
    child = next((item for item in companions if item.get("role") == "child"), {})
    spouse = next(
        (item for item in companions if item.get("role") in {"wife", "partner"}),
        {},
    )
    needs = canonicalize_tags(spouse.get("needs", []))
    mom_diet = (
        "low_calorie"
        if spouse.get("state") == "dieting"
        or "low_calorie" in needs
        or "light_food" in needs
        else None
    )
    location = intent.get("location", {}) or {}
    budget = intent.get("budget", {}) or {}
    time_info = intent.get("time", {}) or {}
    explicit_constraints = intent.get("explicit_constraints", {}) or {}
    avoid = intent["constraints"]["avoid"]
    normalized_avoid = canonicalize_tags(avoid)
    max_queue_time_min = 15 if "long_queue" in normalized_avoid else None
    destination_city = location.get("destination_city") or location.get("trip_city") or location.get("city")
    trip_city = location.get("trip_city") or destination_city
    current_city = location.get("current_city")
    location_payload = {
        "origin": location["origin"],
        "current_location": location.get("current_location"),
        "route_origin": location.get("route_origin"),
        "city": destination_city,
        "current_city": current_city,
        "trip_city": trip_city,
        "destination_city": destination_city,
    }

    return {
        "task_type": intent["task_type"],
        "scene": intent["scene"],
        "time_window": time_info["window"],
        "duration_range": time_info["duration_range"],
        "start_time": time_info.get("start_time"),
        "end_time": time_info.get("end_time"),
        "companions": companions,
        "people_count": intent["people_count"],
        "child_age": child.get("age"),
        "mom_diet": mom_diet,
        "origin": location["origin"],
        "current_location": location.get("current_location"),
        "route_origin": location.get("route_origin"),
        "location": location_payload,
        "distance_preference": location["distance_preference"],
        "max_distance_km": location["max_distance_km"],
        "transport_mode": location["transport_mode"],
        "route_mode": location.get("route_mode", location["transport_mode"]),
        "city": destination_city,
        "current_city": current_city,
        "trip_city": trip_city,
        "destination_city": destination_city,
        "budget": budget["amount"],
        "budget_type": budget.get("type"),
        "budget_sensitivity": budget.get("sensitivity"),
        "max_queue_time_min": max_queue_time_min,
        "hard_tags": intent["constraints"]["hard"],
        "soft_tags": intent["constraints"]["soft"],
        "avoid": avoid,
        "planning_preferences": intent["planning_preferences"],
        "sequence_preference": _infer_sequence_preference(intent.get("raw_text", "")),
        "ritual_need": intent.get("ritual_need", False),
        "emotion_need": intent.get("emotion_need", []),
        "explicit_constraints": explicit_constraints,
        "dietary_constraints": explicit_constraints.get("dietary", []),
        "accessibility_constraints": explicit_constraints.get("accessibility", []),
        "logistics_constraints": explicit_constraints.get("logistics", []),
        "time_anchors": explicit_constraints.get("time_anchors", []),
        "missing_slots": intent["missing_slots"],
        "confidence": intent["confidence"],
        "raw_text": intent["raw_text"],
    }


def _state_user_input(state: PlanState) -> str:
    user_input = normalize_user_input(state.get("user_input"))
    if user_input:
        return user_input
    for fallback_key in ("messages", "input", "query"):
        user_input = normalize_user_input(state.get(fallback_key))
        if user_input:
            return user_input
    return ""


def _intent_parser_log_message(llm_metadata: dict[str, Any] | None) -> str:
    if not llm_metadata:
        return "[intent_parser] parsed user input into structured intent"
    if llm_metadata.get("success"):
        return "[intent_parser] parsed user input with LongCat OpenAI-format LLM"
    return "[intent_parser] used deterministic parser after LongCat fallback"


def intent_parser_node(state: PlanState) -> dict[str, Any]:
    user_input = _state_user_input(state)
    prompt = build_intent_prompt(user_input)
    mock_intent = _mock_intent(user_input)
    intent, llm_metadata = maybe_parse_intent_with_llm(user_input, mock_intent)
    if llm_metadata is None or not llm_metadata.get("success"):
        intent = parse_intent(user_input)
        intent["parse_source"] = "mock"
    constraints = constraints_from_intent(intent)
    tool_results = merge_tool_results(state, "intent_parser_prompt", prompt)
    result = {
        "user_input": user_input,
        "intent": intent,
        "constraints": constraints,
        "scene_type": intent["scene"],
        "need_confirm": intent["task_type"] == "clarify_request",
        "tool_results": tool_results,
        "execution_log": append_log(
            state,
            _intent_parser_log_message(llm_metadata),
        ),
    }
    if llm_metadata:
        result["a_llm_intent"] = llm_metadata
    return result
