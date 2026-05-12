try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

from .mock_api_adapter import (
    fetch_activity_candidates,
    fetch_restaurant_candidates,
)
from .b_utils import (
    expand_preference_tags,
    to_float,
    parse_child_age,
    get_scene_template,
)


def _build_activity_candidates() -> list[dict]:
    return [
        {
            "poi_id": "act_001",
            "name": "亲子陶艺体验馆",
            "type": "activity",
            "tags": ["kid_friendly", "indoor", "low_intensity", "family_friendly"],
            "price": 198,
            "distance_km": 3.2,
            "duration_min": 90,
            "rating": 4.7,
            "queue_time_min": 5,
            "available": True,
            "available_slots": [{"time": "14:00"}, {"time": "15:30"}, {"time": "17:00"}],
            "location": "杨浦区大学路",
        },
        {
            "poi_id": "act_002",
            "name": "远郊探险亲子营",
            "type": "activity",
            "tags": ["outdoor", "adventure", "high_intensity"],
            "price": 250,
            "distance_km": 9.5,
            "duration_min": 120,
            "rating": 4.8,
            "queue_time_min": 20,
            "available": True,
            "available_slots": [{"time": "13:00"}, {"time": "15:00"}],
            "location": "浦东新区体育中心",
        },
        {
            "poi_id": "act_003",
            "name": "儿童室内乐园",
            "type": "activity",
            "tags": ["kid_friendly", "indoor", "playful"],
            "price": 160,
            "distance_km": 2.8,
            "duration_min": 100,
            "rating": 4.3,
            "queue_time_min": 80,
            "available": True,
            "available_slots": [{"time": "14:00"}, {"time": "16:00"}],
            "location": "闵行区万达广场",
        },
        {
            "poi_id": "act_004",
            "name": "都市密室逃脱",
            "type": "activity",
            "tags": ["indoor", "strategy", "high_intensity"],
            "price": 180,
            "distance_km": 1.5,
            "duration_min": 90,
            "rating": 4.9,
            "queue_time_min": 10,
            "available": True,
            "available_slots": [{"time": "14:00"}, {"time": "15:30"}],
            "location": "静安区商业街",
        },
    ]


def _build_restaurant_candidates() -> list[dict]:
    return [
        {
            "poi_id": "res_001",
            "name": "轻食日料餐厅",
            "type": "restaurant",
            "tags": ["low_calorie", "light_food", "family_friendly"],
            "price": 220,
            "distance_km": 1.1,
            "duration_min": 90,
            "rating": 4.6,
            "queue_time_min": 10,
            "available": True,
            "available_slots": [{"time": "17:30"}, {"time": "18:30"}, {"time": "19:30"}],
            "location": "五角场商圈",
        },
        {
            "poi_id": "res_002",
            "name": "健身餐盒餐厅",
            "type": "restaurant",
            "tags": ["low_calorie", "light_food"],
            "price": 180,
            "distance_km": 2.2,
            "duration_min": 80,
            "rating": 4.4,
            "queue_time_min": 25,
            "available": True,
            "available_slots": [{"time": "17:00"}, {"time": "18:00"}, {"time": "19:00"}],
            "location": "杨浦区友谊路",
        },
        {
            "poi_id": "res_003",
            "name": "本帮大排档",
            "type": "restaurant",
            "tags": ["high_calorie", "local_cuisine", "crowded_mall"],
            "price": 260,
            "distance_km": 3.8,
            "duration_min": 90,
            "rating": 4.8,
            "queue_time_min": 40,
            "available": True,
            "available_slots": [{"time": "17:00"}, {"time": "18:00"}, {"time": "19:00"}],
            "location": "浦东新区老街",
        },
        {
            "poi_id": "res_005",
            "name": "蔬菜沙拉店",
            "type": "restaurant",
            "tags": ["low_calorie", "light_food", "family_friendly"],
            "price": 150,
            "distance_km": 3.8,
            "duration_min": 60,
            "rating": 4.2,
            "queue_time_min": 15,
            "available": True,
            "available_slots": [{"time": "17:30"}, {"time": "18:30"}],
            "location": "浦东新区世纪大道",
        },
        {
            "poi_id": "res_004",
            "name": "附近快餐小店",
            "type": "restaurant",
            "tags": ["budget", "fast_food"],
            "price": 120,
            "distance_km": 0.8,
            "duration_min": 60,
            "rating": 3.8,
            "queue_time_min": 5,
            "available": True,
            "available_slots": [{"time": "17:00"}, {"time": "18:00"}],
            "location": "杨浦区小北路",
        },
    ]


def _slot_to_minutes(slot: str) -> int:
    if not slot or ":" not in str(slot):
        return -1
    hour, minute = str(slot).split(":", 1)
    return int(hour) * 60 + int(minute)


def _pick_time_slots(activity: dict, restaurant: dict, constraints: dict) -> tuple[str | None, str | None]:
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = _slot_to_minutes(start_time)
    activity_slots = [slot.get("time") for slot in activity.get("available_slots", []) if slot.get("time")]
    restaurant_slots = [slot.get("time") for slot in restaurant.get("available_slots", []) if slot.get("time")]

    activity_start = next(
        (slot for slot in activity_slots if _slot_to_minutes(slot) >= start_minutes),
        activity_slots[0] if activity_slots else None,
    )
    if activity_start is None:
        return None, None

    min_restaurant_minutes = _slot_to_minutes(activity_start) + int(activity.get("duration_min", 0)) + 30
    restaurant_start = next(
        (slot for slot in restaurant_slots if _slot_to_minutes(slot) >= min_restaurant_minutes),
        None,
    )

    return activity_start, restaurant_start


def _sort_candidates(
    candidates: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict = None,
    scenario_activities: list = None,
) -> list[dict]:
    user_profile = user_profile or {}
    scenario_activities = scenario_activities or []

    child_age = parse_child_age(constraints.get("child_age"))
    mom_diet = constraints.get("mom_diet")

    food_pref = user_profile.get("food_preference") or []
    if isinstance(food_pref, str):
        food_pref = [food_pref]

    preference_tags = expand_preference_tags(food_pref) + expand_preference_tags(scenario_activities)
    preference_tags = list(set(preference_tags))

    max_distance_km = to_float(constraints.get("max_distance_km"), 8.0)
    max_queue_time = to_float(constraints.get("max_queue_time"), 30.0)

    def score(item: dict) -> float:
        base = item.get("rating", 0) * 2

        if not item.get("available", True):
            base -= 12

        distance = to_float(item.get("distance_km"), 0.0)
        if distance > max_distance_km:
            base -= 6 + (distance - max_distance_km)
        else:
            base += max(0.0, (max_distance_km - distance) * 0.1)

        queue_time = to_float(item.get("queue_time_min"), 0.0)
        if queue_time > max_queue_time:
            base -= 4 + (queue_time - max_queue_time) * 0.1
        else:
            base += max(0.0, (max_queue_time - queue_time) * 0.05)

        item_type = item.get("type", "")
        tags = item.get("tags", []) or []
        normalized_tags = expand_preference_tags(tags) + tags
        normalized_tags = list(set(normalized_tags))

        if child_age is not None and child_age <= 6 and item_type == "activity":
            if "kid_friendly" in normalized_tags or "low_intensity" in normalized_tags:
                base += 6

        if mom_diet == "low_calorie" and item_type == "restaurant":
            if "low_calorie" in normalized_tags or "light_food" in normalized_tags:
                base += 5

        if scene_type == "low_budget" and item_type == "restaurant":
            price = to_float(item.get("price", 0.0), 0.0)
            if price <= 180:
                base += 4
            else:
                base -= 2

        match_count = sum(1 for tag in normalized_tags if tag in preference_tags)
        base += match_count * 2

        if item_type == "restaurant" and "budget" in preference_tags and "budget" in normalized_tags:
            base += 2

        if item_type == "activity" and "nearby" in preference_tags and distance <= max_distance_km:
            base += 1

        return base

    return sorted(candidates, key=score, reverse=True)


def _combine_plan_candidates(
    activities: list[dict],
    restaurants: list[dict],
    constraints: dict,
    scene_type: str,
) -> list[dict]:
    plan_candidates = []
    plan_index = 1
    plan_template = get_scene_template(scene_type)

    max_distance_km = to_float(constraints.get("max_distance_km"), 8.0)
    max_queue_time = to_float(constraints.get("max_queue_time"), 30.0)
    budget = to_float(constraints.get("budget"), 500.0)
    child_age_value = parse_child_age(constraints.get("child_age"))

    for activity in activities:
        for restaurant in restaurants:
            activity_start, restaurant_start = _pick_time_slots(activity, restaurant, constraints)
            if not activity_start or not restaurant_start:
                continue

            total_distance_km = round(activity["distance_km"] + restaurant["distance_km"] + 0.8, 1)
            total_travel_time_min = int(total_distance_km * 6 + 10)
            total_price = activity["price"] + restaurant["price"]
            max_queue_time_plan = max(activity["queue_time_min"], restaurant["queue_time_min"])
            available = activity["available"] and restaurant["available"]
            activity_end_minutes = _slot_to_minutes(activity_start) + int(activity["duration_min"])
            restaurant_start_minutes = _slot_to_minutes(restaurant_start)
            estimated_duration_min = max(
                240,
                restaurant_start_minutes + int(restaurant["duration_min"]) - _slot_to_minutes(activity_start),
            )
            tags = list({*activity.get("tags", []), *restaurant.get("tags", [])})

            constraint_snapshot = {
                "max_distance_km": max_distance_km,
                "max_queue_time": max_queue_time,
                "budget": budget,
                "child_age": child_age_value,
                "mom_diet": constraints.get("mom_diet"),
            }

            route_legs = [
                {
                    "from": "start_point",
                    "to": activity.get("name", "activity"),
                    "distance_km": activity.get("distance_km", 0),
                    "travel_time_min": int(activity.get("distance_km", 0) * 6 + 5),
                    "poi_id": activity.get("poi_id"),
                },
                {
                    "from": activity.get("name", "activity"),
                    "to": restaurant.get("name", "restaurant"),
                    "distance_km": restaurant.get("distance_km", 0),
                    "travel_time_min": int(restaurant.get("distance_km", 0) * 6 + 5),
                    "poi_id": restaurant.get("poi_id"),
                },
            ]

            availability_detail = {
                "activity_available": activity.get("available", True),
                "restaurant_available": restaurant.get("available", True),
                "activity_queue_min": activity.get("queue_time_min", 0),
                "restaurant_queue_min": restaurant.get("queue_time_min", 0),
                "total_queue_min": max_queue_time_plan,
            }

            if scene_type == "couple":
                min_people = 2
            elif scene_type == "friends":
                min_people = 3
            else:
                min_people = 4

            execution_requirements = {
                "min_people": min_people,
                "adult_count": 2 if scene_type == "family" else 1,
                "child_count": 1 if scene_type == "family" else 0,
                "pre_booking_required": (
                    activity.get("queue_time_min", 0) > 20
                    or restaurant.get("queue_time_min", 0) > 30
                ),
                "special_preparation": [],
            }

            if constraints.get("mom_diet") == "low_calorie":
                if "low_calorie" not in restaurant.get("tags", []):
                    execution_requirements["special_preparation"].append("提前告知餐厅低卡需求")

            if child_age_value is not None and child_age_value <= 3:
                execution_requirements["special_preparation"].append("携带儿童座椅或垫子")
                execution_requirements["special_preparation"].append("准备纸尿裤和湿巾")

            plan_candidates.append(
                {
                    "plan_id": f"cand_{plan_index:03d}",
                    "scene_type": scene_type,
                    "plan_template": plan_template,
                    "nodes": [activity, restaurant],
                    "route": {
                        "total_distance_km": total_distance_km,
                        "total_travel_time_min": total_travel_time_min,
                        "legs": route_legs,
                    },
                    "schedule": {
                        "activity_start": activity_start,
                        "activity_end": f"{activity_end_minutes // 60:02d}:{activity_end_minutes % 60:02d}",
                        "restaurant_start": restaurant_start,
                    },
                    "budget": {
                        "total_price": total_price,
                    },
                    "availability": {
                        "all_available": available,
                        "max_queue_time_min": max_queue_time_plan,
                        "detail": availability_detail,
                    },
                    "estimated_duration_min": estimated_duration_min,
                    "tags": tags,
                    "constraint_snapshot": constraint_snapshot,
                    "execution_requirements": execution_requirements,
                }
            )
            plan_index += 1

    return plan_candidates


def candidate_generator_node(state: PlanState) -> dict:
    execution_log = state.get("execution_log", [])
    if state.get("need_confirm"):
        execution_log.append(
            "[B] candidate_generator_node 等待用户补充输入，跳过候选生成"
        )
        return {
            "candidates": [],
            "execution_log": execution_log,
        }

    scene_type = state.get("scene_type", "family")
    constraints = state.get("constraints", {})
    user_profile = state.get("user_profile", {})
    scenario_activities = state.get("scenario_activities", []) or []

    activity_candidates = fetch_activity_candidates(
        constraints=constraints,
        scene_type=scene_type,
        scenario_activities=scenario_activities,
    ) or _build_activity_candidates()
    restaurant_candidates = fetch_restaurant_candidates(
        constraints=constraints,
        scene_type=scene_type,
        scenario_activities=scenario_activities,
    ) or _build_restaurant_candidates()

    selected_activities = _sort_candidates(
        activity_candidates,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
    )[:3]
    selected_restaurants = _sort_candidates(
        restaurant_candidates,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
    )[:3]

    plan_candidates = _combine_plan_candidates(selected_activities, selected_restaurants, constraints, scene_type)

    execution_log.append(
        f"[B] candidate_generator_node 生成 {len(plan_candidates)} 个 plan_candidates "
        f"(activities={len(activity_candidates)}, restaurants={len(restaurant_candidates)})"
    )

    return {
        "candidates": plan_candidates,
        "execution_log": execution_log,
    }
