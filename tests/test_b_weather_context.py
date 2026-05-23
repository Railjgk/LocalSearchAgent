from src.nodes import weather_client
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.weather_client import classify_weather, get_weather_context


def test_weather_context_falls_back_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GAODE_API_KEY", raising=False)
    monkeypatch.delenv("GAODE_WEATHER_API_KEY", raising=False)

    context = get_weather_context({"city": "上海"})

    assert context["available"] is False
    assert context["source"] == "missing_api_key"
    assert context["adcode"] == "310000"


def test_classify_rainy_weather_prefers_indoor() -> None:
    context = classify_weather("小雨", temperature="24", windpower="≤3")

    assert "rainy" in context["condition_tags"]
    assert "rain" in context["risk_tags"]
    assert context["prefer_indoor"] is True


def test_get_weather_context_uses_c_weather_forecaster(monkeypatch) -> None:
    class FakeForecaster:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_weather(self, city, extensions="base"):
            assert self.api_key == "test-key"
            assert city == "310000"
            assert extensions == "base"
            return {
                "feasible": True,
                "city": "上海市",
                "adcode": "310000",
                "weather": "小雨",
                "temperature": "24",
                "winddirection": "东",
                "windpower": "≤3",
                "humidity": "78",
                "reporttime": "2026-05-22 10:00:00",
            }

    weather_client._fetch_weather_cached.cache_clear()
    monkeypatch.setenv("GAODE_API_KEY", "test-key")
    monkeypatch.setattr(weather_client, "WeatherForecaster", FakeForecaster)

    context = get_weather_context({"city": "上海"})

    assert context["available"] is True
    assert context["source"] == "weather_forecaster"
    assert context["weather"] == "小雨"
    assert context["temperature"] == 24
    assert context["prefer_indoor"] is True


def test_candidate_generator_uses_existing_weather_context(monkeypatch) -> None:
    monkeypatch.delenv("GAODE_API_KEY", raising=False)
    state = {
        "scene_type": "friends",
        "constraints": {
            "scene": "friends",
            "people_count": 4,
            "budget": 600,
            "max_distance_km": 12,
            "max_queue_time_min": 60,
            "duration_range_min": [180, 420],
        },
        "scenario_activities": ["citywalk", "local_market"],
        "weather_context": {
            "available": True,
            "source": "test_weather",
            "weather": "小雨",
            "condition_tags": ["rainy"],
            "risk_tags": ["rain"],
            "prefer_indoor": True,
            "outdoor_caution": True,
        },
        "execution_log": [],
    }

    result = candidate_generator_node(state)

    assert result["weather_context"]["source"] == "test_weather"
    assert result["candidates"]
    assert all(plan.get("weather_context", {}).get("available") for plan in result["candidates"])


def test_plan_optimizer_adds_weather_objective_and_risk() -> None:
    weather_context = {
        "available": True,
        "source": "test_weather",
        "weather": "小雨",
        "condition_tags": ["rainy"],
        "risk_tags": ["rain"],
        "prefer_indoor": True,
        "outdoor_caution": True,
    }
    plan = {
        "plan_id": "cand_weather_outdoor",
        "scene_type": "friends",
        "nodes": [
            {
                "poi_id": "act_outdoor",
                "type": "activity",
                "name": "Outdoor Citywalk",
                "category": "citywalk",
                "tags": ["citywalk", "outdoor"],
                "price": 80,
                "duration_min": 90,
                "rating": 4.6,
                "weather_sensitivity": "high",
                "available_slots": [{"time": "14:00"}],
                "product_ids": ["prod_act_outdoor"],
                "products": [{"product_id": "prod_act_outdoor", "poi_id": "act_outdoor"}],
                "deals": [],
                "merchant_id": "m_act_outdoor",
            },
            {
                "poi_id": "res_healthy",
                "type": "restaurant",
                "name": "Healthy Bistro",
                "restaurant_category": "salad_light_food",
                "tags": ["light_food", "low_calorie"],
                "price": 120,
                "duration_min": 60,
                "rating": 4.7,
                "dine_in_available": True,
                "available_slots": [{"time": "16:30"}],
                "product_ids": ["prod_res_healthy"],
                "products": [{"product_id": "prod_res_healthy", "poi_id": "res_healthy"}],
                "deals": [],
                "merchant_id": "m_res_healthy",
            },
        ],
        "route": {"total_distance_km": 2.0, "total_travel_time_min": 20, "traffic_status": "low"},
        "schedule": {"activity_start": "14:00", "activity_end": "15:30", "restaurant_start": "16:30"},
        "budget": {"total_price": 200},
        "availability": {"all_available": True, "max_queue_time_min": 5},
        "estimated_duration_min": 210,
        "tags": ["citywalk", "outdoor", "light_food"],
        "weather_context": weather_context,
    }

    result = plan_optimizer_node(
        {
            "scene_type": "friends",
            "constraints": {
                "scene": "friends",
                "people_count": 4,
                "budget": 600,
                "duration_range_min": [180, 420],
                "max_distance_km": 12,
                "max_queue_time_min": 60,
            },
            "filtered_candidates": [plan],
            "weather_context": weather_context,
            "execution_log": [],
        }
    )

    selected = result["selected_plan"]
    assert selected["objective_vector"]["weather_fit"] < 0.7
    assert selected["weather_context"]["weather"] == "小雨"
    assert any("Weather risk" in item for item in selected["risk_factors"])

